"""
Background biweekly XGBoost re-training.

Runs every 14 days at 03:00 UTC (low-liquidity hour, minimal impact).
Fetches 50k candles per symbol from Binance, trains per-symbol models with:
  - forward_return > 1.5% label (proven learnable, AUC 0.69-0.75)
  - SPW capped at 8x (prevents probability compression)
  - horizon from SYMBOL_CONFIG per symbol

Replaces weekly_retrainer.py which had: horizon=5, round_trip=0.13%, mixed symbols.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.ml_predictor import SYMBOL_CONFIG, MODELS_DIR
from strategy.quant_features import add_quant_features, forward_return_label

_MODELS_DIR = Path(MODELS_DIR)
_RETRAIN_INTERVAL_DAYS: int = int(os.environ.get("RETRAIN_INTERVAL_DAYS", "14"))
_RETRAIN_HOUR_UTC: int = int(os.environ.get("RETRAIN_HOUR_UTC", "3"))
_RETRAIN_DATA_CANDLES: int = int(os.environ.get("RETRAIN_DATA_CANDLES", "50000"))
_RETRAIN_MIN_AUC: float = float(os.environ.get("RETRAIN_MIN_AUC", "0.60"))
_RETRAIN_MAX_SPW: float = float(os.environ.get("RETRAIN_MAX_SPW", "8.0"))
_RETRAIN_LABEL_THRESHOLD: float = float(os.environ.get("RETRAIN_LABEL_THRESHOLD", "0.015"))
_RETRAIN_ENABLED: bool = os.environ.get("RETRAIN_ENABLED", "1").strip() == "1"

FEATURES = [
    "rsi", "macd_line", "macd_signal", "macd_hist", "atr", "bb_pct_b", "bb_width",
    "vol_delta_norm", "log_ret_1", "log_ret_5", "log_ret_10", "log_ret_20",
    "close_vs_sma200_1h", "vol_rel", "adx", "stoch_rsi_k", "stoch_rsi_d",
    "williams_r", "obv", "cmf", "ret_skew_20", "ret_kurt_20", "roll_sharpe_20",
    "hour_sin", "hour_cos", "dow",
]


def _seconds_until_next_retrain() -> float:
    now = datetime.now(tz=timezone.utc)
    # Next occurrence of _RETRAIN_HOUR_UTC on any future day, rounded to interval
    candidate = now.replace(hour=_RETRAIN_HOUR_UTC, minute=0, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    # Round up to next interval boundary
    days_since_epoch = (candidate - datetime(2020, 1, 1, tzinfo=timezone.utc)).days
    days_to_add = (_RETRAIN_INTERVAL_DAYS - (days_since_epoch % _RETRAIN_INTERVAL_DAYS)) % _RETRAIN_INTERVAL_DAYS
    candidate += timedelta(days=days_to_add)
    return max(60.0, (candidate - now).total_seconds())


def _fetch_binance_paginated(sym: str, tf: str, n: int = 50000) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    tf_ms = {"15m": 15 * 60 * 1000, "30m": 30 * 60 * 1000}.get(tf, 15 * 60 * 1000)
    since = exchange.milliseconds() - n * tf_ms
    all_ohlcv: list = []
    while len(all_ohlcv) < n:
        chunk = exchange.fetch_ohlcv(sym, tf, since=since, limit=1000)
        if not chunk:
            break
        all_ohlcv.extend(chunk)
        since = chunk[-1][0] + tf_ms
        if len(chunk) < 1000:
            break
        time.sleep(0.15)
    df = pd.DataFrame(all_ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def _train_one(sym: str, cfg: dict, log: logging.Logger) -> bool:
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    tf = cfg.get("timeframe", "15m")
    horizon = cfg.get("horizon", 8)
    model_path = _MODELS_DIR / f"{sym.replace('/', '_')}_v2.json"

    log.info("[RETRAIN] %s: fetching %d candles (%s)...", sym, _RETRAIN_DATA_CANDLES, tf)
    try:
        raw = _fetch_binance_paginated(sym, tf, _RETRAIN_DATA_CANDLES)
    except Exception as exc:
        log.error("[RETRAIN] %s: fetch failed: %s", sym, exc)
        return False

    log.info("[RETRAIN] %s: %d candles fetched (%s to %s)", sym, len(raw), raw.index[0].date(), raw.index[-1].date())

    df = add_quant_features(raw)
    df["label"] = forward_return_label(df["close"], horizon=horizon, round_trip_cost=_RETRAIN_LABEL_THRESHOLD)
    df.dropna(subset=FEATURES + ["label"], inplace=True)

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    pos_rate = n_pos / len(df)
    log.info("[RETRAIN] %s: %d rows, pos=%d (%.1f%%), neg=%d", sym, len(df), n_pos, pos_rate * 100, n_neg)

    if n_pos < 100:
        log.warning("[RETRAIN] %s: too few positives (%d < 100) — SKIP", sym, n_pos)
        return False

    X = df[FEATURES].values
    y = df["label"].values
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_ho, y_ho = X[split:], y[split:]

    n_pos_tr = int(y_tr.sum())
    n_neg_tr = int((y_tr == 0).sum())
    spw = min(n_neg_tr / max(n_pos_tr, 1), _RETRAIN_MAX_SPW)

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="auc",
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], verbose=False)

    if y_ho.sum() < 5:
        log.warning("[RETRAIN] %s: holdout has <5 positives — cannot evaluate AUC", sym)
        return False

    probs_ho = model.predict_proba(X_ho)[:, 1]
    auc = float(roc_auc_score(y_ho, probs_ho))
    log.info("[RETRAIN] %s: holdout AUC=%.4f (min=%.4f)", sym, auc, _RETRAIN_MIN_AUC)

    if auc < _RETRAIN_MIN_AUC:
        log.warning("[RETRAIN] %s: AUC %.4f < %.4f — DISCARDING new model", sym, auc, _RETRAIN_MIN_AUC)
        return False

    # Backup current model
    if model_path.exists():
        bak = model_path.with_suffix(f".{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.bak")
        shutil.copy2(model_path, bak)
        log.info("[RETRAIN] %s: backed up old model to %s", sym, bak.name)

    model.save_model(str(model_path))

    fi = dict(zip(FEATURES, [round(float(v), 6) for v in model.feature_importances_]))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    meta = {
        "symbol": sym,
        "timeframe": tf,
        "horizon": horizon,
        "label": "forward_return",
        "label_threshold": _RETRAIN_LABEL_THRESHOLD,
        "max_spw": _RETRAIN_MAX_SPW,
        "actual_spw": round(float(spw), 4),
        "rows_total": int(len(df)),
        "pos_rate": round(float(pos_rate), 6),
        "holdout_auc": round(auc, 6),
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "feature_importance": fi_sorted,
    }
    meta_path = model_path.with_suffix(".meta.json").with_stem(model_path.stem)
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    # Clear cached booster so next inference loads the new model
    from strategy.ml_predictor import _booster_cache
    _booster_cache.pop(str(model_path.resolve()), None)

    log.info("[RETRAIN] %s: model saved (AUC=%.4f, SPW=%.1fx, %d rows)", sym, auc, spw, len(df))
    return True


_SHORT_HORIZON = 8  # 2h at 15m bars (fixed for all SHORT models)


def _train_short_one(sym: str, cfg: dict, log: logging.Logger) -> bool:
    """Retrain SHORT model for sym. Label: price drops > threshold in _SHORT_HORIZON bars."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    tf = cfg.get("timeframe", "15m")
    model_path = _MODELS_DIR / f"{sym.replace('/', '_')}_short_v2.json"

    log.info("[RETRAIN SHORT] %s: fetching %d candles (%s)...", sym, _RETRAIN_DATA_CANDLES, tf)
    try:
        raw = _fetch_binance_paginated(sym, tf, _RETRAIN_DATA_CANDLES)
    except Exception as exc:
        log.error("[RETRAIN SHORT] %s: fetch failed: %s", sym, exc)
        return False

    df = add_quant_features(raw)
    fwd = df["close"].shift(-_SHORT_HORIZON) / df["close"] - 1.0
    df["label"] = (fwd < -_RETRAIN_LABEL_THRESHOLD).astype(int)
    df.dropna(subset=FEATURES + ["label"], inplace=True)
    df = df.iloc[:-_SHORT_HORIZON].copy()

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    pos_rate = n_pos / len(df)
    log.info("[RETRAIN SHORT] %s: %d rows, pos=%d (%.1f%%)", sym, len(df), n_pos, pos_rate * 100)

    if n_pos < 100:
        log.warning("[RETRAIN SHORT] %s: too few positives (%d) — SKIP", sym, n_pos)
        return False

    X = df[FEATURES].values
    y = df["label"].values
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_ho, y_ho = X[split:], y[split:]

    spw = min((y_tr == 0).sum() / max(int(y_tr.sum()), 1), _RETRAIN_MAX_SPW)

    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="auc", early_stopping_rounds=30, verbosity=0, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], verbose=False)

    if y_ho.sum() < 5:
        log.warning("[RETRAIN SHORT] %s: holdout <5 positives — cannot evaluate", sym)
        return False

    auc = float(roc_auc_score(y_ho, model.predict_proba(X_ho)[:, 1]))
    log.info("[RETRAIN SHORT] %s: holdout AUC=%.4f (min=%.4f)", sym, auc, _RETRAIN_MIN_AUC)

    if auc < _RETRAIN_MIN_AUC:
        log.warning("[RETRAIN SHORT] %s: AUC %.4f < %.4f — DISCARDING", sym, auc, _RETRAIN_MIN_AUC)
        return False

    if model_path.exists():
        bak = model_path.with_suffix(f".{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.bak")
        shutil.copy2(model_path, bak)

    model.save_model(str(model_path))

    fi_sorted = dict(sorted(
        zip(FEATURES, [round(float(v), 6) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True,
    ))
    meta = {
        "symbol": sym, "direction": "short", "timeframe": tf,
        "horizon": _SHORT_HORIZON, "label": "short_forward_return",
        "label_threshold": _RETRAIN_LABEL_THRESHOLD, "max_spw": _RETRAIN_MAX_SPW,
        "actual_spw": round(float(spw), 4), "rows_total": int(len(df)),
        "pos_rate": round(float(pos_rate), 6), "holdout_auc": round(auc, 6),
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES, "feature_importance": fi_sorted,
    }
    with open(str(model_path.with_stem(model_path.stem).with_suffix(".meta.json")), "w") as f:
        json.dump(meta, f, indent=2)

    # Clear cached SHORT booster
    from strategy.ml_predictor import _short_booster_cache
    _short_booster_cache.pop(sym, None)

    log.info("[RETRAIN SHORT] %s: saved (AUC=%.4f, SPW=%.1fx)", sym, auc, spw)
    return True


async def biweekly_retrainer(watchlist: list[str]) -> None:
    log = logging.getLogger("clawdbot.retrainer")

    if not _RETRAIN_ENABLED:
        log.info("[RETRAIN] Disabled (RETRAIN_ENABLED=0)")
        return

    while True:
        wait_secs = _seconds_until_next_retrain()
        log.info(
            "[RETRAIN] Next retrain in %.1f hours (every %d days at %02d:00 UTC)",
            wait_secs / 3600, _RETRAIN_INTERVAL_DAYS, _RETRAIN_HOUR_UTC,
        )
        await asyncio.sleep(wait_secs)

        log.info("[RETRAIN] Starting biweekly retrain for %s", watchlist)
        results = {}
        for sym in watchlist:
            cfg = SYMBOL_CONFIG.get(sym)
            if cfg is None:
                log.warning("[RETRAIN] %s: no SYMBOL_CONFIG entry — skip", sym)
                continue
            try:
                ok = await asyncio.get_event_loop().run_in_executor(
                    None, _train_one, sym, cfg, log
                )
                results[sym] = "OK" if ok else "FAIL"
            except Exception as exc:
                log.error("[RETRAIN] %s: unexpected error: %s", sym, exc)
                results[sym] = "ERROR"

            try:
                short_ok = await asyncio.get_event_loop().run_in_executor(
                    None, _train_short_one, sym, cfg, log
                )
                results[sym + "_short"] = "OK" if short_ok else "FAIL/SKIP"
            except Exception as exc:
                log.error("[RETRAIN SHORT] %s: unexpected error: %s", sym, exc)
                results[sym + "_short"] = "ERROR"

        log.info("[RETRAIN] Done: %s", results)
