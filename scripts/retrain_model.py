"""XGBoost model retraining script with market-data features and VIBE injection (v2 pipeline).

Usage (24 quant features):
    python scripts/retrain_model.py --symbols BTC/USDT ETH/USDT --days 90

Usage (28 features with VIBE, saves as _v3.json):
    python scripts/retrain_model.py --symbols BTC/USDT ETH/USDT --days 90 --vibe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

logger = logging.getLogger("clawdbot.retrain")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from strategy.quant_features import (
    FINAL_FEATURE_ORDER,
    QUANT_FEATURE_COLS,
    VIBE_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
    triple_barrier_label,
    DEFAULT_LABEL_ROUND_TRIP,
)
from strategy.ml_predictor import get_symbol_config
from vibe.feature_bridge import VIBE_FEATURE_NEUTRAL

_MODELS_DIR = _ROOT / "models"

_VIBE_NEUTRAL_NOISE_STD: float = 0.05


def _build_dsn() -> str:
    """Build PostgreSQL DSN from env vars or defaults."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    user = os.environ.get("DB_USER", "clawdbot")
    password = os.environ.get("DB_PASSWORD", "clawdbot_secret")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "clawdbot")
    return f"postgres://{user}:{password}@{host}:{port}/{dbname}"


_MIN_ROWS_FOR_TRAINING = 5_000  # ~52 days of 15m data minimum


def _bars_per_day(timeframe: str) -> int:
    """Return number of bars per day for a CCXT timeframe string (15m/30m/1h/4h)."""
    tf = timeframe.strip().lower()
    if tf.endswith("m"):
        return 24 * 60 // int(tf[:-1])
    if tf.endswith("h"):
        return 24 // int(tf[:-1])
    return 96  # default 15m


def _fetch_ohlcv_ccxt(symbol: str, days: int, timeframe: str = "15m") -> pd.DataFrame | None:
    """Download OHLCV from Binance via CCXT (no auth required for public data)."""
    try:
        import ccxt
    except ImportError:
        logger.warning("ccxt not installed — skipping Binance fallback (pip install ccxt)")
        return None
    bars_needed = days * _bars_per_day(timeframe)
    limit = min(bars_needed, 1000)
    try:
        ex = ccxt.binance({"enableRateLimit": True})
        # Fetch in chunks of 1000 bars (Binance max per request)
        all_rows: list = []
        since = int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        while len(all_rows) < bars_needed:
            rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not rows:
                break
            all_rows.extend(rows)
            since = rows[-1][0] + 1
            if len(rows) < 1000:
                break
        if not all_rows:
            return None
        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        logger.info("[CCXT] Downloaded %d rows for %s from Binance", len(df), symbol)
        return df
    except Exception as exc:
        logger.warning("[CCXT] Download failed for %s: %s", symbol, exc)
        return None


async def _fetch_ohlcv(symbol: str, days: int, timeframe: str = "15m") -> pd.DataFrame | None:
    """Fetch OHLCV from TimescaleDB; falls back to Binance CCXT if DB has insufficient data."""
    dsn = _build_dsn()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM market_data WHERE symbol = $1 AND timestamp >= $2 "
        "ORDER BY timestamp ASC"
    )
    df_db: pd.DataFrame | None = None
    try:
        conn = await asyncpg.connect(dsn=dsn)
        rows = await conn.fetch(query, symbol, cutoff)
        await conn.close()
        if rows:
            df_db = pd.DataFrame([dict(r) for r in rows])
            for col in ("open", "high", "low", "close", "volume"):
                df_db[col] = pd.to_numeric(df_db[col], errors="coerce")
            df_db = df_db.dropna()
    except Exception as exc:
        logger.warning("DB fetch failed for %s: %s", symbol, exc)

    if df_db is not None and len(df_db) >= _MIN_ROWS_FOR_TRAINING:
        logger.info("[DB] %d rows for %s", len(df_db), symbol)
        return df_db

    db_rows = len(df_db) if df_db is not None else 0
    logger.warning(
        "[DB] Only %d rows for %s (need %d) — falling back to Binance CCXT download",
        db_rows, symbol, _MIN_ROWS_FOR_TRAINING,
    )
    return _fetch_ohlcv_ccxt(symbol, days, timeframe)


def _add_vibe_features(df: pd.DataFrame, add_noise: bool = True) -> pd.DataFrame:
    """Append 4 VIBE columns with neutral values + optional Gaussian noise."""
    neutral = np.array(VIBE_FEATURE_NEUTRAL, dtype=float)
    n = len(df)
    for i, col in enumerate(VIBE_FEATURE_COLS):
        base = np.full(n, neutral[i], dtype=float)
        if add_noise:
            base += np.random.normal(0.0, _VIBE_NEUTRAL_NOISE_STD, size=n)
        df[col] = base
    return df


def _calc_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def _compute_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = 0.5
    return metrics


def retrain_and_validate(
    symbol: str,
    model_path: Path,
    lookback_days: int = 90,
    min_auc: float = 0.55,
    use_vibe: bool = False,
) -> bool:
    """Retrain XGBoost v2 with TimeSeriesSplit and full metrics quality gate.

    Parameters
    ----------
    symbol: Trading pair to train on.
    model_path: Where to save the XGB JSON model.
    lookback_days: Days of OHLCV history to load.
    min_auc: Minimum mean ROC-AUC across CV splits to accept the model.
    use_vibe: If True, inject 4 VIBE features (28-dim vector) and save as _v3.json.

    Returns
    -------
    True if model was saved, False if quality gate rejected it.
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return False

    # 1. Resolve per-symbol timeframe from production cfg
    sym_cfg = get_symbol_config(symbol)
    timeframe = str(sym_cfg.get("timeframe", "15m"))
    tf_min = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 15
    if timeframe.endswith("h"):
        tf_min = int(timeframe[:-1]) * 60
    logger.info("[CFG] %s using timeframe=%s (%dmin)", symbol, timeframe, tf_min)

    # 2. Load market data on the right timeframe
    df = asyncio.run(_fetch_ohlcv(symbol, lookback_days, timeframe))
    if df is None or len(df) < 500:
        logger.warning(
            "Not enough OHLCV data for %s (need >=500 rows, got %s).",
            symbol,
            len(df) if df is not None else 0,
        )
        return False

    # 3. Compute quant features with correct base_timeframe_min (affects ATR/MACD scale)
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    # forward_return_label: model predicts whether short-term move exceeds threshold within RETRAIN_LABEL_HORIZON bars.
    # Keep horizon=8 (proven AUC 0.71+); cfg "horizon" controls live TTL not label horizon.
    label_horizon = int(os.environ.get("RETRAIN_LABEL_HORIZON", "8") or "8")
    label_thr = float(os.environ.get("RETRAIN_LABEL_THRESHOLD", "0.015"))
    feat["label"] = forward_return_label(feat["close"], horizon=label_horizon, round_trip_cost=label_thr)
    logger.info("[LABEL] %s  forward_return  horizon=%d  threshold=%.4f", symbol, label_horizon, label_thr)

    # 3. Inject VIBE features if requested
    if use_vibe:
        feat = _add_vibe_features(feat)
        feature_cols = list(FINAL_FEATURE_ORDER)
    else:
        feature_cols = list(QUANT_FEATURE_COLS)

    # Drop rows with NaN in features or label
    feat = feat.dropna(subset=feature_cols + ["label"])
    if len(feat) < 200:
        logger.warning("Not enough labelled samples for %s (need >=200, got %d).", symbol, len(feat))
        return False

    X = feat[feature_cols]
    y = feat["label"]

    # 4. TimeSeriesSplit cross-validation
    n_splits = min(5, max(2, len(X) // 200))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: list[float] = []
    cv_metrics: list[dict[str, float]] = []
    # Accumulate hold-out predictions across all folds for calibration fitting
    all_y_va: list[np.ndarray] = []
    all_probas: list[np.ndarray] = []

    for train_idx, val_idx in tscv.split(X):
        x_tr = X.iloc[train_idx].to_numpy(dtype=np.float32)
        y_tr = y.iloc[train_idx].to_numpy(dtype=np.int32)
        x_va = X.iloc[val_idx].to_numpy(dtype=np.float32)
        y_va = y.iloc[val_idx].to_numpy(dtype=np.int32)

        spw = _calc_scale_pos_weight(y_tr)
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_weight=2,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=spw,
            early_stopping_rounds=20,
        )
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        preds = model.predict(x_va)
        probas = model.predict_proba(x_va)[:, 1]
        m = _compute_metrics(y_va, preds, probas)
        cv_scores.append(m["auc"])
        cv_metrics.append(m)
        all_y_va.append(y_va)
        all_probas.append(probas)

    mean_metrics = {
        k: round(sum(m[k] for m in cv_metrics) / len(cv_metrics), 4)
        for k in ("accuracy", "precision", "recall", "f1", "auc")
    }
    mean_auc = mean_metrics["auc"]

    # 5. Load previous AUC for comparison
    old_auc = 0.0
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text())
            old_auc = float(old_meta.get("metrics", {}).get("auc", old_meta.get("auc", 0.0)))
        except Exception:
            pass

    logger.info(
        "Retrain %s (vibe=%s): old_auc=%.4f new_auc=%.4f acc=%.4f f1=%.4f (splits=%d)",
        symbol,
        use_vibe,
        old_auc,
        mean_auc,
        mean_metrics["accuracy"],
        mean_metrics["f1"],
        len(cv_scores),
    )

    # 6. Quality Gate
    if mean_auc < min_auc:
        logger.warning(
            "Model NOT saved for %s - mean AUC %.4f below min_auc %.4f.",
            symbol,
            mean_auc,
            min_auc,
        )
        return False

    if old_auc > 0.0 and mean_auc <= old_auc + 0.005:
        logger.info(
            "Model NOT saved for %s - no meaningful improvement over old_auc %.4f.",
            symbol,
            old_auc,
        )
        return False

    # 7. Final retrain on all data and save
    x_all = X.to_numpy(dtype=np.float32)
    y_all = y.to_numpy(dtype=np.int32)
    spw_all = _calc_scale_pos_weight(y_all)
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=2,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=spw_all,
    )
    final_model.fit(x_all, y_all)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.get_booster().feature_names = list(feature_cols)
    final_model.save_model(str(model_path))

    importance = final_model.feature_importances_
    feat_imp = {
        col: float(importance[i])
        for i, col in enumerate(feature_cols)
    }

    meta = {
        "metrics": mean_metrics,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": symbol,
        "samples": len(y_all),
        "splits": len(cv_scores),
        "features": feature_cols,
        "vibe_enabled": use_vibe,
        "feature_importance": dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Model UPDATED for %s → %s (AUC %.4f)", symbol, model_path, mean_auc)

    # 8. Fit isotonic calibration from accumulated hold-out predictions
    try:
        from strategy.prob_calibration import fit_and_save_calibration as _fit_cal
        cal_path = _fit_cal(
            symbol,
            np.concatenate(all_y_va),
            np.concatenate(all_probas),
        )
        logger.info("Calibration saved → %s", cal_path)
    except Exception as exc:
        logger.warning("Calibration fitting failed (non-fatal): %s", exc)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain ClawdBot XGBoost models from OHLCV history (v2)")
    parser.add_argument(
        "--symbols",
        default=os.environ.get("WATCHLIST", "BTC/USDT,ETH/USDT"),
        help="Comma-separated list of symbols to train",
    )
    parser.add_argument("--days", type=int, default=90, help="Lookback days for OHLCV")
    parser.add_argument("--min-auc", type=float, default=0.55, help="Minimum mean AUC to accept model")
    parser.add_argument("--vibe", action="store_true", help="Train with 28 features (24 base + 4 VIBE) and save as _v3.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols provided.")
        return 1

    all_ok = True
    for symbol in symbols:
        safe_name = symbol.replace("/", "_")
        suffix = "_v3" if args.vibe else "_v2"
        model_path = _MODELS_DIR / f"{safe_name}{suffix}.json"
        ok = retrain_and_validate(
            symbol=symbol,
            model_path=model_path,
            lookback_days=args.days,
            min_auc=args.min_auc,
            use_vibe=args.vibe,
        )
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
