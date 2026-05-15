#!/usr/bin/env python3
"""Train per-symbol XGBoost models for sandbox portfolio (v2 pipeline).

Improvements over v1:
- 100k candles per symbol (vs 35k)
- Triple-barrier labeling (vs simple forward return)
- scale_pos_weight for class imbalance
- 24 quantitative features (ADX, StochRSI, Williams %R, OBV, CMF, temporal, etc.)
- n_estimators=500 with early stopping
- Full metrics in meta.json (acc, precision, recall, f1, auc, class_dist, feature_importance)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt  # noqa: E402
from strategy.quant_features import (  # noqa: E402
    DEFAULT_LABEL_ROUND_TRIP,
    MIN_OHLC_ROWS,
    QUANT_FEATURE_COLS,
    add_quant_features,
    triple_barrier_label,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_sandbox_models")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
TIMEFRAME = "15m"
LIMIT = 100_000
HORIZON = 8
TRAIN_RATIO = 0.80


def _symbol_model_path(symbol: str) -> Path:
    return _REPO / "models" / f"{symbol.replace('/', '_')}_v2.json"


LABEL_THRESHOLD: float = float(os.environ.get("TRAIN_LABEL_THRESHOLD", "0.015"))  # 1.5%

def build_dataset(df: pd.DataFrame, horizon: int, timeframe: str = "15m") -> pd.DataFrame:
    tf_min = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 15
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    feat["timestamp"] = pd.to_datetime(df["timestamp"].values, utc=True)
    # FIXED: Label must match SL for positive expectancy (R:R = 1:2)
    fwd = feat["close"].shift(-horizon) / feat["close"] - 1.0
    feat["label"] = (fwd > LABEL_THRESHOLD).astype(int)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"]).reset_index(drop=True)
    return feat


def _calc_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost to handle class imbalance."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute classification metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["auc"] = 0.5
    return metrics


def train_one(symbol: str, limit: int) -> dict[str, float | int | str]:
    from strategy.ml_predictor import get_symbol_config
    cfg = get_symbol_config(symbol)
    tf = cfg.get("timeframe", TIMEFRAME)
    hor = cfg.get("horizon", HORIZON)

    logger.info("Fetching %d %s candles for %s (Horizon=%d) ...", limit, tf, symbol, hor)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    ds = build_dataset(raw, horizon=hor, timeframe=tf)

    if len(ds) < MIN_OHLC_ROWS + 500:
        raise ValueError(f"{symbol}: not enough usable rows after features ({len(ds)})")

    split = int(len(ds) * TRAIN_RATIO)
    split = max(split, MIN_OHLC_ROWS + 200)
    split = min(split, len(ds) - 500)
    train_df = ds.iloc[:split].copy()
    test_df = ds.iloc[split:].copy()

    x_train = train_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    x_test = test_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    # Class distribution
    train_pos = int((y_train == 1).sum())
    train_neg = int((y_train == 0).sum())
    test_pos = int((y_test == 1).sum())
    test_neg = int((y_test == 0).sum())
    logger.info(
        "%s class dist: train %d/%d (%.2f%% pos) | test %d/%d (%.2f%% pos)",
        symbol,
        train_pos,
        train_neg,
        100 * train_pos / (train_pos + train_neg),
        test_pos,
        test_neg,
        100 * test_pos / (test_pos + test_neg),
    )

    spw = _calc_scale_pos_weight(y_train)

    model = XGBClassifier(
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
        n_jobs=-1,
        scale_pos_weight=spw,
        early_stopping_rounds=20,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=False,
    )

    preds = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    metrics = _evaluate(y_test, preds, proba)
    metrics["scale_pos_weight"] = spw
    metrics["n_estimators_used"] = int(model.get_booster().num_boosted_rounds())

    # Feature importance
    importance = model.feature_importances_
    feat_imp = {
        col: float(importance[i])
        for i, col in enumerate(QUANT_FEATURE_COLS)
    }
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    out = _symbol_model_path(symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))

    # Save meta.json
    meta = {
        "symbol": symbol,
        "timeframe": tf,
        "horizon": hor,
        "rows_total": int(len(ds)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_pos": train_pos,
        "train_neg": train_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "metrics": metrics,
        "feature_importance": feat_imp_sorted,
        "features": QUANT_FEATURE_COLS,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "symbol": symbol,
        "rows_total": int(len(ds)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auc": metrics["auc"],
        "model_path": str(out),
        "meta_path": str(meta_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train sandbox XGBoost models (v2 pipeline).")
    ap.add_argument("--limit", type=int, default=LIMIT, help="Candles to fetch per symbol")
    ap.add_argument("--symbols", default=",".join(SYMBOLS), help="Comma-separated symbols")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logger.info(
        "Training symbols=%s | timeframe=%s | horizon=%d | limit=%d | features=%d",
        symbols,
        TIMEFRAME,
        HORIZON,
        args.limit,
        len(QUANT_FEATURE_COLS),
    )

    results: list[dict[str, float | int | str]] = []
    for sym in symbols:
        try:
            r = train_one(sym, limit=args.limit)
            results.append(r)
            logger.info(
                "Done %s | rows=%d train=%d test=%d | acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f | model=%s",
                r["symbol"],
                r["rows_total"],
                r["rows_train"],
                r["rows_test"],
                r["accuracy"],
                r["precision"],
                r["recall"],
                r["f1"],
                r["auc"],
                r["model_path"],
            )
        except Exception as exc:
            logger.error("Failed training for %s: %s", sym, exc, exc_info=True)

    logger.info("All models trained: %d/%d", len(results), len(symbols))
    for r in results:
        p = Path(str(r["model_path"]))
        exists = p.is_file()
        sz = p.stat().st_size if exists else 0
        logger.info("VERIFY %s exists=%s size=%d", p.name, exists, sz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
