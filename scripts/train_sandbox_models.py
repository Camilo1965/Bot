#!/usr/bin/env python3
"""Train per-symbol XGBoost models for sandbox portfolio."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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
    forward_return_label,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_sandbox_models")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
TIMEFRAME = "15m"
LIMIT = 35_000
HORIZON = 8  # Balanced default
TRAIN_RATIO = 0.80


def _symbol_model_path(symbol: str) -> Path:
    return _REPO / "models" / f"{symbol.replace('/', '_')}_v1.json"


def build_dataset(df: pd.DataFrame, horizon: int, round_trip_cost: float, timeframe: str = "15m") -> pd.DataFrame:
    tf_min = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 15
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    feat["timestamp"] = pd.to_datetime(df["timestamp"].values, utc=True)
    feat["label"] = forward_return_label(feat["close"], horizon, round_trip_cost)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"]).reset_index(drop=True)
    return feat


def train_one(symbol: str, limit: int, round_trip_cost: float) -> dict[str, float | int | str]:
    from strategy.ml_predictor import get_symbol_config
    cfg = get_symbol_config(symbol)
    tf = cfg.get("timeframe", TIMEFRAME)
    hor = cfg.get("horizon", HORIZON)
    
    logger.info("Fetching %d %s candles for %s (Horizon=%d) ...", limit, tf, symbol, hor)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    ds = build_dataset(raw, horizon=hor, round_trip_cost=round_trip_cost, timeframe=tf)

    if len(ds) < MIN_OHLC_ROWS + 200:
        raise ValueError(f"{symbol}: not enough usable rows after features ({len(ds)})")

    split = int(len(ds) * TRAIN_RATIO)
    split = max(split, MIN_OHLC_ROWS + 100)
    split = min(split, len(ds) - 200)
    train_df = ds.iloc[:split].copy()
    test_df = ds.iloc[split:].copy()

    x_train = train_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    x_test = test_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    model = XGBClassifier(
        n_estimators=240,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=2,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    proba = model.predict_proba(x_test)[:, 1]
    positives = int((proba >= 0.5).sum())

    out = _symbol_model_path(symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))

    return {
        "symbol": symbol,
        "rows_total": int(len(ds)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "accuracy": acc,
        "positive_preds": positives,
        "model_path": str(out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train sandbox XGBoost models.")
    ap.add_argument("--limit", type=int, default=LIMIT)
    ap.add_argument(
        "--round-trip-cost",
        type=float,
        default=max(DEFAULT_LABEL_ROUND_TRIP, 0.0010),
        help="Label friction threshold (round-trip commissions+spread proxy).",
    )
    args = ap.parse_args()

    logger.info(
        "Training symbols=%s | timeframe=%s | horizon=%d | split=%d/%d | round_trip_cost=%.5f",
        SYMBOLS,
        TIMEFRAME,
        HORIZON,
        int(TRAIN_RATIO * 100),
        int((1 - TRAIN_RATIO) * 100),
        args.round_trip_cost,
    )

    results: list[dict[str, float | int | str]] = []
    for sym in SYMBOLS:
        try:
            r = train_one(sym, limit=args.limit, round_trip_cost=args.round_trip_cost)
            results.append(r)
            logger.info(
                "Done %s | rows=%d train=%d test=%d acc=%.4f model=%s",
                r["symbol"],
                r["rows_total"],
                r["rows_train"],
                r["rows_test"],
                r["accuracy"],
                r["model_path"],
            )
        except Exception as exc:
            logger.error("Failed training for %s: %s", sym, exc)

    logger.info("All models trained: %d", len(results))
    for r in results:
        p = Path(str(r["model_path"]))
        exists = p.is_file()
        sz = p.stat().st_size if exists else 0
        logger.info("VERIFY %s exists=%s size=%d", p.name, exists, sz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
