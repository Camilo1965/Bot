#!/usr/bin/env python3
"""Train per-symbol REGIME (trending vs ranging) XGBoost models.

Regime is highly predictable (AUC ~0.94) compared to direction (AUC ~0.52).
This script trains regime classifiers for all symbols and saves them as
models/{symbol}_regime.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_regime_models")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
LIMIT = 50_000


def build_regime_dataset(df: pd.DataFrame, tf_min: int, adx_threshold: float = 25.0, look_ahead: int = 12) -> pd.DataFrame:
    """Label: 1 = will be trending (ADX > threshold) in *look_ahead* bars, 0 = ranging."""
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    # Label: future ADX > threshold (predicting regime ahead of time)
    feat["future_adx"] = feat["adx"].shift(-look_ahead)
    feat["label"] = (feat["future_adx"] > adx_threshold).astype(int)
    # Keep only clear cases: current ADX either clearly trending or clearly ranging
    feat = feat[(feat["adx"] > adx_threshold + 5) | (feat["adx"] < adx_threshold - 5)].copy()
    return feat.dropna(subset=list(QUANT_FEATURE_COLS) + ["label"]).reset_index(drop=True)


def train_regime(symbol: str, limit: int) -> dict[str, float | str]:
    from strategy.ml_predictor import get_symbol_config
    cfg = get_symbol_config(symbol)
    tf = cfg.get("timeframe", "15m")
    tf_min = int(tf[:-1]) if tf[:-1].isdigit() else 15
    hor = cfg.get("horizon", 8)

    logger.info("Fetching %d %s candles for %s (regime prediction)...", limit, tf, symbol)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    df = build_regime_dataset(raw, tf_min, adx_threshold=25.0, look_ahead=hor)

    if len(df) < 1000:
        raise ValueError(f"{symbol}: not enough data ({len(df)} rows)")

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    x_train = train_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    x_test = test_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
    }

    out = Path("models") / f"{symbol.replace('/', '_')}_regime.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))

    meta = {
        "symbol": symbol,
        "timeframe": tf,
        "metrics": metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "pos_rate": round(float(y_train.mean()), 4),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(
        "%s regime model | AUC=%.4f ACC=%.4f F1=%.4f | pos=%d neg=%d | %s",
        symbol, metrics["auc"], metrics["accuracy"], metrics["f1"],
        n_pos, n_neg, out,
    )
    return {**metrics, "model_path": str(out), "meta_path": str(meta_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--limit", type=int, default=LIMIT)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    results = {}
    for sym in symbols:
        try:
            results[sym] = train_regime(sym, args.limit)
        except Exception as exc:
            logger.error("Failed %s: %s", sym, exc)

    logger.info("=" * 60)
    logger.info("REGIME MODELS SUMMARY")
    logger.info("%-15s | %-6s | %-6s | %-6s | %-6s", "Symbol", "AUC", "ACC", "F1", "PREC")
    for sym, m in results.items():
        logger.info(
            "%-15s | %.4f | %.4f | %.4f | %.4f",
            sym, m.get("auc", 0), m.get("accuracy", 0), m.get("f1", 0), m.get("precision", 0),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
