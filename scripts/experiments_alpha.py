#!/usr/bin/env python3
"""
scripts/experiments_alpha.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pipeline comparativo de 6 estrategias para aumentar la probabilidad de ganar.

Estrategias:
1. BASELINE: 26 features técnicas (v2 actual)
2. MICRO: + proxies de microestructura (gap, volume zscore, vol_of_vol)
3. CROSS: + features cross-asset (btc_eth_corr, relative strength)
4. REGIME: target cambiado a TRENDING vs RANGING (más predecible)
5. 1M: datos 1min con horizonte 60 velas (1h)
6. MLP: MLPClassifier (sklearn) con features MICRO

Para cada estrategia entrena con 50k velas, reporta AUC/ACC/F1.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import (
    DEFAULT_LABEL_ROUND_TRIP,
    QUANT_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("experiments_alpha")


def _calc_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
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


def build_baseline(df: pd.DataFrame, horizon: int, tf_min: int) -> tuple[pd.DataFrame, list[str]]:
    """Dataset baseline con 26 features."""
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    feat["label"] = forward_return_label(feat["close"], horizon, DEFAULT_LABEL_ROUND_TRIP)
    cols = list(QUANT_FEATURE_COLS)
    return feat.dropna(subset=cols + ["label"]).reset_index(drop=True), cols


def build_micro(df: pd.DataFrame, horizon: int, tf_min: int) -> tuple[pd.DataFrame, list[str]]:
    """Dataset baseline + proxies de microestructura."""
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    feat["label"] = forward_return_label(feat["close"], horizon, DEFAULT_LABEL_ROUND_TRIP)

    # Microstructure proxies
    feat["gap_1m"] = (feat["high"] - feat["low"]) / feat["close"]  # candle range
    feat["volume_zscore"] = (feat["volume"] - feat["volume"].rolling(20).mean()) / feat["volume"].rolling(20).std().replace(0, np.nan)
    feat["volume_zscore"] = feat["volume_zscore"].fillna(0.0)
    feat["vol_of_vol"] = feat["log_ret_1"].rolling(20).std().fillna(0.0)
    feat["large_wick"] = ((feat["high"] - feat["low"]) / feat["close"] > feat["gap_1m"].rolling(50).quantile(0.9)).astype(float)
    feat["volume_spike"] = (feat["volume"] > feat["volume"].rolling(20).mean() * 2.0).astype(float)

    extra = ["gap_1m", "volume_zscore", "vol_of_vol", "large_wick", "volume_spike"]
    cols = list(QUANT_FEATURE_COLS) + extra
    return feat.dropna(subset=cols + ["label"]).reset_index(drop=True), cols


def build_cross(df_btc: pd.DataFrame, df_eth: pd.DataFrame, horizon: int, tf_min: int) -> tuple[pd.DataFrame, list[str]]:
    """Dataset cross-asset: BTC como principal, ETH como referencia."""
    feat = add_quant_features(df_btc, base_timeframe_min=tf_min)
    feat["label"] = forward_return_label(feat["close"], horizon, DEFAULT_LABEL_ROUND_TRIP)

    # ETH features aligned by index (assuming same timestamps)
    feat_eth = add_quant_features(df_eth, base_timeframe_min=tf_min)
    min_len = min(len(feat), len(feat_eth))
    feat = feat.iloc[:min_len].copy()
    feat_eth = feat_eth.iloc[:min_len]

    feat["eth_rsi"] = feat_eth["rsi"].values
    feat["eth_ret_1"] = feat_eth["log_ret_1"].values
    feat["btc_eth_corr_20"] = feat["log_ret_1"].rolling(20).corr(pd.Series(feat_eth["log_ret_1"].values)).fillna(0.0)
    feat["btc_vs_eth_str"] = (feat["log_ret_1"] - feat_eth["log_ret_1"]).fillna(0.0)
    # Fear/greed proxy from BTC RSI
    feat["fear_greed_proxy"] = (feat["rsi"] - 50) / 50.0  # [-1, 1]

    extra = ["eth_rsi", "eth_ret_1", "btc_eth_corr_20", "btc_vs_eth_str", "fear_greed_proxy"]
    cols = list(QUANT_FEATURE_COLS) + extra
    return feat.dropna(subset=cols + ["label"]).reset_index(drop=True), cols


def build_regime(df: pd.DataFrame, tf_min: int) -> tuple[pd.DataFrame, list[str]]:
    """Dataset con target = TRENDING (1) vs RANGING (0)."""
    feat = add_quant_features(df, base_timeframe_min=tf_min)
    # Trending if ADX > 25, ranging if ADX < 20, else drop
    feat["label"] = ((feat["adx"] > 25.0) & (feat["adx"].shift(-12) > 25.0)).astype(int)
    feat = feat[(feat["adx"] > 25.0) | (feat["adx"] < 20.0)].copy()
    cols = list(QUANT_FEATURE_COLS)
    return feat.dropna(subset=cols + ["label"]).reset_index(drop=True), cols


def build_1m(symbol: str = "BTC/USDT", limit: int = 30000) -> tuple[pd.DataFrame, list[str], int]:
    """Fetch 1m data and build dataset."""
    logger.info("Fetching %d 1m candles for %s...", limit, symbol)
    raw = fetch_ohlcv_ccxt(symbol, timeframe="1m", limit=limit)
    feat = add_quant_features(raw, base_timeframe_min=1)
    # 1h horizon = 60 velas
    feat["label"] = forward_return_label(feat["close"], 60, DEFAULT_LABEL_ROUND_TRIP)
    cols = list(QUANT_FEATURE_COLS)
    return feat.dropna(subset=cols + ["label"]).reset_index(drop=True), cols, 60


def train_xgb(X: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    spw = _calc_scale_pos_weight(y)
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, early_stopping_rounds=20,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X, y, eval_set=[(x_test, y_test)], verbose=False)
    preds = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return _evaluate(y_test, preds, proba)


def train_mlp(X: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    # Normalize inputs for MLP
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    x_test_s = scaler.transform(x_test)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_s, y)
    preds = model.predict(x_test_s)
    proba = model.predict_proba(x_test_s)[:, 1]
    return _evaluate(y_test, preds, proba)


def run_experiment(name: str, df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    if len(df) < 1000:
        return {"error": "insufficient data"}
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    x_train = train_df[cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    x_test = test_df[cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    if name == "MLP":
        metrics = train_mlp(x_train, y_train, x_test, y_test)
    else:
        metrics = train_xgb(x_train, y_train, x_test, y_test)

    metrics["n_features"] = len(cols)
    metrics["n_train"] = len(train_df)
    metrics["n_test"] = len(test_df)
    metrics["pos_rate"] = round(float(y_train.mean()), 4)
    return metrics


def main() -> int:
    results: dict[str, dict[str, dict[str, float]]] = {}

    # ── BASELINE & MICRO & REGIME ──
    for symbol, tf, hor in [("BTC/USDT", "30m", 12), ("ETH/USDT", "15m", 8)]:
        logger.info("=" * 60)
        logger.info("EXPERIMENTS for %s | %s | horizon=%d", symbol, tf, hor)
        tf_min = int(tf[:-1])
        raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=50000)

        # Baseline
        df_base, cols_base = build_baseline(raw, hor, tf_min)
        res_base = run_experiment("BASELINE", df_base, cols_base)
        logger.info("BASELINE: %s", json.dumps(res_base))

        # Micro
        df_micro, cols_micro = build_micro(raw, hor, tf_min)
        res_micro = run_experiment("MICRO", df_micro, cols_micro)
        logger.info("MICRO: %s", json.dumps(res_micro))

        # Regime
        df_reg, cols_reg = build_regime(raw, tf_min)
        res_reg = run_experiment("REGIME", df_reg, cols_reg)
        logger.info("REGIME: %s", json.dumps(res_reg))

        results[f"{symbol}_{tf}"] = {
            "BASELINE": res_base,
            "MICRO": res_micro,
            "REGIME": res_reg,
        }

    # ── CROSS-ASSET (BTC as principal, ETH as reference) ──
    logger.info("=" * 60)
    logger.info("CROSS-ASSET EXPERIMENT")
    raw_btc = fetch_ohlcv_ccxt("BTC/USDT", timeframe="30m", limit=50000)
    raw_eth = fetch_ohlcv_ccxt("ETH/USDT", timeframe="30m", limit=50000)
    df_cross, cols_cross = build_cross(raw_btc, raw_eth, 12, 30)
    res_cross = run_experiment("CROSS", df_cross, cols_cross)
    logger.info("CROSS: %s", json.dumps(res_cross))
    results["BTC_ETH_30m"] = {"CROSS": res_cross}

    # ── 1M EXPERIMENT ──
    logger.info("=" * 60)
    logger.info("1M EXPERIMENT")
    df_1m, cols_1m, hor_1m = build_1m("BTC/USDT", limit=30000)
    res_1m = run_experiment("1M", df_1m, cols_1m)
    logger.info("1M: %s", json.dumps(res_1m))
    results["BTC_1m"] = {"1M": res_1m}

    # ── MLP EXPERIMENT (same data as MICRO) ──
    logger.info("=" * 60)
    logger.info("MLP EXPERIMENT")
    raw_btc_mlp = fetch_ohlcv_ccxt("BTC/USDT", timeframe="30m", limit=50000)
    df_mlp, cols_mlp = build_micro(raw_btc_mlp, 12, 30)
    res_mlp = run_experiment("MLP", df_mlp, cols_mlp)
    logger.info("MLP: %s", json.dumps(res_mlp))
    results["BTC_30m_MLP"] = {"MLP": res_mlp}

    # ── FINAL SUMMARY ──
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("%-20s | %-10s | %-6s | %-6s | %-6s | %-6s | %-4s", "Experiment", "Symbol", "AUC", "ACC", "F1", "PREC", "REC")
    logger.info("-" * 80)
    for key, exps in results.items():
        for exp_name, metrics in exps.items():
            if "error" in metrics:
                continue
            logger.info(
                "%-20s | %-10s | %.4f | %.4f | %.4f | %.4f | %.4f",
                exp_name, key,
                metrics.get("auc", 0.0),
                metrics.get("accuracy", 0.0),
                metrics.get("f1", 0.0),
                metrics.get("precision", 0.0),
                metrics.get("recall", 0.0),
            )

    # Save results
    out_path = Path("logs/experiments_alpha_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Results saved to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
