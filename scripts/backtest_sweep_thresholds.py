#!/usr/bin/env python3
"""
scripts/backtest_sweep_thresholds.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweeps label thresholds and SL values to find profitable configuration.

Problem: current label is fwd > 0.13% but SL is 2-2.5%. R:R is terrible.
Solution: increase label threshold to match or exceed SL.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features


def run_single_config(df: pd.DataFrame, label_thresh: float, sl_pct: float, risk_pct: float, horizon: int) -> dict:
    """Train model + simple backtest on a single config."""
    feat = add_quant_features(df.copy(), base_timeframe_min=15)
    fwd = feat["close"].shift(-horizon) / feat["close"] - 1.0
    feat["label"] = (fwd > label_thresh).astype(int)
    feat = feat.dropna(subset=list(QUANT_FEATURE_COLS) + ["label"])

    if len(feat) < 1000 or feat["label"].sum() < 50:
        return {"error": "insufficient positives"}

    split = int(len(feat) * 0.7)
    train = feat.iloc[:split]
    test = feat.iloc[split:].reset_index(drop=True)

    X_tr = train[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_tr = train["label"].to_numpy(dtype=np.int32)
    X_te = test[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_te = test["label"].to_numpy(dtype=np.int32)

    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.6).astype(int)  # threshold 0.60

    auc = roc_auc_score(y_te, probs)
    acc = accuracy_score(y_te, preds)
    pos_rate = y_te.mean()

    # Simple backtest: when model says BUY, check if fwd return > 0
    balance = 10000.0
    trades = 0
    wins = 0
    pnl = 0.0
    max_dd = 0.0
    peak = balance

    for i in range(len(test) - horizon):
        if probs[i] < 0.60:
            continue
        trades += 1
        entry = float(test["close"].iloc[i])
        # Find exit: either hit SL, or horizon reached
        sl_price = entry * (1.0 - sl_pct)
        future_lows = test["low"].iloc[i + 1 : i + 1 + horizon].to_numpy()
        future_closes = test["close"].iloc[i + 1 : i + 1 + horizon].to_numpy()

        # Check if SL hit
        sl_idx = np.where(future_lows <= sl_price)[0]
        if len(sl_idx) > 0:
            exit_p = sl_price
            reason = "SL"
        else:
            exit_p = float(future_closes[-1])
            reason = "HORIZON"

        trade_pnl_pct = (exit_p - entry) / entry
        # Apply leverage
        trade_pnl = trade_pnl_pct * balance * risk_pct / sl_pct
        # Commission
        trade_pnl -= balance * risk_pct / sl_pct * 0.0004 * 2

        pnl += trade_pnl
        balance += trade_pnl
        if trade_pnl > 0:
            wins += 1
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd

    wr = wins / trades if trades > 0 else 0
    return {
        "label_thresh": label_thresh,
        "sl_pct": sl_pct,
        "trades": trades,
        "win_rate": round(wr, 3),
        "auc": round(auc, 4),
        "acc": round(acc, 4),
        "pnl_pct": round(pnl / 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "pos_rate": round(float(pos_rate), 4),
    }


def main():
    print("Fetching ETH 15m data...")
    raw = fetch_ohlcv_ccxt("ETH/USDT", timeframe="15m", limit=20000)

    configs = [
        # (label_threshold, SL, risk)
        (0.0013, 0.02, 0.03),   # Current (BAD)
        (0.005, 0.02, 0.03),    # 0.5% move
        (0.010, 0.02, 0.03),    # 1.0% move
        (0.015, 0.02, 0.03),    # 1.5% move
        (0.020, 0.02, 0.03),    # 2.0% move
        (0.010, 0.01, 0.03),    # 1% move, 1% SL (R:R = 1:1)
        (0.015, 0.015, 0.03),   # 1.5% move, 1.5% SL (R:R = 1:1)
        (0.020, 0.020, 0.03),   # 2% move, 2% SL (R:R = 1:1)
    ]

    print("\nLabelThresh | SL   | Trades | Win% | AUC    | ACC    | PnL%   | DD%  | PosRate")
    print("-" * 85)
    for lt, sl, rk in configs:
        res = run_single_config(raw, lt, sl, rk, horizon=8)
        if "error" in res:
            print(f"{lt:8.4f}  | {sl:.2f} | ERROR: {res['error']}")
            continue
        print(
            f"{lt:8.4f}  | {sl:.2f} | {res['trades']:6d} | {res['win_rate']*100:4.1f} | "
            f"{res['auc']:.4f} | {res['acc']:.4f} | {res['pnl_pct']:+7.2f} | {res['max_dd_pct']:5.2f} | {res['pos_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
