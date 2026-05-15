#!/usr/bin/env python3
"""Quick backtest with adjustable R:R to find if bot can be profitable."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

def quick_backtest(symbol, tf, limit=20000, label_thresh=0.015, sl_pct=0.015, tp_pct=0.030, risk_pct=0.03, horizon=8):
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    feat = add_quant_features(raw.copy(), base_timeframe_min=int(tf[:-1]))
    fwd = feat["close"].shift(-horizon) / feat["close"] - 1.0
    feat["label"] = (fwd > label_thresh).astype(int)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"])
    
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
        scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    
    balance = 10000.0
    trades = 0
    wins = 0
    gross_wins = 0.0
    gross_losses = 0.0
    max_dd = 0.0
    peak = balance
    
    closes = test["close"].to_numpy()
    highs = test["high"].to_numpy()
    lows = test["low"].to_numpy()
    
    for i in range(len(test) - horizon):
        if probs[i] < 0.60:
            continue
        trades += 1
        entry = float(closes[i])
        sl_price = entry * (1.0 - sl_pct)
        tp_price = entry * (1.0 + tp_pct)
        
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]
        
        # Check TP or SL
        tp_idx = np.where(future_highs >= tp_price)[0]
        sl_idx = np.where(future_lows <= sl_price)[0]
        
        if len(tp_idx) > 0 and (len(sl_idx) == 0 or tp_idx[0] < sl_idx[0]):
            exit_p = tp_price
            win = True
        elif len(sl_idx) > 0:
            exit_p = sl_price
            win = False
        else:
            exit_p = float(closes[i+horizon])
            win = exit_p > entry
        
        pnl_pct = (exit_p - entry) / entry
        position_size = balance * risk_pct / sl_pct
        pnl_usd = pnl_pct * position_size
        commission = position_size * 0.0008
        pnl_usd -= commission
        
        balance += pnl_usd
        if pnl_usd > 0:
            wins += 1
            gross_wins += pnl_usd
        else:
            gross_losses += abs(pnl_usd)
        
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd
    
    wr = wins / trades if trades > 0 else 0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    pnl_pct = (balance - 10000) / 100
    
    test_days = len(test) * int(tf[:-1]) / 1440
    return {
        "symbol": symbol, "tf": tf, "label": label_thresh, "sl": sl_pct, "tp": tp_pct,
        "trades": trades, "wr": wr, "pnl": pnl_pct, "dd": max_dd * 100,
        "pf": pf, "auc": roc_auc_score(y_te, probs),
        "trades_per_week": trades / max(test_days / 7, 1),
    }

print("Testing ETH with different R:R configurations:")
print("Label | SL   | TP   | Trades | WR%  | PnL%  | DD%  | PF   | Trd/Wk")
print("-" * 75)

configs = [
    (0.015, 0.015, 0.030),  # R:R 1:2, label=SL
    (0.010, 0.010, 0.020),  # R:R 1:2, label=SL
    (0.020, 0.020, 0.040),  # R:R 1:2, label=SL
    (0.015, 0.010, 0.030),  # R:R 1:3, SL < label
    (0.010, 0.010, 0.030),  # R:R 1:3
]

for lt, sl, tp in configs:
    r = quick_backtest("ETH/USDT", "15m", label_thresh=lt, sl_pct=sl, tp_pct=tp)
    print(f"{lt:.3f} | {sl:.3f} | {tp:.3f} | {r['trades']:6d} | {r['wr']*100:4.1f} | {r['pnl']:+6.2f} | {r['dd']:5.2f} | {r['pf']:4.2f} | {r['trades_per_week']:5.1f}")

print("\nTesting BTC:")
for lt, sl, tp in [(0.015, 0.015, 0.030), (0.010, 0.010, 0.020)]:
    r = quick_backtest("BTC/USDT", "30m", label_thresh=lt, sl_pct=sl, tp_pct=tp, horizon=12)
    print(f"{lt:.3f} | {sl:.3f} | {tp:.3f} | {r['trades']:6d} | {r['wr']*100:4.1f} | {r['pnl']:+6.2f} | {r['dd']:5.2f} | {r['pf']:4.2f} | {r['trades_per_week']:5.1f}")
