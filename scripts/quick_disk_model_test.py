#!/usr/bin/env python3
"""Quick validation: use DISK model + production SL/TP/HTF logic on last 60d."""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from xgboost import XGBClassifier
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features
from strategy.prob_calibration import calibrate_probability


def simulate(symbol: str, tf: str, tf_min: int, pt: float, sl_frac: float, tp_frac: float, ttl_hours: float, htf_period: int = 100):
    m = XGBClassifier()
    m.load_model(f"models/{symbol.replace('/', '_')}_v2.json")
    df = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=4000)
    feat = add_quant_features(df, base_timeframe_min=tf_min).dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)
    if len(feat) < 200:
        return None

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    raw_probas = m.predict_proba(X)[:, 1]
    probas = np.array([calibrate_probability(float(p), symbol) for p in raw_probas])

    closes = feat["close"].to_numpy()
    highs = feat["high"].to_numpy()
    lows = feat["low"].to_numpy()
    htf_sma = feat["close"].rolling(htf_period, min_periods=htf_period).mean().to_numpy()
    bars_ttl = int(ttl_hours * 60 / tf_min)

    # Last 30d slice: ~30 * (24*60/tf_min) bars
    test_start = max(htf_period, len(feat) - 30 * 24 * 60 // tf_min)

    open_pos = None
    trades = []
    for i in range(test_start, len(feat)):
        price = float(closes[i])

        # Manage open
        if open_pos:
            held = i - open_pos["entry_i"]
            hi_p = float(highs[i]); lo_p = float(lows[i])
            if lo_p <= open_pos["sl"]:
                trades.append({"exit": open_pos["sl"], "reason": "SL", "held": held, "entry": open_pos["entry"]})
                open_pos = None
            elif hi_p >= open_pos["tp"]:
                trades.append({"exit": open_pos["tp"], "reason": "TP", "held": held, "entry": open_pos["entry"]})
                open_pos = None
            elif held >= bars_ttl:
                trades.append({"exit": price, "reason": "TTL", "held": held, "entry": open_pos["entry"]})
                open_pos = None
            else:
                continue

        # Entry checks
        if open_pos is not None:
            continue
        if np.isnan(htf_sma[i]) or price < htf_sma[i]:
            continue
        if probas[i] < pt:
            continue
        open_pos = {
            "entry_i": i, "entry": price,
            "sl": price * (1.0 - sl_frac),
            "tp": price * (1.0 + tp_frac),
        }

    if not trades:
        return {"trades": 0}
    pnls = [(t["exit"] - t["entry"]) / t["entry"] for t in trades]
    wr = sum(1 for p in pnls if p > 0) / len(pnls)
    total = sum(pnls)
    pos = sum(p for p in pnls if p > 0)
    neg = -sum(p for p in pnls if p < 0)
    pf = pos / neg if neg > 0 else float("inf")
    return {
        "trades": len(trades), "win_rate": round(wr, 3),
        "total_pct": round(total * 100, 2), "pf": round(pf, 2),
        "best": round(max(pnls) * 100, 2), "worst": round(min(pnls) * 100, 2),
    }


def main():
    print("DISK model validation (last ~30d, calibrated probs, HTF filter, SL/TP/TTL)\n")
    configs = [
        ("BTC/USDT", "30m", 30, 0.60, 0.025, 0.030, 18.0),
        ("ETH/USDT", "15m", 15, 0.30, 0.025, 0.060, 5.0),
    ]
    for cfg in configs:
        sym = cfg[0]
        r = simulate(*cfg)
        print(f"{sym}: {r}")


if __name__ == "__main__":
    main()
