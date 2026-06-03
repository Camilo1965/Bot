#!/usr/bin/env python3
"""
scripts/eth_tf_test.py
~~~~~~~~~~~~~~~~~~~~~~

ETH walk-forward across alternative (timeframe, horizon, prob_threshold,
fixed_tp_pct) configs to find ANY robust combo. ETH 15m model has no
cross-temporal edge; try 30m + different horizons.
"""

from __future__ import annotations

import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import scripts.backtest_full_bot as bt
from scripts.backtest_full_bot import (
    SYMBOL_CONFIG,
    calc_metrics,
    simulate_bot,
    train_direction_model,
    train_regime_model,
)
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eth_tf")

SYMBOL = "ETH/USDT"

VARIANTS = [
    # (timeframe, horizon, prob_threshold, sl_pct, tp_pct)
    ("15m", 20, 0.30, 0.025, 0.060),   # current
    ("30m", 12, 0.50, 0.025, 0.040),
    ("30m", 12, 0.55, 0.025, 0.040),
    ("30m", 18, 0.55, 0.025, 0.045),
    ("30m", 24, 0.55, 0.030, 0.050),
    ("30m", 24, 0.60, 0.030, 0.050),
    ("1h",  12, 0.55, 0.030, 0.050),
    ("1h",  24, 0.55, 0.030, 0.060),
]


def build_folds_for_tf(tf: str, horizon: int, n_folds: int = 5, limit: int = 20_000):
    raw = fetch_ohlcv_ccxt(SYMBOL, timeframe=tf, limit=limit)
    total = len(raw)
    train_start = int(total * 0.50)
    step = (total - train_start) // n_folds
    folds = []
    for i in range(n_folds):
        train_end = train_start + i * step
        test_start = train_end
        test_end = min(test_start + step, total)
        train_df = raw.iloc[:train_end].copy()
        test_df = raw.iloc[test_start:test_end].copy()
        if len(test_df) < 100:
            continue
        test_days = (
            pd.to_datetime(test_df["timestamp"].iloc[-1])
            - pd.to_datetime(test_df["timestamp"].iloc[0])
        ).total_seconds() / 86400
        rm = train_regime_model(train_df, horizon, adx_threshold=25.0)
        dm = train_direction_model(train_df, horizon, max_spw=8.0)
        folds.append({"test_df": test_df, "rm": rm, "dm": dm, "days": test_days})
    return folds


def eval_variant(folds, tf: str, horizon: int, pt: float, sl: float, tp: float) -> dict:
    orig = dict(bt.SYMBOL_CONFIG[SYMBOL])
    bt.SYMBOL_CONFIG[SYMBOL] = {
        **orig,
        "timeframe": tf, "horizon": horizon, "prob_threshold": pt,
        "fixed_sl_pct": sl, "fixed_tp_pct": tp,
    }
    try:
        per_fold = []
        for f in folds:
            state = simulate_bot(f["test_df"], f["rm"], f["dm"], SYMBOL, use_regime=True)
            m = calc_metrics(state, f["days"])
            start_p = float(f["test_df"]["close"].iloc[0])
            end_p = float(f["test_df"]["close"].iloc[-1])
            m["alpha"] = m.get("pnl_pct", 0.0) - (end_p - start_p) / start_p * 100
            per_fold.append(m)
    finally:
        bt.SYMBOL_CONFIG[SYMBOL] = orig

    pnls = np.array([m.get("pnl_pct", 0.0) for m in per_fold])
    pfs = np.array([m.get("profit_factor", 0.0) for m in per_fold if m.get("profit_factor", 0.0) != float("inf")])
    dds = np.array([m.get("max_drawdown_pct", 0.0) for m in per_fold])
    alphas = np.array([m.get("alpha", 0.0) for m in per_fold])
    profitable = int((pnls > 0).sum())
    trades = sum(m.get("trades", 0) for m in per_fold)

    return {
        "tf": tf, "horizon": horizon, "pt": pt, "sl": sl, "tp": tp,
        "trades": trades, "profitable": profitable, "n_folds": len(per_fold),
        "avg_pnl": float(pnls.mean()), "std_pnl": float(pnls.std()),
        "median_pf": float(np.median(pfs)) if len(pfs) > 0 else 0.0,
        "avg_dd": float(dds.mean()), "max_dd": float(dds.max()),
        "avg_alpha": float(alphas.mean()),
        "robust": bool(profitable >= len(per_fold) * 0.6 and pnls.mean() > 0 and (np.median(pfs) if len(pfs) > 0 else 0) > 1.20),
    }


def main() -> int:
    results = []
    cached_folds: dict[tuple, list] = {}
    for tf, h, pt, sl, tp in VARIANTS:
        key = (tf, h)
        if key not in cached_folds:
            logger.info("Building folds tf=%s h=%d ...", tf, h)
            cached_folds[key] = build_folds_for_tf(tf, h)
        logger.info("Evaluating tf=%s h=%d pt=%.2f sl=%.3f tp=%.3f", tf, h, pt, sl, tp)
        r = eval_variant(cached_folds[key], tf, h, pt, sl, tp)
        results.append(r)

    logger.info("=" * 110)
    logger.info("ETH variant comparison")
    logger.info("=" * 110)
    logger.info("%-5s | %-3s | %-4s | %-6s | %-6s | %-7s | %-7s | %-6s | %-7s | %-7s | %-7s | ROBUST",
                "TF", "H", "pt", "sl", "tp", "trades", "profit", "PF", "avgPnL", "avgDD", "alpha")
    for r in sorted(results, key=lambda x: -x["avg_pnl"]):
        logger.info(
            "%-5s | %-3d | %.2f | %.3f | %.3f | %-7d | %d/%d   | %5.2f | %+6.2f | %6.2f | %+6.2f | %s",
            r["tf"], r["horizon"], r["pt"], r["sl"], r["tp"],
            r["trades"], r["profitable"], r["n_folds"], r["median_pf"],
            r["avg_pnl"], r["avg_dd"], r["avg_alpha"],
            "YES" if r["robust"] else "no",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
