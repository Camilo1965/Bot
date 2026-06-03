#!/usr/bin/env python3
"""
scripts/param_sweep_wf.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Walk-forward integrated sweep for ETH (NO-ROBUST in prior validation).

Evaluates each combo across 5 folds and ranks by robustness score:
  score = avg_pnl * 0.5 + median_pf_excess * 5.0 + profitable_fold_pct * 10
         - drawdown_penalty

Caches direction+regime models per fold so cost = N_combos × 5_sims (~1s each).

Output: logs/eth_sweep_wf.csv ranked by composite robustness score.
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
logger = logging.getLogger("eth_sweep_wf")


GRIDS = {
    "ETH/USDT": {
        "regime_threshold": [0.55, 0.60, 0.65, 0.70],
        "max_atr_pct":      [None, 0.025, 0.035, 0.045],
        "prob_threshold":   [0.22, 0.26, 0.30, 0.35],
    },
    "SOL/USDT": {
        "regime_threshold": [0.55, 0.60, 0.65, 0.70],
        "max_atr_pct":      [None, 0.030, 0.045, 0.060],
        "prob_threshold":   [0.50, 0.55, 0.60, 0.65],
    },
}


def build_folds(symbol: str = "ETH/USDT", n_folds: int = 5, limit: int = 30_000):
    cfg = SYMBOL_CONFIG[symbol]
    raw = fetch_ohlcv_ccxt(symbol, timeframe=cfg["timeframe"], limit=limit)
    if len(raw) < 5000:
        raise RuntimeError("insufficient data")

    total = len(raw)
    train_start = int(total * 0.50)
    remaining = total - train_start
    step = remaining // n_folds

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
        logger.info("Pre-training fold %d (train=%d, test=%d, %.1fd)", i + 1, len(train_df), len(test_df), test_days)
        regime_m = train_regime_model(train_df, cfg["horizon"], adx_threshold=cfg.get("regime_adx", 25.0))
        dir_m = train_direction_model(train_df, cfg["horizon"], max_spw=cfg.get("max_spw", 8.0))
        folds.append({"test_df": test_df, "regime_m": regime_m, "dir_m": dir_m, "test_days": test_days})
    return folds


def eval_combo(folds: list[dict], symbol: str, rt: float, atr_pct: float | None, pt: float) -> dict:
    bt.REGIME_BUY_THRESHOLD = rt
    bt.MAX_ATR_PCT = atr_pct
    orig_cfg = dict(bt.SYMBOL_CONFIG[symbol])
    bt.SYMBOL_CONFIG[symbol] = {**orig_cfg, "prob_threshold": pt}

    try:
        per_fold = []
        for f in folds:
            state = simulate_bot(f["test_df"], f["regime_m"], f["dir_m"], symbol, use_regime=True)
            m = calc_metrics(state, f["test_days"])
            per_fold.append(m)
    finally:
        bt.SYMBOL_CONFIG[symbol] = orig_cfg

    pnls = np.array([m.get("pnl_pct", 0.0) for m in per_fold])
    pfs = np.array([m.get("profit_factor", 0.0) for m in per_fold if m.get("profit_factor", 0.0) != float("inf")])
    dds = np.array([m.get("max_drawdown_pct", 0.0) for m in per_fold])
    trades_total = sum(m.get("trades", 0) for m in per_fold)
    profitable = int((pnls > 0).sum())

    avg_pnl = float(pnls.mean())
    median_pf = float(np.median(pfs)) if len(pfs) > 0 else 0.0
    avg_dd = float(dds.mean())
    pct_profitable = profitable / len(per_fold)

    # Composite: reward consistent profitability + good PF; penalize DD
    score = (
        avg_pnl * 0.5
        + max(0.0, median_pf - 1.0) * 8.0
        + pct_profitable * 12.0
        - max(0.0, avg_dd - 5.0) * 0.5
    )
    if trades_total < 50:
        score -= 20.0  # under-trading penalty

    return {
        "regime_threshold": rt,
        "max_atr_pct": atr_pct if atr_pct is not None else 0.0,
        "prob_threshold": pt,
        "trades_total": trades_total,
        "profitable_folds": profitable,
        "n_folds": len(per_fold),
        "avg_pnl_pct": round(avg_pnl, 2),
        "std_pnl_pct": round(float(pnls.std()), 2),
        "median_pf": round(median_pf, 2),
        "avg_dd_pct": round(avg_dd, 2),
        "score": round(score, 2),
    }


def main() -> int:
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["ETH/USDT"]
    for symbol in symbols:
        if symbol not in GRIDS:
            logger.warning("No grid defined for %s, skipping", symbol)
            continue
        grid = GRIDS[symbol]
        logger.info("Building folds for %s...", symbol)
        folds = build_folds(symbol)

        combos = list(product(grid["regime_threshold"], grid["max_atr_pct"], grid["prob_threshold"]))
        logger.info("Evaluating %d combos across %d folds for %s...", len(combos), len(folds), symbol)

        rows = []
        for i, (rt, atr, pt) in enumerate(combos):
            rows.append(eval_combo(folds, symbol, rt, atr, pt))
            if (i + 1) % 20 == 0:
                logger.info("  %d/%d done", i + 1, len(combos))

        df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        slug = symbol.replace("/", "_").lower()
        out = _REPO / "logs" / f"{slug}_sweep_wf.csv"
        df.to_csv(out, index=False)
        logger.info("Saved %s (%d rows)", out, len(df))

        logger.info("=" * 80)
        logger.info("%s — top 10 by walk-forward composite score", symbol)
        logger.info("=" * 80)
        for i in range(min(10, len(df))):
            r = df.iloc[i]
            logger.info(
                "#%d  rt=%.2f atr=%.3f pt=%.2f → trades=%d profitable=%d/%d avg_PnL=%+.2f%% med_PF=%.2f DD=%.2f score=%.2f",
                i + 1, r["regime_threshold"], r["max_atr_pct"], r["prob_threshold"],
                int(r["trades_total"]), int(r["profitable_folds"]), int(r["n_folds"]),
                r["avg_pnl_pct"], r["median_pf"], r["avg_dd_pct"], r["score"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
