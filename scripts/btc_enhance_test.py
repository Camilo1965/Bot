#!/usr/bin/env python3
"""
scripts/btc_enhance_test.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Walk-forward compare 4 enhancement configs on BTC:
- BASE: current production
- KELLY: fractional Kelly sizing
- HTF: HTF SMA trend filter
- ALL: KELLY + HTF + cooldown after 2 losses
"""

from __future__ import annotations

import logging
import sys
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
logger = logging.getLogger("btc_enhance")

SYMBOL = "BTC/USDT"


def build_folds(n_folds: int = 5, limit: int = 30_000):
    cfg = SYMBOL_CONFIG[SYMBOL]
    raw = fetch_ohlcv_ccxt(SYMBOL, timeframe=cfg["timeframe"], limit=limit)
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
        logger.info("Pre-train fold %d", i + 1)
        rm = train_regime_model(train_df, cfg["horizon"], adx_threshold=cfg.get("regime_adx", 25.0))
        dm = train_direction_model(train_df, cfg["horizon"], max_spw=cfg.get("max_spw", 12.0))
        folds.append({"test_df": test_df, "rm": rm, "dm": dm, "days": test_days})
    return folds


def run_config(folds, name: str, *, kelly=False, htf=False, cooldown=0) -> dict:
    bt.USE_KELLY = kelly
    bt.USE_HTF_FILTER = htf
    bt.COOLDOWN_AFTER_LOSSES = cooldown
    per_fold = []
    for f in folds:
        state = simulate_bot(f["test_df"], f["rm"], f["dm"], SYMBOL, use_regime=True)
        m = calc_metrics(state, f["days"])
        start_p = float(f["test_df"]["close"].iloc[0])
        end_p = float(f["test_df"]["close"].iloc[-1])
        bnh = (end_p - start_p) / start_p * 100
        m["alpha_pct"] = m.get("pnl_pct", 0.0) - bnh
        m["bnh_pct"] = bnh
        per_fold.append(m)
    pnls = np.array([m.get("pnl_pct", 0.0) for m in per_fold])
    pfs = np.array([m.get("profit_factor", 0.0) for m in per_fold if m.get("profit_factor", 0.0) != float("inf")])
    dds = np.array([m.get("max_drawdown_pct", 0.0) for m in per_fold])
    alphas = np.array([m.get("alpha_pct", 0.0) for m in per_fold])
    profitable = int((pnls > 0).sum())
    return {
        "name": name,
        "avg_pnl": float(pnls.mean()),
        "std_pnl": float(pnls.std()),
        "median_pf": float(np.median(pfs)) if len(pfs) > 0 else 0.0,
        "avg_dd": float(dds.mean()),
        "max_dd": float(dds.max()),
        "avg_alpha": float(alphas.mean()),
        "profitable_folds": profitable,
        "n_folds": len(per_fold),
        "per_fold_pnl": pnls.tolist(),
        "robust": bool(profitable >= len(per_fold) * 0.6 and pnls.mean() > 0 and (np.median(pfs) if len(pfs) > 0 else 0) > 1.20),
    }


def main() -> int:
    logger.info("Building 5 folds for %s...", SYMBOL)
    folds = build_folds()

    configs = [
        ("BASE",  {"kelly": False, "htf": False, "cooldown": 0}),
        ("KELLY", {"kelly": True,  "htf": False, "cooldown": 0}),
        ("HTF",   {"kelly": False, "htf": True,  "cooldown": 0}),
        ("ALL",   {"kelly": True,  "htf": True,  "cooldown": 24}),
        ("HTF+CD",{"kelly": False, "htf": True,  "cooldown": 24}),
    ]

    results = []
    for name, kw in configs:
        logger.info("Running %s ...", name)
        r = run_config(folds, name, **kw)
        results.append(r)

    logger.info("=" * 95)
    logger.info("BTC walk-forward enhancement comparison (5 folds × 62.5d)")
    logger.info("=" * 95)
    logger.info("%-8s | %-8s | %-7s | %-6s | %-7s | %-7s | %-6s | %-7s | ROBUST",
                "Cfg", "avgPnL%", "std", "PF", "avgDD%", "maxDD%", "Prof", "alpha%")
    for r in results:
        logger.info(
            "%-8s | %+8.2f | %6.2f  | %5.2f  | %7.2f | %7.2f | %d/%d   | %+7.2f | %s",
            r["name"], r["avg_pnl"], r["std_pnl"], r["median_pf"],
            r["avg_dd"], r["max_dd"], r["profitable_folds"], r["n_folds"],
            r["avg_alpha"], "YES" if r["robust"] else "no",
        )
    logger.info("Per-fold PnL%%:")
    for r in results:
        logger.info("  %-8s: %s", r["name"], "  ".join(f"{x:+.2f}" for x in r["per_fold_pnl"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
