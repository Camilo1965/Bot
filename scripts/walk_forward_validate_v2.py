#!/usr/bin/env python3
"""
scripts/walk_forward_validate_v2.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Walk-forward chronological validation of the tuned SYMBOL_CONFIG.

Splits the full CCXT history into K expanding folds. For each fold:
- Train on rows [0 : train_end]
- Test on rows  [train_end : train_end + test_size]
- Run `simulate_bot` with current SYMBOL_CONFIG values

Goal: detect overfitting to one specific OOS slice. If NEW config wins in
at least 60% of folds AND avg PnL > 0 AND PF > 1.20, parameters are robust.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.backtest_full_bot import (
    SYMBOL_CONFIG,
    calc_metrics,
    simulate_bot,
    train_direction_model,
    train_regime_model,
)
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("walk_forward")


def walk_forward(symbol: str, n_folds: int = 5, limit: int = 30_000) -> dict:
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    hor = cfg["horizon"]

    logger.info("Fetching %d %s candles for %s...", limit, tf, symbol)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    if len(raw) < 5000:
        return {"error": "insufficient data"}

    total = len(raw)
    # Initial train = 50% of data. Remaining 50% split into N_folds test slices.
    train_start_size = int(total * 0.50)
    remaining = total - train_start_size
    test_size = remaining // n_folds

    results = []
    for fold_i in range(n_folds):
        train_end = train_start_size + fold_i * test_size
        test_start = train_end
        test_end = min(test_start + test_size, total)

        train_df = raw.iloc[:train_end].copy()
        test_df = raw.iloc[test_start:test_end].copy()
        if len(test_df) < 100:
            continue

        test_days = (
            pd.to_datetime(test_df["timestamp"].iloc[-1])
            - pd.to_datetime(test_df["timestamp"].iloc[0])
        ).total_seconds() / 86400

        logger.info(
            "Fold %d/%d %s: train=%d → test=%d (%.1fd) [%s → %s]",
            fold_i + 1, n_folds, symbol, len(train_df), len(test_df), test_days,
            test_df["timestamp"].iloc[0], test_df["timestamp"].iloc[-1],
        )

        regime_model = train_regime_model(train_df, hor, adx_threshold=cfg.get("regime_adx", 25.0))
        direction_model = train_direction_model(train_df, hor, max_spw=cfg.get("max_spw", 12.0))

        use_regime = not cfg.get("skip_regime", False)
        state = simulate_bot(test_df, regime_model, direction_model, symbol, use_regime=use_regime)
        m = calc_metrics(state, test_days)

        # Buy & hold benchmark for fold
        start_p = float(test_df["close"].iloc[0])
        end_p = float(test_df["close"].iloc[-1])
        bnh_pct = (end_p - start_p) / start_p * 100

        results.append({
            "fold": fold_i + 1,
            "test_start": str(test_df["timestamp"].iloc[0]),
            "test_end": str(test_df["timestamp"].iloc[-1]),
            "test_days": round(test_days, 1),
            "trades": m["trades"],
            "win_rate_pct": round(m.get("win_rate", 0.0) * 100, 1),
            "pnl_pct": m.get("pnl_pct", 0.0),
            "max_dd_pct": m.get("max_drawdown_pct", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "bnh_pct": round(bnh_pct, 2),
            "alpha_pct": round(m.get("pnl_pct", 0.0) - bnh_pct, 2),
        })

    if not results:
        return {"error": "no folds completed"}

    # Aggregate
    pnl_arr = np.array([r["pnl_pct"] for r in results])
    pf_arr = np.array([r["profit_factor"] for r in results if r["profit_factor"] != float("inf")])
    alpha_arr = np.array([r["alpha_pct"] for r in results])

    summary = {
        "symbol": symbol,
        "folds": results,
        "n_folds": len(results),
        "n_profitable_folds": int((pnl_arr > 0).sum()),
        "n_beat_bnh_folds": int((alpha_arr > 0).sum()),
        "avg_pnl_pct": round(float(pnl_arr.mean()), 2),
        "std_pnl_pct": round(float(pnl_arr.std()), 2),
        "median_pf": round(float(np.median(pf_arr)) if len(pf_arr) > 0 else 0.0, 2),
        "avg_alpha_vs_bnh_pct": round(float(alpha_arr.mean()), 2),
        "robust": bool(
            (pnl_arr > 0).sum() >= len(results) * 0.6
            and pnl_arr.mean() > 0
            and (np.median(pf_arr) if len(pf_arr) > 0 else 0) > 1.20
        ),
    }
    return summary


def main() -> int:
    summary = {}
    syms = sys.argv[1:] if len(sys.argv) > 1 else ["BTC/USDT", "ETH/USDT"]
    for sym in syms:
        logger.info("=" * 80)
        logger.info("WALK-FORWARD: %s", sym)
        logger.info("=" * 80)
        res = walk_forward(sym, n_folds=5, limit=30_000)
        summary[sym] = res

        if "error" in res:
            logger.error("%s: %s", sym, res["error"])
            continue

        logger.info("Fold-by-fold:")
        logger.info("%-6s | %-10s | %-7s | %-7s | %-7s | %-8s | %-8s | %-7s",
                    "Fold", "Days", "Trades", "WR%", "PnL%", "DD%", "PF", "B&H%")
        for r in res["folds"]:
            pf_str = f"{r['profit_factor']:.2f}" if r["profit_factor"] != float("inf") else "INF"
            logger.info(
                "%-6d | %-10.1f | %-7d | %-7.1f | %+7.2f | %-8.2f | %-8s | %+6.2f",
                r["fold"], r["test_days"], r["trades"], r["win_rate_pct"],
                r["pnl_pct"], r["max_dd_pct"], pf_str, r["bnh_pct"],
            )
        logger.info(
            "AGG: profitable=%d/%d  beat_B&H=%d/%d  avg_PnL=%+.2f%% (±%.2f)  median_PF=%.2f  alpha=%+.2f%%  ROBUST=%s",
            res["n_profitable_folds"], res["n_folds"],
            res["n_beat_bnh_folds"], res["n_folds"],
            res["avg_pnl_pct"], res["std_pnl_pct"],
            res["median_pf"], res["avg_alpha_vs_bnh_pct"],
            "YES" if res["robust"] else "NO",
        )

    out = _REPO / "logs" / "walk_forward_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Summary saved to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
