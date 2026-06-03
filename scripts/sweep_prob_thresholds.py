#!/usr/bin/env python3
"""
scripts/sweep_prob_thresholds.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Grid-sweep prob_threshold per symbol using disk-loaded models + current calibrations.
Trains regime+direction models once per symbol, then varies threshold 0.30→0.90.
Outputs best threshold per symbol (by Sharpe, min 5 trades).

Usage:
    python scripts/sweep_prob_thresholds.py
    python scripts/sweep_prob_thresholds.py --symbol BTC/USDT
    python scripts/sweep_prob_thresholds.py --min-trades 10
"""

from __future__ import annotations

import argparse
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
    _run_simulation_loop,
    add_quant_features,
    calc_metrics,
    train_direction_model,
    train_regime_model,
)
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("sweep_thresholds")

THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
LIMIT = 30_000


def sweep_symbol(symbol: str, min_trades: int = 5) -> dict:
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    hor = cfg["horizon"]
    tf_min = int(tf[:-1])

    logger.info("[%s] Fetching %d %s candles ...", symbol, LIMIT, tf)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=LIMIT)
    if len(raw) < 1000:
        logger.error("[%s] Insufficient data (%d rows)", symbol, len(raw))
        return {}

    split = int(len(raw) * 0.7)
    train_df = raw.iloc[:split].copy()
    test_df = raw.iloc[split:].copy()
    test_days = (
        pd.to_datetime(test_df["timestamp"].iloc[-1])
        - pd.to_datetime(test_df["timestamp"].iloc[0])
    ).total_seconds() / 86400

    logger.info("[%s] Train=%d rows, Test=%d rows (%.1fd)", symbol, len(train_df), len(test_df), test_days)

    regime_model = train_regime_model(train_df, hor, adx_threshold=cfg.get("regime_adx", 25.0))
    direction_model = train_direction_model(train_df, hor, max_spw=cfg.get("max_spw", 12.0))

    feat = add_quant_features(test_df.copy(), base_timeframe_min=tf_min)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)
    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    regime_probs = regime_model.predict_proba(X)[:, 1]
    direction_probs = direction_model.predict_proba(X)[:, 1]

    results = []
    original_threshold = cfg["prob_threshold"]

    for pt in THRESHOLDS:
        cfg["prob_threshold"] = pt
        state = _run_simulation_loop(feat, regime_probs, direction_probs, symbol, use_regime=False)
        m = calc_metrics(state, test_days)
        if m["trades"] >= min_trades:
            results.append({
                "prob_threshold": pt,
                "trades": m["trades"],
                "win_rate": round(m["win_rate"] * 100, 1),
                "pnl_pct": m["pnl_pct"],
                "max_dd_pct": m["max_drawdown_pct"],
                "profit_factor": m["profit_factor"],
                "sharpe": m["sharpe"],
            })

    cfg["prob_threshold"] = original_threshold  # restore

    if not results:
        logger.warning("[%s] No threshold yielded >= %d trades", symbol, min_trades)
        return {}

    best = max(results, key=lambda r: r["sharpe"])
    best_pnl = max(results, key=lambda r: r["pnl_pct"])
    return {"symbol": symbol, "best_sharpe": best, "best_pnl": best_pnl, "all": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--min-trades", type=int, default=5)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CONFIG.keys())
    all_results = {}

    for sym in symbols:
        try:
            r = sweep_symbol(sym, min_trades=args.min_trades)
            if r:
                all_results[sym] = r
        except Exception as exc:
            logger.error("[%s] sweep failed: %s", sym, exc, exc_info=True)

    print("\n" + "=" * 90)
    print(f"{'Symbol':<12} {'BestPT(S)':<10} {'Sharpe':>7} {'PnL%':>7} {'WR%':>6} {'Trades':>7} {'DD%':>7} | {'BestPT(P)':<10} {'PnL%':>7}")
    print("=" * 90)
    for sym, r in all_results.items():
        bs = r["best_sharpe"]
        bp = r["best_pnl"]
        current_pt = SYMBOL_CONFIG[sym]["prob_threshold"]
        marker = " *" if bs["prob_threshold"] != current_pt else "  "
        print(
            f"{sym:<12} {bs['prob_threshold']:<10.2f} {bs['sharpe']:>7.2f} {bs['pnl_pct']:>7.2f} "
            f"{bs['win_rate']:>6.1f} {bs['trades']:>7} {bs['max_dd_pct']:>7.2f} | "
            f"{bp['prob_threshold']:<10.2f} {bp['pnl_pct']:>7.2f}{marker}"
        )
    print("=" * 90)
    print("\n* = recommended threshold differs from current config\n")

    out = _REPO / "logs" / "threshold_sweep_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    logger.info("Saved → %s", out)

    print("\nRecommended SYMBOL_CONFIG updates:")
    for sym, r in all_results.items():
        current_pt = SYMBOL_CONFIG[sym]["prob_threshold"]
        best_pt = r["best_sharpe"]["prob_threshold"]
        if best_pt != current_pt:
            bs = r["best_sharpe"]
            print(f"  {sym}: {current_pt:.2f} -> {best_pt:.2f}  (Sharpe {bs['sharpe']:+.2f}, PnL {bs['pnl_pct']:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
