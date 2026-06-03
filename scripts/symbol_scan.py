#!/usr/bin/env python3
"""
scripts/symbol_scan.py
~~~~~~~~~~~~~~~~~~~~~~

Screening pass for candidate USDT pairs. For each candidate:
  1. Fetch 180d at 15m CCXT.
  2. 70/30 split. Train XGB direction + regime on the training slice.
  3. Mini grid sweep on the test slice (8 combos: pt × tp).
  4. Record best metrics + score.

Survivor criteria: score > 5  AND  trades >= 20  AND  PF >= 1.2  AND  DD <= 12%.

Survivors should then be:
  - Promoted via `python scripts/retrain_model.py --symbols X/USDT --days 180`
  - Calibration fit via `python scripts/refit_calibrations.py --symbol X/USDT`
  - Disk-mode param-swept via `python scripts/param_sweep.py --mode disk --symbol X/USDT`
  - Added to canonical `strategy/ml_predictor.SYMBOL_CONFIG`

Output: logs/symbol_scan_results.csv (ranked)
"""

from __future__ import annotations

import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

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
from data.ohlcv_cache import get_cache as _get_ohlcv_cache
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from scripts.param_sweep import score as composite_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("symbol_scan")

CANDIDATES: list[str] = [
    # Original candidates
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "BNB/USDT",
    "DOT/USDT",
    "NEAR/USDT",
    "ATOM/USDT",
    "POL/USDT",  # MATIC renamed to POL/Polygon 2023
    # Phase D narrative-2026 expansion (I-20)
    "ARB/USDT",
    "OP/USDT",
    "SUI/USDT",
    "SEI/USDT",
    "INJ/USDT",
    "TIA/USDT",
    "JTO/USDT",
    "RUNE/USDT",
    "KAS/USDT",
]

# Seed config used for inline training & sweeping
SEED = {
    "prob_threshold": 0.55,
    "fixed_sl_pct": 0.025,
    "fixed_tp_pct": 0.040,
    "use_sma_filter": True,
    "risk": 0.010,
    "timeframe": "15m",
    "horizon": 20,
    "regime_adx": 25.0,
    "max_spw": 8.0,
    "skip_regime": True,
}

# Mini grid (8 combos)
MINI_GRID = {
    "prob_threshold": [0.45, 0.55],
    "fixed_tp_pct": [0.035, 0.050],
    "fixed_sl_pct": [0.025, 0.030],
}

SURVIVOR_MIN_SCORE = 5.0
SURVIVOR_MIN_TRADES = 20
SURVIVOR_MIN_PF = 1.2
SURVIVOR_MAX_DD = 12.0


def scan_one(symbol: str, limit: int = 17280) -> dict[str, Any]:
    """One candidate. limit=17280 ≈ 180d at 15m."""
    logger.info("[%s] fetching %d candles...", symbol, limit)
    try:
        raw = _get_ohlcv_cache().fetch(symbol, SEED["timeframe"], limit)
    except Exception as exc:
        return {"symbol": symbol, "error": f"fetch_failed:{exc}"}
    if raw is None or len(raw) < 8000:
        return {"symbol": symbol, "error": f"insufficient_data:{0 if raw is None else len(raw)}"}

    split = int(len(raw) * 0.7)
    train_df = raw.iloc[:split].copy()
    test_df = raw.iloc[split:].copy()
    test_days = (
        pd.to_datetime(test_df["timestamp"].iloc[-1]) - pd.to_datetime(test_df["timestamp"].iloc[0])
    ).total_seconds() / 86400.0

    logger.info("[%s] training models on %d rows; test=%d (%.1fd)", symbol, len(train_df), len(test_df), test_days)
    try:
        regime_model = train_regime_model(train_df, SEED["horizon"], adx_threshold=SEED["regime_adx"])
        direction_model = train_direction_model(train_df, SEED["horizon"], max_spw=SEED["max_spw"])
    except Exception as exc:
        return {"symbol": symbol, "error": f"train_failed:{exc}"}

    # Inject transient config so simulate_bot's SYMBOL_CONFIG lookup works
    import scripts.backtest_full_bot as backtest_mod
    backtest_mod.SYMBOL_CONFIG[symbol] = {**SEED}

    best: dict[str, Any] | None = None
    try:
        for pt, tp, sl in product(MINI_GRID["prob_threshold"], MINI_GRID["fixed_tp_pct"], MINI_GRID["fixed_sl_pct"]):
            backtest_mod.SYMBOL_CONFIG[symbol] = {**SEED, "prob_threshold": pt, "fixed_tp_pct": tp, "fixed_sl_pct": sl}
            state = simulate_bot(test_df, regime_model, direction_model, symbol, use_regime=False)
            m = calc_metrics(state, max(test_days, 1.0))
            sc = composite_score(m)
            row = {
                "prob_threshold": pt, "fixed_tp_pct": tp, "fixed_sl_pct": sl,
                "trades": m.get("trades", 0),
                "win_rate": m.get("win_rate", 0.0),
                "pnl_pct": m.get("pnl_pct", 0.0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
                "sharpe": m.get("sharpe", 0.0),
                "score": sc,
            }
            if best is None or sc > best["score"]:
                best = row
    finally:
        backtest_mod.SYMBOL_CONFIG.pop(symbol, None)

    if best is None:
        return {"symbol": symbol, "error": "no_results"}

    pf_val = best.get("profit_factor", 0.0)
    if pf_val == float("inf"):
        pf_check = True
    else:
        pf_check = pf_val >= SURVIVOR_MIN_PF

    survivor = (
        best["score"] > SURVIVOR_MIN_SCORE
        and best["trades"] >= SURVIVOR_MIN_TRADES
        and pf_check
        and best["max_drawdown_pct"] <= SURVIVOR_MAX_DD
    )
    return {"symbol": symbol, "test_days": round(test_days, 1), "survivor": bool(survivor), "best": best}


def main() -> int:
    rows: list[dict[str, Any]] = []
    for sym in CANDIDATES:
        try:
            res = scan_one(sym)
        except Exception as exc:
            logger.error("[%s] crashed: %s", sym, exc, exc_info=True)
            res = {"symbol": sym, "error": str(exc)}
        rows.append(res)

    # Build CSV
    flat: list[dict[str, Any]] = []
    for r in rows:
        if "best" in r:
            b = r["best"]
            flat.append({
                "symbol": r["symbol"],
                "test_days": r.get("test_days", 0),
                "survivor": r.get("survivor", False),
                **b,
            })
        else:
            flat.append({"symbol": r["symbol"], "error": r.get("error", "?")})

    df = pd.DataFrame(flat)
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    out_csv = _REPO / "logs" / "symbol_scan_results.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)

    logger.info("=" * 110)
    logger.info("SYMBOL SCAN RESULTS  (inline 70/30, mini grid 8 combos)")
    logger.info("=" * 110)
    logger.info("%-12s | %-8s | %-7s | %-6s | %-8s | %-7s | %-7s | %-7s", "Symbol", "Survivor", "Trades", "Win%", "PnL%", "DD%", "PF", "Score")
    logger.info("-" * 110)
    for r in rows:
        if "error" in r:
            logger.info("%-12s | ERROR: %s", r["symbol"], r["error"])
            continue
        b = r["best"]
        pf_str = "inf" if b["profit_factor"] == float("inf") else f"{b['profit_factor']:>7.2f}"
        logger.info(
            "%-12s | %-8s | %-7d | %-6.1f | %-+8.2f | %-7.2f | %-7s | %-7.2f",
            r["symbol"], "YES" if r["survivor"] else "no",
            int(b["trades"]), b["win_rate"] * 100, b["pnl_pct"],
            b["max_drawdown_pct"], pf_str, b["score"],
        )
    survivors = [r["symbol"] for r in rows if r.get("survivor")]
    logger.info("=" * 110)
    logger.info("SURVIVORS (%d): %s", len(survivors), ", ".join(survivors) if survivors else "(none)")

    out_json = _REPO / "logs" / "symbol_scan_results.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    logger.info("Saved %s and %s", out_csv, out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
