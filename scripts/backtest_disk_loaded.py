#!/usr/bin/env python3
"""
scripts/backtest_disk_loaded.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Production-replica backtest. Loads `{SYMBOL}_v2.json` direction booster +
`{SYMBOL}_regime.json` (if regime is used) + `{SYMBOL}_calibration.json` from
disk and runs them through the *exact* simulation loop the inline backtest uses
(`backtest_full_bot._run_simulation_loop`). Calibrated probabilities pass
through `strategy.prob_calibration.calibrate_probability` — same path as the
live signal emitter.

Usage:
    python scripts/backtest_disk_loaded.py --symbol BTC/USDT --days 60
    python scripts/backtest_disk_loaded.py --all --days 60
    python scripts/backtest_disk_loaded.py --report --days 30        # Phase F
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.backtest_full_bot import (
    INITIAL_BALANCE,
    REGIME_BUY_THRESHOLD,
    SYMBOL_CONFIG,
    _run_simulation_loop,
    calc_metrics,
)
from data.ohlcv_cache import get_cache as _get_ohlcv_cache
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt  # kept for direct callers outside cache
from strategy.prob_calibration import calibrate_probability, load_calibration
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backtest_disk_loaded")

MODELS_DIR: Path = _REPO / "models"

# How many candles to fetch per day per timeframe. ~ceil for safety.
_CANDLES_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6}


def _load_booster(path: Path) -> XGBClassifier | None:
    if not path.is_file():
        logger.warning("Model not found: %s", path)
        return None
    booster = XGBClassifier()
    booster.load_model(str(path))
    return booster


def _slice_recent_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    cutoff = ts.max() - pd.Timedelta(days=days)
    return df[ts >= cutoff].reset_index(drop=True)


def _fetch_for_symbol(symbol: str, days: int, buffer_days: int = 30) -> pd.DataFrame:
    """Fetch enough candles to cover `days` of test window + `buffer_days` warmup
    for indicators (ATR/SMA200 need ~200 bars of context)."""
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    per_day = _CANDLES_PER_DAY.get(tf, 96)
    limit = (days + buffer_days) * per_day
    logger.info("Fetching %d %s candles for %s (~%dd)...", limit, tf, symbol, days + buffer_days)
    return _get_ohlcv_cache().fetch(symbol, tf, limit)


def run_symbol(symbol: str, days: int) -> dict[str, Any]:
    """Disk-loaded backtest for one symbol. Returns metrics dict."""
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    tf_min = int(tf[:-1])
    skip_regime = bool(cfg.get("skip_regime", False))

    raw = _fetch_for_symbol(symbol, days)
    if raw is None or len(raw) < 500:
        return {"symbol": symbol, "error": "insufficient_data", "rows": int(len(raw) if raw is not None else 0)}

    sym_key = symbol.replace("/", "_")
    dir_path = MODELS_DIR / f"{sym_key}_v2.json"
    regime_path = MODELS_DIR / f"{sym_key}_regime.json"

    direction_model = _load_booster(dir_path)
    if direction_model is None:
        return {"symbol": symbol, "error": f"missing_model:{dir_path.name}"}

    regime_model = None if skip_regime else _load_booster(regime_path)
    if not skip_regime and regime_model is None:
        logger.warning("Regime model missing for %s — forcing use_regime=False", symbol)

    # Compute features on FULL window so indicators warm up, then slice to last `days`.
    feat_full = add_quant_features(raw.copy(), base_timeframe_min=tf_min)
    feat_full = feat_full.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)
    feat = _slice_recent_days(feat_full, days)
    if len(feat) < 50:
        return {"symbol": symbol, "error": "insufficient_features_post_warmup", "rows": int(len(feat))}

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)

    # Direction probs → calibrated like signal_emitter
    raw_dir = direction_model.predict_proba(X)[:, 1]
    cal = load_calibration(symbol)
    if cal is None:
        logger.info("[%s] no calibration file — using raw probs (identity)", symbol)
        direction_probs = raw_dir.astype(np.float64)
    else:
        xs, ys = cal
        direction_probs = np.clip(np.interp(raw_dir, xs, ys), 0.0, 1.0)
        logger.info("[%s] calibrated probs: raw mean=%.3f cal mean=%.3f", symbol, float(raw_dir.mean()), float(direction_probs.mean()))

    # Regime probs
    if regime_model is not None and not skip_regime:
        regime_probs = regime_model.predict_proba(X)[:, 1]
    else:
        regime_probs = np.ones(len(X), dtype=np.float64)  # always passes filter

    use_regime = (regime_model is not None) and not skip_regime

    state = _run_simulation_loop(feat, regime_probs, direction_probs, symbol, use_regime=use_regime)

    test_days_actual = float((pd.to_datetime(feat["timestamp"].iloc[-1], utc=True) - pd.to_datetime(feat["timestamp"].iloc[0], utc=True)).total_seconds() / 86400.0)
    metrics = calc_metrics(state, max(test_days_actual, 1.0))

    return {
        "symbol": symbol,
        "test_days": round(test_days_actual, 2),
        "raw_prob_mean": round(float(raw_dir.mean()), 4),
        "cal_prob_mean": round(float(direction_probs.mean()), 4),
        "skip_regime": skip_regime,
        "metrics": metrics,
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    logger.info("=" * 96)
    logger.info("DISK-LOADED BACKTEST RESULTS (production stack — calibrated)")
    logger.info("=" * 96)
    logger.info("%-12s | %-7s | %-6s | %-8s | %-7s | %-7s | %-7s | %-6s | %-7s",
                "Symbol", "Trades", "Win%", "PnL%", "DD%", "PF", "Sharpe", "Trd/Wk", "RawCal")
    logger.info("-" * 96)
    for r in results:
        if "error" in r:
            logger.info("%-12s | ERROR: %s", r.get("symbol", "?"), r["error"])
            continue
        m = r["metrics"]
        if m.get("trades", 0) == 0:
            logger.info("%-12s | NO TRADES (thresholds too strict or calib compresses probs)", r["symbol"])
            continue
        logger.info(
            "%-12s | %-7d | %-6.1f | %-+8.2f | %-7.2f | %-7.2f | %-7.2f | %-6.1f | %.2f/%.2f",
            r["symbol"], m["trades"], m["win_rate"] * 100, m["pnl_pct"],
            m.get("max_drawdown_pct", 0.0), m.get("profit_factor", 0.0),
            m.get("sharpe", 0.0), m.get("trades_per_week", 0.0),
            r.get("raw_prob_mean", 0.0), r.get("cal_prob_mean", 0.0),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol, e.g. BTC/USDT")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--all", action="store_true", help="Run for every symbol in SYMBOL_CONFIG")
    parser.add_argument("--days", type=int, default=60, help="Recent days window for the test")
    parser.add_argument("--report", action="store_true", help="Save Phase F final_verification_report.json")
    parser.add_argument("--prob-threshold", type=float, default=None,
                        help="Override prob_threshold for all symbols in this run")
    args = parser.parse_args()

    if args.symbol is None and args.symbols is None and not args.all and not args.report:
        parser.error("Pass --symbol SYMBOL/USDT or --symbols SYM1,SYM2 or --all or --report")

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = list(SYMBOL_CONFIG.keys())

    if args.prob_threshold is not None:
        for sym in symbols:
            if sym in SYMBOL_CONFIG:
                SYMBOL_CONFIG[sym] = {**SYMBOL_CONFIG[sym], "prob_threshold": args.prob_threshold}
        logger.info("prob_threshold override: %.3f for %s", args.prob_threshold, symbols)

    results: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            res = run_symbol(sym, args.days)
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", sym, exc, exc_info=True)
            res = {"symbol": sym, "error": str(exc)}
        results.append(res)

    _print_table(results)

    # Save report
    out_name = "final_verification_report.json" if args.report else f"backtest_disk_loaded_{args.days}d.json"
    out = _REPO / "logs" / out_name
    out.parent.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "initial_balance": INITIAL_BALANCE,
        "regime_buy_threshold": REGIME_BUY_THRESHOLD,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Saved %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
