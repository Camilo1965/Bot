#!/usr/bin/env python3
"""
scripts/param_sweep.py
~~~~~~~~~~~~~~~~~~~~~~

Per-symbol parameter sweep over (prob_threshold, fixed_tp_pct, fixed_sl_pct,
ttl_hours, risk).

Two modes:

* ``--mode inline`` (default) — fetches 180d of OHLCV, splits 70/30, trains
  XGB direction + regime models once on the training slice, then re-simulates
  the test slice across the Cartesian grid. Reflects "theoretical ceiling"
  with fresh inline-trained models. Calibration NOT applied.

* ``--mode disk`` — loads ``models/{SYMBOL}_v2.json`` and
  ``models/{SYMBOL}_calibration.json`` from disk and re-simulates a 60d
  out-of-sample window. Calibrated probabilities are computed once; the
  sweep varies only the execution parameters. Matches live behavior — this
  is the mode to trust for production tuning.

Ranks each row by a composite ``score`` that penalises PF<1.2 / DD>8% and
rewards PnL plus win-rate above 50%.

Output:
* logs/param_sweep_{symbol}_{mode}.csv
* logs/param_sweep_summary_{mode}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.backtest_disk_loaded import _CANDLES_PER_DAY, _slice_recent_days
from scripts.backtest_full_bot import (
    SYMBOL_CONFIG,
    _run_simulation_loop,
    calc_metrics,
    simulate_bot,
    train_direction_model,
    train_regime_model,
)
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.prob_calibration import load_calibration
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("param_sweep")

# Per-symbol grids. Wider than v1 to cover all 5 production symbols with the
# calibrated probability distribution (cal_mean ~0.25–0.45 depending on symbol).
GRIDS: dict[str, dict[str, list[float]]] = {
    "BTC/USDT": {
        "prob_threshold": [0.28, 0.35, 0.45, 0.55, 0.60, 0.65, 0.70],
        "fixed_tp_pct":   [0.025, 0.030, 0.040, 0.050],
        "fixed_sl_pct":   [0.020, 0.025, 0.030],
        "ttl_hours":      [12.0, 18.0],
        "risk":           [0.010, 0.015],
    },
    "ETH/USDT": {
        "prob_threshold": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        "fixed_tp_pct":   [0.030, 0.040, 0.050, 0.060],
        "fixed_sl_pct":   [0.020, 0.025, 0.030],
        "ttl_hours":      [5.0, 8.0, 12.0],
        "risk":           [0.008, 0.010, 0.015],
    },
    "XRP/USDT": {
        "prob_threshold": [0.50, 0.60, 0.65, 0.70, 0.75, 0.80],
        "fixed_tp_pct":   [0.030, 0.040, 0.050, 0.060],
        "fixed_sl_pct":   [0.020, 0.025, 0.030],
        "ttl_hours":      [4.0, 8.0, 12.0],
        "risk":           [0.005, 0.008, 0.010],
    },
    "SOL/USDT": {
        "prob_threshold": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
        "fixed_tp_pct":   [0.035, 0.045, 0.055],
        "fixed_sl_pct":   [0.025, 0.030, 0.035],
        "ttl_hours":      [8.0, 12.0, 18.0],
        "risk":           [0.008, 0.010, 0.015],
    },
    "DOGE/USDT": {
        "prob_threshold": [0.30, 0.40, 0.45, 0.50, 0.55, 0.60],
        "fixed_tp_pct":   [0.035, 0.045, 0.055],
        "fixed_sl_pct":   [0.025, 0.030, 0.035],
        "ttl_hours":      [4.0, 8.0, 12.0],
        "risk":           [0.006, 0.008, 0.010],
    },
    "LINK/USDT": {
        "prob_threshold": [0.30, 0.40, 0.50, 0.55, 0.60, 0.65],
        "fixed_tp_pct":   [0.030, 0.040, 0.050],
        "fixed_sl_pct":   [0.020, 0.025, 0.030],
        "ttl_hours":      [5.0, 8.0, 12.0],
        "risk":           [0.008, 0.010, 0.015],
    },
    "NEAR/USDT": {
        "prob_threshold": [0.30, 0.40, 0.50, 0.55, 0.60, 0.65],
        "fixed_tp_pct":   [0.030, 0.040, 0.050],
        "fixed_sl_pct":   [0.020, 0.025, 0.030],
        "ttl_hours":      [5.0, 8.0, 12.0],
        "risk":           [0.008, 0.010, 0.015],
    },
    "ATOM/USDT": {
        "prob_threshold": [0.30, 0.40, 0.50, 0.55, 0.60, 0.65],
        "fixed_tp_pct":   [0.030, 0.040, 0.050],
        "fixed_sl_pct":   [0.020, 0.025, 0.030],
        "ttl_hours":      [5.0, 8.0, 12.0],
        "risk":           [0.008, 0.010, 0.015],
    },
}

MIN_TRADES_FLOOR = 5  # absolute floor; high-quality low-frequency configs allowed if PF + PnL strong
HIGH_FREQ_FLOOR = 20  # below this, require PF >= 2.0 AND PnL > 0 to score positive


def score(metrics: dict[str, float]) -> float:
    """Composite score. Heavy penalty for PF<1.2 or DD>8%. Reward PnL + WR.
    Allows low-frequency configs when PF + PnL are strong (PF >= 2.0 AND PnL > 0)."""
    trades = int(metrics.get("trades", 0))
    if trades < MIN_TRADES_FLOOR:
        return -999.0
    pnl = float(metrics.get("pnl_pct", 0.0))
    pf_raw = metrics.get("profit_factor", 0.0)
    pf = float("inf") if pf_raw == float("inf") else float(pf_raw)
    dd = float(metrics.get("max_drawdown_pct", 100.0))
    wr = float(metrics.get("win_rate", 0.0))

    # Low-frequency quality gate: require strong PF and positive PnL
    if trades < HIGH_FREQ_FLOOR and (pf < 2.0 or pnl <= 0):
        return -999.0

    pf_effective = min(pf, 20.0)  # cap pf bonus contribution from inf
    pf_bonus = max(0.0, pf_effective - 1.0) * 10.0
    dd_pen = max(0.0, dd - 8.0) * 2.0
    wr_bonus = max(0.0, wr - 0.50) * 50.0
    return pnl + pf_bonus + wr_bonus - dd_pen


def _fetch_disk_test_slice(symbol: str, days: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """Returns (feat_test, regime_probs, calibrated_direction_probs) for disk mode."""
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    tf_min = int(tf[:-1])
    skip_regime = bool(cfg.get("skip_regime", False))
    per_day = _CANDLES_PER_DAY.get(tf, 96)
    limit = (days + 30) * per_day  # warmup buffer

    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    if raw is None or len(raw) < 500:
        logger.error("[%s] disk-mode fetch insufficient (%s rows)", symbol, 0 if raw is None else len(raw))
        return None

    sym_key = symbol.replace("/", "_")
    dir_path = _REPO / "models" / f"{sym_key}_v2.json"
    regime_path = _REPO / "models" / f"{sym_key}_regime.json"
    if not dir_path.is_file():
        logger.error("[%s] missing direction model %s", symbol, dir_path.name)
        return None

    from xgboost import XGBClassifier
    direction_model = XGBClassifier()
    direction_model.load_model(str(dir_path))
    regime_model = None
    if not skip_regime and regime_path.is_file():
        regime_model = XGBClassifier()
        regime_model.load_model(str(regime_path))

    feat_full = add_quant_features(raw.copy(), base_timeframe_min=tf_min)
    feat_full = feat_full.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)
    feat = _slice_recent_days(feat_full, days)
    if len(feat) < 50:
        logger.error("[%s] post-warmup feat too small (%d)", symbol, len(feat))
        return None

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    raw_dir = direction_model.predict_proba(X)[:, 1]
    cal = load_calibration(symbol)
    if cal is None:
        direction_probs = raw_dir.astype(np.float64)
    else:
        xs, ys = cal
        direction_probs = np.clip(np.interp(raw_dir, xs, ys), 0.0, 1.0)

    if regime_model is not None:
        regime_probs = regime_model.predict_proba(X)[:, 1]
    else:
        regime_probs = np.ones(len(X), dtype=np.float64)

    logger.info("[%s] disk mode ready: rows=%d raw_mean=%.3f cal_mean=%.3f",
                symbol, len(feat), float(raw_dir.mean()), float(direction_probs.mean()))
    return feat, regime_probs, direction_probs


def sweep_inline(symbol: str, limit: int = 30_000) -> pd.DataFrame:
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    hor = cfg["horizon"]

    logger.info("[%s] inline mode — fetching %d %s candles...", symbol, limit, tf)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    if len(raw) < 1000:
        logger.error("[%s] insufficient data", symbol)
        return pd.DataFrame()

    split = int(len(raw) * 0.7)
    train_df = raw.iloc[:split].copy()
    test_df = raw.iloc[split:].copy()
    test_days = (
        pd.to_datetime(test_df["timestamp"].iloc[-1])
        - pd.to_datetime(test_df["timestamp"].iloc[0])
    ).total_seconds() / 86400

    logger.info("[%s] training on %d rows, test=%d rows (%.1fd)...", symbol, len(train_df), len(test_df), test_days)
    regime_model = train_regime_model(train_df, hor, adx_threshold=cfg.get("regime_adx", 25.0))
    direction_model = train_direction_model(train_df, hor, max_spw=cfg.get("max_spw", 12.0))

    return _run_grid(symbol, GRIDS[symbol], test_days, mode_fn=lambda: simulate_bot(
        test_df, regime_model, direction_model, symbol,
        use_regime=not cfg.get("skip_regime", False),
    ))


def sweep_disk(symbol: str, days: int = 60) -> pd.DataFrame:
    cfg = SYMBOL_CONFIG[symbol]
    pack = _fetch_disk_test_slice(symbol, days)
    if pack is None:
        return pd.DataFrame()
    feat, regime_probs, direction_probs = pack
    test_days = float(
        (pd.to_datetime(feat["timestamp"].iloc[-1], utc=True) - pd.to_datetime(feat["timestamp"].iloc[0], utc=True))
        .total_seconds() / 86400.0
    )
    use_regime = not cfg.get("skip_regime", False)

    return _run_grid(symbol, GRIDS[symbol], test_days, mode_fn=lambda: _run_simulation_loop(
        feat, regime_probs, direction_probs, symbol, use_regime=use_regime,
    ))


def _run_grid(symbol: str, grid: dict[str, list[float]], test_days: float, mode_fn) -> pd.DataFrame:
    import scripts.backtest_full_bot as backtest_mod
    orig_cfg = dict(backtest_mod.SYMBOL_CONFIG[symbol])
    orig_ttl = backtest_mod.TTL_HOURS

    rows: list[dict[str, Any]] = []
    combos = list(product(
        grid["prob_threshold"], grid["fixed_tp_pct"], grid["fixed_sl_pct"],
        grid["ttl_hours"], grid["risk"],
    ))
    logger.info("[%s] evaluating %d combinations...", symbol, len(combos))

    try:
        for i, (pt, tp, sl, ttl, risk) in enumerate(combos):
            backtest_mod.SYMBOL_CONFIG[symbol] = {
                **orig_cfg,
                "prob_threshold": pt,
                "fixed_tp_pct": tp,
                "fixed_sl_pct": sl,
                "risk": risk,
            }
            backtest_mod.TTL_HOURS = ttl
            state = mode_fn()
            m = calc_metrics(state, max(test_days, 1.0))
            rows.append({
                "prob_threshold": pt,
                "fixed_tp_pct": tp,
                "fixed_sl_pct": sl,
                "ttl_hours": ttl,
                "risk": risk,
                "trades": m.get("trades", 0),
                "win_rate": m.get("win_rate", 0.0),
                "pnl_pct": m.get("pnl_pct", 0.0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
                "sharpe": m.get("sharpe", 0.0),
                "score": score(m),
            })
            if (i + 1) % 50 == 0:
                logger.info("  [%s] %d/%d combos done", symbol, i + 1, len(combos))
    finally:
        backtest_mod.SYMBOL_CONFIG[symbol] = orig_cfg
        backtest_mod.TTL_HOURS = orig_ttl

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inline", "disk"], default="disk")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol; otherwise sweep all")
    parser.add_argument("--days", type=int, default=60, help="Disk mode test window")
    parser.add_argument("--limit", type=int, default=30_000, help="Inline mode candle fetch limit")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(GRIDS.keys())

    summary: dict[str, Any] = {}
    for sym in symbols:
        if sym not in GRIDS:
            logger.warning("No grid defined for %s — skipping", sym)
            continue

        if args.mode == "inline":
            df = sweep_inline(sym, limit=args.limit)
        else:
            df = sweep_disk(sym, days=args.days)
        if df.empty:
            continue

        out = _REPO / "logs" / f"param_sweep_{sym.replace('/', '_')}_{args.mode}.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("[%s] saved %s (%d rows)", sym, out, len(df))

        baseline_cfg = SYMBOL_CONFIG[sym]
        baseline_row = df[
            (df["prob_threshold"] == baseline_cfg.get("prob_threshold"))
            & (df["fixed_tp_pct"] == baseline_cfg.get("fixed_tp_pct"))
            & (df["fixed_sl_pct"] == baseline_cfg.get("fixed_sl_pct"))
        ]
        baseline = baseline_row.iloc[0].to_dict() if len(baseline_row) > 0 else None
        best = df.iloc[0].to_dict()
        summary[sym] = {"baseline": baseline, "best": best}

        logger.info("=" * 100)
        logger.info("RESULTS: %s (mode=%s)", sym, args.mode)
        logger.info("=" * 100)
        if baseline:
            logger.info(
                "BASELINE: pt=%.2f tp=%.3f sl=%.3f ttl=%.0fh risk=%.3f → trades=%d WR=%.1f%% PnL=%+.2f%% DD=%.2f%% PF=%.2f score=%.2f",
                baseline["prob_threshold"], baseline["fixed_tp_pct"], baseline["fixed_sl_pct"], baseline["ttl_hours"], baseline["risk"],
                int(baseline["trades"]), baseline["win_rate"] * 100, baseline["pnl_pct"],
                baseline["max_drawdown_pct"], baseline["profit_factor"], baseline["score"],
            )
        logger.info(
            "BEST:     pt=%.2f tp=%.3f sl=%.3f ttl=%.0fh risk=%.3f → trades=%d WR=%.1f%% PnL=%+.2f%% DD=%.2f%% PF=%.2f score=%.2f",
            best["prob_threshold"], best["fixed_tp_pct"], best["fixed_sl_pct"], best["ttl_hours"], best["risk"],
            int(best["trades"]), best["win_rate"] * 100, best["pnl_pct"],
            best["max_drawdown_pct"], best["profit_factor"], best["score"],
        )
        logger.info("TOP-5:")
        for i in range(min(5, len(df))):
            r = df.iloc[i]
            logger.info(
                "  #%d pt=%.2f tp=%.3f sl=%.3f ttl=%.0fh risk=%.3f trades=%d WR=%.1f%% PnL=%+.2f%% DD=%.2f%% PF=%.2f",
                i + 1, r["prob_threshold"], r["fixed_tp_pct"], r["fixed_sl_pct"], r["ttl_hours"], r["risk"],
                int(r["trades"]), r["win_rate"] * 100, r["pnl_pct"],
                r["max_drawdown_pct"], r["profit_factor"],
            )

    out = _REPO / "logs" / f"param_sweep_summary_{args.mode}.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Summary saved to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
