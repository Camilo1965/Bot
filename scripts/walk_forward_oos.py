#!/usr/bin/env python3
"""
scripts/walk_forward_oos.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

5-fold rolling walk-forward OOS validation for all active symbols.

NO actual retrain of disk models — trains ephemeral in-memory XGBClassifiers
per fold and runs a minimal simulation loop to compute per-fold metrics.

Usage::

    python scripts/walk_forward_oos.py [--symbols BTC/USDT ETH/USDT ...] \
                                       [--days 240] [--folds 5] [--test-days 30]

Output: logs/walk_forward_oos_report.json + console table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.ohlcv_cache import get_cache
from scripts.backtest_full_bot import (
    SYMBOL_CONFIG,
    _run_simulation_loop,
    calc_metrics,
    BacktestState,
)
from strategy.quant_features import (
    QUANT_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
    DEFAULT_LABEL_ROUND_TRIP,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("walk_forward_oos")

ACTIVE_SYMBOLS = [s for s in SYMBOL_CONFIG if s != "XRP/USDT"]

# Candles per day per timeframe
_CPD: dict[str, int] = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48, "1h": 24, "2h": 12, "4h": 6, "1d": 1,
}


def _candles_per_day(tf: str) -> int:
    return _CPD.get(tf, 96)


def _fetch_data(symbol: str, tf: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data via cache, fallback to CCXT raw fetch."""
    limit = days * _candles_per_day(tf)
    try:
        df = get_cache().fetch(symbol, tf, limit=limit, max_age_s=3600)
        if df is not None and len(df) >= limit // 2:
            logger.info("[%s] Cache returned %d rows", symbol, len(df))
            return df
    except Exception as exc:
        logger.warning("[%s] Cache fetch failed: %s — falling back to CCXT", symbol, exc)

    # Direct CCXT fallback
    try:
        from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
        df = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
        logger.info("[%s] CCXT returned %d rows", symbol, len(df) if df is not None else 0)
        return df if df is not None else pd.DataFrame()
    except Exception as exc2:
        logger.error("[%s] CCXT fallback also failed: %s", symbol, exc2)
        return pd.DataFrame()


def _train_direction_model(
    train_df: pd.DataFrame,
    horizon: int,
    max_spw: float | None,
    tf_min: int,
) -> XGBClassifier | None:
    """Train ephemeral XGBClassifier on train_df. Returns None if insufficient data."""
    feat = add_quant_features(train_df.copy(), base_timeframe_min=tf_min)
    y_series = forward_return_label(
        feat["close"], horizon=horizon, round_trip_cost=DEFAULT_LABEL_ROUND_TRIP
    )
    feat = feat.copy()
    feat["_label"] = y_series.values
    feat = feat.dropna(subset=list(QUANT_FEATURE_COLS) + ["_label"])

    if len(feat) < 150:
        logger.warning("  Insufficient train rows after NaN drop (%d)", len(feat))
        return None

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = feat["_label"].to_numpy(dtype=np.int32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    raw_spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
    spw = min(raw_spw, max_spw) if max_spw is not None else raw_spw

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _fit_isotonic_calibrator(
    model: XGBClassifier,
    val_df: pd.DataFrame,
    horizon: int,
    tf_min: int,
) -> IsotonicRegression | None:
    """Fit isotonic regression calibrator on validation portion of train."""
    feat = add_quant_features(val_df.copy(), base_timeframe_min=tf_min)
    y_series = forward_return_label(
        feat["close"], horizon=horizon, round_trip_cost=DEFAULT_LABEL_ROUND_TRIP
    )
    feat = feat.copy()
    feat["_label"] = y_series.values
    feat = feat.dropna(subset=list(QUANT_FEATURE_COLS) + ["_label"])

    if len(feat) < 30:
        return None

    X_val = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = feat["_label"].to_numpy(dtype=np.int32)
    raw_probs = model.predict_proba(X_val)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_val)
    return iso


def _simulate_fold(
    test_df: pd.DataFrame,
    model: XGBClassifier,
    calibrator: IsotonicRegression | None,
    symbol: str,
    tf_min: int,
) -> BacktestState:
    """Run the standard _run_simulation_loop on the test fold."""
    cfg = SYMBOL_CONFIG[symbol]
    feat = add_quant_features(test_df.copy(), base_timeframe_min=tf_min)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)

    if len(feat) < 10:
        return BacktestState()

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    raw_probs = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        direction_probs = calibrator.predict(raw_probs)
    else:
        direction_probs = raw_probs

    # Regime: use neutral (0.5 → trending allowed since skip_regime=True for all active)
    regime_probs = np.full(len(feat), 0.5, dtype=np.float32)

    use_regime = not cfg.get("skip_regime", True)
    return _run_simulation_loop(feat, regime_probs, direction_probs, symbol, use_regime=use_regime)


def _bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI on the mean of values."""
    if len(values) == 0:
        return (0.0, 0.0)
    arr = np.array(values)
    boot_means = np.array([
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (round(lo, 4), round(hi, 4))


def run_symbol(
    symbol: str,
    days: int = 240,
    n_folds: int = 5,
    test_days: int = 30,
) -> dict[str, Any]:
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    horizon = int(cfg.get("horizon", 20))
    max_spw = cfg.get("max_spw", None)
    tf_min = int(tf[:-1])
    cpd = _candles_per_day(tf)
    test_size = test_days * cpd  # candles per test fold

    logger.info("=== %s | tf=%s | %dd | %d folds | test=%dd ===", symbol, tf, days, n_folds, test_days)
    df = _fetch_data(symbol, tf, days)

    if df is None or len(df) < test_size * 2 + 200:
        logger.warning("[%s] Not enough data (%d rows), skipping.", symbol, len(df) if df is not None else 0)
        return {"symbol": symbol, "error": "insufficient_data", "rows": len(df) if df is not None else 0}

    total_rows = len(df)
    # Compute fold split points so that each test window doesn't overlap
    # with previous test windows. Train always starts at row 0 (expanding window).
    # test fold k starts at: total_rows - (n_folds - k) * test_size
    fold_metrics: list[dict[str, float]] = []

    # López de Prado embargo: purge label-overlap zone at fold boundaries.
    # Labels look forward `horizon` bars, so rows within horizon*2 of a
    # boundary carry information from both sides → must be excluded.
    embargo_bars = horizon * 2

    for k in range(n_folds):
        test_start = total_rows - (n_folds - k) * test_size
        test_end = test_start + test_size

        if test_start < 300:
            logger.warning("  Fold %d: train too small (%d rows), skipping fold.", k, test_start)
            continue

        train_df = df.iloc[:test_start].copy()
        # Embargo: drop last embargo_bars of train so labels don't bleed into test
        train_df_purged = train_df.iloc[:-embargo_bars] if len(train_df) > embargo_bars else train_df
        # Test: skip first embargo_bars to avoid train-label overlap
        test_df = df.iloc[test_start + embargo_bars:test_end].copy()

        if len(test_df) < 50:
            logger.warning("  Fold %d: test too small after embargo (%d rows), skipping.", k, len(test_df))
            continue

        # Val portion: last 20% of purged train for calibrator
        val_split = int(len(train_df_purged) * 0.80)
        pure_train_df = train_df_purged.iloc[:val_split].copy()
        # Embargo between pure_train and val
        val_df = train_df_purged.iloc[val_split + embargo_bars:].copy()
        if len(val_df) < 30:
            val_df = train_df_purged.iloc[val_split:].copy()  # fallback: no embargo on val

        logger.info(
            "  Fold %d/%d: train=%d rows, val=%d rows, test=%d rows (embargo=%d)",
            k + 1, n_folds, len(pure_train_df), len(val_df), len(test_df), embargo_bars,
        )

        model = _train_direction_model(pure_train_df, horizon, max_spw, tf_min)
        if model is None:
            logger.warning("  Fold %d: model training failed, skipping.", k)
            continue

        calibrator = _fit_isotonic_calibrator(model, val_df, horizon, tf_min)

        state = _simulate_fold(test_df, model, calibrator, symbol, tf_min)

        test_days_actual = (
            pd.to_datetime(test_df["timestamp"].iloc[-1]) - pd.to_datetime(test_df["timestamp"].iloc[0])
        ).total_seconds() / 86400 if len(test_df) > 1 else float(test_days)

        m = calc_metrics(state, test_days_actual)
        m["fold"] = k + 1
        m["train_rows"] = len(pure_train_df)
        m["test_rows"] = len(test_df)
        fold_metrics.append(m)

        logger.info(
            "  Fold %d → trades=%d WR=%.1f%% PnL=%.2f%% PF=%.2f DD=%.2f%% Sharpe=%.2f",
            k + 1,
            m.get("trades", 0),
            m.get("win_rate", 0) * 100,
            m.get("pnl_pct", 0),
            m.get("profit_factor", 0),
            m.get("max_drawdown_pct", 0),
            m.get("sharpe", 0),
        )

    if not fold_metrics:
        return {"symbol": symbol, "error": "no_valid_folds"}

    # Aggregate: mean ± std across folds
    agg_keys = ["trades", "win_rate", "pnl_pct", "profit_factor", "max_drawdown_pct", "sharpe"]
    agg: dict[str, Any] = {}
    for key in agg_keys:
        vals = [m[key] for m in fold_metrics if key in m]
        if vals:
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)

    # Bootstrap CI on pnl_pct
    pnl_vals = [m["pnl_pct"] for m in fold_metrics if "pnl_pct" in m]
    ci_lo, ci_hi = _bootstrap_ci(pnl_vals)
    agg["pnl_pct_ci95"] = [ci_lo, ci_hi]

    return {
        "symbol": symbol,
        "n_folds_run": len(fold_metrics),
        "folds": fold_metrics,
        "aggregate": agg,
    }


def print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'Symbol':<14} {'Folds':>5} {'Trades':>8} {'WinRate':>9} {'PnL%':>8} {'PF':>7} {'MaxDD%':>8} {'Sharpe':>8} {'PnL CI95%'}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        sym = r["symbol"]
        if "error" in r:
            print(f"{sym:<14}  ERROR: {r['error']}")
            continue
        agg = r.get("aggregate", {})
        n = r.get("n_folds_run", 0)
        trades_m = agg.get("trades_mean", 0)
        wr_m = agg.get("win_rate_mean", 0) * 100
        pnl_m = agg.get("pnl_pct_mean", 0)
        pf_m = agg.get("profit_factor_mean", 0)
        dd_m = agg.get("max_drawdown_pct_mean", 0)
        sh_m = agg.get("sharpe_mean", 0)
        ci = agg.get("pnl_pct_ci95", [0, 0])
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        print(
            f"{sym:<14} {n:>5} {trades_m:>8.1f} {wr_m:>8.1f}% "
            f"{pnl_m:>7.2f}% {pf_m:>7.2f} {dd_m:>7.2f}% {sh_m:>8.2f}  {ci_str}"
        )
    print("=" * len(header))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5-fold rolling walk-forward OOS validation")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to run (default: all active minus XRP)")
    parser.add_argument("--days", type=int, default=240, help="Total historical days to fetch (default 240)")
    parser.add_argument("--folds", type=int, default=5, help="Number of OOS folds (default 5)")
    parser.add_argument("--test-days", type=int, default=30, help="Days per test fold (default 30)")
    args = parser.parse_args(argv)

    symbols = args.symbols if args.symbols else ACTIVE_SYMBOLS
    # Validate
    unknown = [s for s in symbols if s not in SYMBOL_CONFIG]
    if unknown:
        logger.warning("Unknown symbols (not in SYMBOL_CONFIG): %s", unknown)
        symbols = [s for s in symbols if s in SYMBOL_CONFIG]

    all_results: list[dict[str, Any]] = []
    for sym in symbols:
        try:
            result = run_symbol(sym, days=args.days, n_folds=args.folds, test_days=args.test_days)
        except Exception as exc:
            logger.exception("[%s] Unexpected error: %s", sym, exc)
            result = {"symbol": sym, "error": str(exc)}
        all_results.append(result)

    print_table(all_results)

    out_path = _REPO / "logs" / "walk_forward_oos_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_params": {"days": args.days, "folds": args.folds, "test_days": args.test_days},
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Report saved → %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
