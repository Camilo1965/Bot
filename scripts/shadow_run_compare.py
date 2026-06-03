#!/usr/bin/env python3
"""
scripts/shadow_run_compare.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare live shadow paper-run signals vs backtest predictions.

Reads:
  - logs/shadow_run_signals.csv  (written by signal_emitter)
  - logs/trade_journal.csv       (written by PaperExecutor / DB)

Schema of shadow_run_signals.csv (expected columns):
  timestamp, symbol, raw_prob, cal_prob, threshold, regime_prob, htf_pass,
  vol_pass, hour_filter, ml_decision, executed, reason_skip

Usage::

    python scripts/shadow_run_compare.py [--days N] [--summary]

Output:
  - logs/shadow_compare_daily.csv  (append)
  - console summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("shadow_run_compare")

_SIGNALS_CSV = _REPO / "logs" / "shadow_run_signals.csv"
_JOURNAL_CSV = _REPO / "logs" / "trade_journal.csv"
_COMPARE_CSV = _REPO / "logs" / "shadow_compare_daily.csv"

_DRIFT_THRESHOLD = 0.20  # flag if abs drift > 20%

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_signals(days: int | None = None) -> pd.DataFrame | None:
    """Load shadow_run_signals.csv. Returns None if file missing."""
    if not _SIGNALS_CSV.exists():
        logger.warning(
            "Shadow signals file not found: %s\n"
            "  The file is written by signal_emitter when SHADOW_RUN_LOG=1 is set.",
            _SIGNALS_CSV,
        )
        return None

    try:
        df = pd.read_csv(_SIGNALS_CSV, parse_dates=["timestamp"])
    except Exception as exc:
        logger.error("Failed to read %s: %s", _SIGNALS_CSV, exc)
        return None

    if df.empty:
        logger.info("Shadow signals file is empty.")
        return df

    # Normalise timestamp to UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]

    # Normalise bool-ish columns
    for col in ("executed", "htf_pass", "vol_pass", "hour_filter"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: str(x).strip().lower() in ("true", "1", "yes") if pd.notna(x) else False
            )

    logger.info("Loaded %d signal rows from %s", len(df), _SIGNALS_CSV)
    return df


def _load_journal(days: int | None = None) -> pd.DataFrame | None:
    """Load trade_journal.csv. Returns None if file missing."""
    if not _JOURNAL_CSV.exists():
        logger.warning("Trade journal not found: %s", _JOURNAL_CSV)
        return None

    try:
        df = pd.read_csv(_JOURNAL_CSV, parse_dates=["entry_time", "exit_time"])
    except Exception as exc:
        logger.error("Failed to read %s: %s", _JOURNAL_CSV, exc)
        return None

    if df.empty:
        return df

    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")

    if days is not None and "entry_time" in df.columns:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[df["entry_time"] >= cutoff]

    logger.info("Loaded %d trade rows from %s", len(df), _JOURNAL_CSV)
    return df


def _portfolio_pnl(journal: pd.DataFrame) -> dict[str, float]:
    """Compute portfolio-level PnL from journal."""
    result: dict[str, float] = {
        "total_trades": 0,
        "total_net_pnl_usd": 0.0,
        "total_gross_pnl_usd": 0.0,
        "win_trades": 0,
        "loss_trades": 0,
        "win_rate": 0.0,
        "max_drawdown_usd": 0.0,
    }

    if journal is None or journal.empty:
        return result

    pnl_col = "net_pnl" if "net_pnl" in journal.columns else ("gross_pnl" if "gross_pnl" in journal.columns else None)
    if pnl_col is None:
        return result

    trades = journal.dropna(subset=[pnl_col])
    if trades.empty:
        return result

    pnls = trades[pnl_col].astype(float)
    result["total_trades"] = len(pnls)
    result["total_net_pnl_usd"] = round(float(pnls.sum()), 4)
    if "gross_pnl" in journal.columns:
        result["total_gross_pnl_usd"] = round(float(trades["gross_pnl"].astype(float).sum()), 4)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    result["win_trades"] = len(wins)
    result["loss_trades"] = len(losses)
    result["win_rate"] = round(len(wins) / max(len(pnls), 1), 4)

    # Simple drawdown on cumulative PnL
    cum = pnls.cumsum()
    rolling_peak = cum.cummax()
    dd = rolling_peak - cum
    result["max_drawdown_usd"] = round(float(dd.max()), 4)

    return result


def _per_symbol_stats(
    signals: pd.DataFrame | None,
    journal: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Compute per-symbol comparison stats."""
    symbols: set[str] = set()

    if signals is not None and "symbol" in signals.columns:
        symbols |= set(signals["symbol"].dropna().unique())
    if journal is not None and "symbol" in journal.columns:
        symbols |= set(journal["symbol"].dropna().unique())

    rows: list[dict[str, Any]] = []
    for sym in sorted(symbols):
        row: dict[str, Any] = {"symbol": sym}

        # ── Signal-side stats ──
        if signals is not None and "symbol" in signals.columns:
            sym_sig = signals[signals["symbol"] == sym]
            if "ml_decision" in sym_sig.columns:
                row["live_signal_count"] = int((sym_sig["ml_decision"] != "HOLD").sum())
            else:
                row["live_signal_count"] = len(sym_sig)

            if "executed" in sym_sig.columns:
                row["live_executed_count"] = int(sym_sig["executed"].sum())
            else:
                row["live_executed_count"] = 0

            # Mean cal_prob when signal fired
            if "cal_prob" in sym_sig.columns:
                fired = sym_sig[sym_sig.get("ml_decision", pd.Series(dtype=str)) != "HOLD"] if "ml_decision" in sym_sig.columns else sym_sig
                row["mean_cal_prob"] = round(float(fired["cal_prob"].mean()), 4) if len(fired) > 0 else None

            # Reason skip breakdown
            if "reason_skip" in sym_sig.columns:
                skip_counts = sym_sig["reason_skip"].value_counts().to_dict()
                row["skip_reasons"] = {str(k): int(v) for k, v in skip_counts.items() if pd.notna(k) and k}
        else:
            row["live_signal_count"] = 0
            row["live_executed_count"] = 0

        # ── Journal-side stats ──
        if journal is not None and "symbol" in journal.columns:
            sym_jrn = journal[journal["symbol"] == sym]
            row["live_trade_count"] = len(sym_jrn)

            pnl_col = "net_pnl" if "net_pnl" in sym_jrn.columns else ("gross_pnl" if "gross_pnl" in sym_jrn.columns else None)
            if pnl_col:
                pnls = sym_jrn[pnl_col].astype(float)
                row["symbol_pnl_usd"] = round(float(pnls.sum()), 4)
                wins = pnls[pnls > 0]
                row["symbol_win_rate"] = round(len(wins) / max(len(pnls), 1), 4)
        else:
            row["live_trade_count"] = 0

        # ── Drift flag ──
        expected = row.get("live_executed_count", 0)
        actual = row.get("live_trade_count", 0)
        drift_ratio = abs(actual - expected) / max(expected, 1)
        row["drift_ratio"] = round(drift_ratio, 4)
        row["drift_flag"] = drift_ratio > _DRIFT_THRESHOLD

        rows.append(row)

    return rows


def _print_summary(stats: list[dict[str, Any]], portfolio: dict[str, float]) -> None:
    print("\n" + "=" * 90)
    print("SHADOW RUN COMPARISON REPORT")
    print("=" * 90)

    hdr = f"{'Symbol':<14} {'Signals':>8} {'Executed':>9} {'Trades':>8} {'Drift':>7} {'Flag':>5} {'PnL USD':>10} {'WinRate':>9}"
    print(hdr)
    print("-" * 90)

    for row in stats:
        sym = row["symbol"]
        sigs = row.get("live_signal_count", 0)
        exec_ = row.get("live_executed_count", 0)
        trades = row.get("live_trade_count", 0)
        drift = row.get("drift_ratio", 0.0)
        flag = "YES" if row.get("drift_flag") else "-"
        pnl = row.get("symbol_pnl_usd", 0.0)
        wr = row.get("symbol_win_rate", 0.0)
        print(
            f"{sym:<14} {sigs:>8} {exec_:>9} {trades:>8} {drift:>7.1%} {flag:>5} {pnl:>10.2f} {wr:>8.1%}"
        )

    print("-" * 90)
    print(f"\nPortfolio PnL (net):   ${portfolio['total_net_pnl_usd']:,.2f}")
    print(f"Portfolio trades:      {portfolio['total_trades']}")
    print(f"Portfolio win rate:    {portfolio['win_rate']:.1%}")
    print(f"Max drawdown (USD):    ${portfolio['max_drawdown_usd']:,.2f}")
    print("=" * 90)


def _append_compare_csv(stats: list[dict[str, Any]], portfolio: dict[str, float]) -> None:
    """Append today's summary line to shadow_compare_daily.csv."""
    _COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now(timezone.utc).isoformat()
    write_header = not _COMPARE_CSV.exists()

    fieldnames = [
        "run_at", "symbol", "live_signal_count", "live_executed_count",
        "live_trade_count", "drift_ratio", "drift_flag", "symbol_pnl_usd",
        "symbol_win_rate", "mean_cal_prob",
    ]

    with open(_COMPARE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in stats:
            row_out = {k: row.get(k, "") for k in fieldnames}
            row_out["run_at"] = now_str
            writer.writerow(row_out)

    logger.info("Appended %d rows to %s", len(stats), _COMPARE_CSV)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare shadow run signals vs live trades")
    parser.add_argument("--days", type=int, default=None, help="Filter to last N days (default: all)")
    parser.add_argument("--summary", action="store_true", help="Print current status only (no CSV write)")
    args = parser.parse_args(argv)

    signals = _load_signals(days=args.days)
    journal = _load_journal(days=args.days)

    if signals is None and journal is None:
        logger.warning("Neither signals nor journal data available. Nothing to compare.")
        return 1

    stats = _per_symbol_stats(signals, journal)
    portfolio = _portfolio_pnl(journal)

    _print_summary(stats, portfolio)

    if not args.summary:
        _append_compare_csv(stats, portfolio)

    # Print any drift warnings
    drifted = [r for r in stats if r.get("drift_flag")]
    if drifted:
        print("\n[DRIFT ALERTS]")
        for r in drifted:
            print(
                f"  {r['symbol']}: executed={r.get('live_executed_count', 0)} "
                f"vs trades={r.get('live_trade_count', 0)} "
                f"(drift {r['drift_ratio']:.1%})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
