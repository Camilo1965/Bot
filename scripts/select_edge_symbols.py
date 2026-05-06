#!/usr/bin/env python3
"""Evaluate per-symbol backtest edge and export VALID universe.

Uses the same simulation as backtest.run_backtest (SL/TTL/trailing/fees).
Each symbol trains its own XGB on the same chronological split (~1y Binance 15m).

Usage
-----
    python scripts/select_edge_symbols.py
    python scripts/select_edge_symbols.py --threshold 0.70
    python scripts/select_edge_symbols.py --sweep
    python scripts/select_edge_symbols.py --output edge_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from xgboost import XGBClassifier

# Repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backtest.run_backtest import (  # noqa: E402
    _FEATURE_COLS,
    _TRAIN_RATIO,
    build_features,
    collect_trades,
    fetch_ohlcv,
    metrics_from_trades,
)

logger = logging.getLogger(__name__)

# --- Universe (operación única ETH/USDT; ampliar lista si vuelves a multi-par) ---

SYMBOLS: list[str] = ["ETH/USDT"]

MIN_TRADES = 30
MIN_PF = 1.1
MIN_EXPECTANCY = 0.0  # classify VALID only if expectancy > this (strictly > 0 for edge)

SWEEP_THRESHOLDS = [0.65, 0.68, 0.70, 0.72, 0.75]

DEFAULT_OUTPUT = _REPO / "edge_results.json"


def _json_float(x: float) -> float | str | None:
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
    return x


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return _json_float(obj)
    return obj


def prepare_symbol(symbol: str) -> dict:
    """Fetch data, train per-symbol XGB; reuse for multiple BUY thresholds."""
    try:
        df = fetch_ohlcv(symbol)
        feat_df = build_features(df)
        split_idx = int(len(feat_df) * _TRAIN_RATIO)
        train_df = feat_df.iloc[:split_idx]
        test_df = feat_df.iloc[split_idx:].reset_index(drop=True)

        X_train = train_df[_FEATURE_COLS]
        y_train = train_df["label"].astype(int)
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        return {
            "ok": True,
            "symbol": symbol,
            "test_df": test_df,
            "model": model,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prepare failed %s", symbol)
        return {"ok": False, "symbol": symbol, "error": str(exc)}


def metrics_from_pack(pack: dict, buy_threshold: float) -> dict:
    """Simulate trades at *buy_threshold* using pre-trained *pack*."""
    symbol = pack["symbol"]
    row: dict = {"symbol": symbol, "error": pack.get("error")}
    if not pack.get("ok"):
        row.update(
            {
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": float("nan"),
                "net_pnl_pct": 0.0,
                "expectancy_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "train_rows": 0,
                "test_rows": 0,
                "buy_threshold": buy_threshold,
                "status": "REJECT",
            },
        )
        return row

    test_df = pack["test_df"]
    model = pack["model"]
    trades = collect_trades(test_df, model, buy_threshold, symbol)
    m = metrics_from_trades(trades)

    pf = float(m["profit_factor"])
    exp = float(m["expectancy_pct"])
    ntr = int(m["trades"])

    valid = (
        ntr >= MIN_TRADES
        and math.isfinite(pf)
        and pf >= MIN_PF
        and exp > MIN_EXPECTANCY
    )

    row.update(
        {
            "error": None,
            "trades": ntr,
            "win_rate": float(m["win_rate"]),
            "profit_factor": pf,
            "net_pnl_pct": float(m["net_pnl_pct"]),
            "expectancy_pct": exp,
            "avg_win_pct": float(m["avg_win_pct"]),
            "avg_loss_pct": float(m["avg_loss_pct"]),
            "max_drawdown_pct": float(m["max_drawdown_pct"]),
            "train_rows": pack["train_rows"],
            "test_rows": pack["test_rows"],
            "buy_threshold": buy_threshold,
            "status": "VALID" if valid else "REJECT",
        },
    )
    return row


def _rank_key(r: dict) -> tuple:
    pf = r.get("profit_factor", float("nan"))
    pf_k = float(pf) if isinstance(pf, (int, float)) and math.isfinite(float(pf)) else float("-inf")
    return (-pf_k, -float(r.get("expectancy_pct", 0.0)), -float(r.get("net_pnl_pct", 0.0)))


def run_evaluation(
    symbols: list[str],
    buy_threshold: float,
    packs: dict[str, dict] | None = None,
) -> list[dict]:
    if packs is None:
        packs = {}
        for sym in symbols:
            logger.info("Training %s …", sym)
            packs[sym] = prepare_symbol(sym)
    rows: list[dict] = []
    for sym in symbols:
        logger.info("Metrics %s @ BUY > %.2f …", sym, buy_threshold)
        rows.append(metrics_from_pack(packs[sym], buy_threshold))
    rows.sort(key=_rank_key)
    return rows


def run_threshold_sweep(packs: dict[str, dict], symbols: list[str]) -> dict:
    """Reuse prepared models; one threshold loop only runs collect_trades."""
    all_pairs: list[dict] = []
    per_threshold: dict[str, list[dict]] = {}

    for th in SWEEP_THRESHOLDS:
        rows: list[dict] = []
        for sym in symbols:
            rows.append(metrics_from_pack(packs[sym], th))
        rows.sort(key=_rank_key)
        per_threshold[f"{th:.2f}"] = rows
        for r in rows:
            if not r.get("error"):
                all_pairs.append(dict(r))

    passing = [p for p in all_pairs if p.get("status") == "VALID"]
    passing.sort(key=_rank_key)

    best_per_symbol: dict[str, dict] = {}
    for p in passing:
        sym = p["symbol"]
        cur = best_per_symbol.get(sym)
        if cur is None or _rank_key(p) < _rank_key(cur):
            best_per_symbol[sym] = p

    return {
        "per_threshold": per_threshold,
        "all_valid_pairs_ranked": passing,
        "best_per_symbol": list(best_per_symbol.values()),
        "global_best_pair": passing[0] if passing else None,
    }


def split_valid_reject(rows: list[dict]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    reject: list[str] = []
    for r in rows:
        sym = r["symbol"]
        if r.get("error"):
            reject.append(sym)
            continue
        if r.get("status") == "VALID":
            valid.append(sym)
        else:
            reject.append(sym)
    return valid, reject


def print_table(rows: list[dict]) -> None:
    hdr = f"{'SYMBOL':<12} {'TR':>5} {'WR%':>7} {'PF':>8} {'EXP%':>9} {'PNL%':>9} {'STATUS':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("error"):
            print(f"{r['symbol']:<12} {'ERR':>5} {'-':>7} {'-':>8} {'-':>9} {'-':>9} {'REJECT':>8}")
            continue
        pf = r["profit_factor"]
        pf_s = f"{pf:.3f}" if isinstance(pf, (int, float)) and math.isfinite(float(pf)) else str(pf)
        print(
            f"{r['symbol']:<12} {r['trades']:>5} {r['win_rate']:>7.2f} "
            f"{pf_s:>8} {r['expectancy_pct']:>9.4f} {r['net_pnl_pct']:>9.2f} "
            f"{r['status']:>8}",
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Select symbols with real edge (backtest).")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.68,
        help="BUY probability threshold (default 0.68)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default {DEFAULT_OUTPUT})",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help=f"Also run thresholds {SWEEP_THRESHOLDS} and rank symbol+threshold pairs",
    )
    ap.add_argument(
        "--symbols",
        type=str,
        default="",
        help='Comma-separated override of SYMBOLS (e.g. "BTC/USDT,ETH/USDT")',
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else list(SYMBOLS)

    packs = {}
    for sym in symbols:
        packs[sym] = prepare_symbol(sym)

    meta = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "min_trades": MIN_TRADES,
        "min_pf": MIN_PF,
        "min_expectancy": MIN_EXPECTANCY,
        "train_ratio": _TRAIN_RATIO,
        "timeframe": "15m",
        "data_source": "Binance OHLCV via ccxt (same ~1y window per symbol)",
        "buy_threshold": args.threshold,
        "symbols_evaluated": symbols,
    }

    rows = run_evaluation(symbols, args.threshold, packs=packs)
    valid, rejected = split_valid_reject(rows)

    print("\n=== EDGE SELECTION (single threshold) ===\n")
    print_table(rows)
    print(f"\nVALID_SYMBOLS   ({len(valid)}): {valid}")
    print(f"REJECTED_SYMBOLS ({len(rejected)}): {rejected}")

    emit_py = "\n# Paste into config / risk module:\nALLOWED_SYMBOLS = " + repr(valid) + "\n"
    print(emit_py)

    payload: dict = {
        "meta": meta,
        "criteria": {
            "MIN_TRADES": MIN_TRADES,
            "MIN_PF": MIN_PF,
            "MIN_EXPECTANCY": MIN_EXPECTANCY,
            "note": "VALID requires expectancy_pct > MIN_EXPECTANCY (strict)",
        },
        "ranked_results": rows,
        "VALID_SYMBOLS": valid,
        "REJECTED_SYMBOLS": rejected,
        "ALLOWED_SYMBOLS": valid,
    }

    if args.sweep:
        print("\n=== THRESHOLD SWEEP ===\n")
        sweep_payload = run_threshold_sweep(packs, symbols)
        payload["threshold_sweep"] = _sanitize(sweep_payload)
        gb = sweep_payload.get("global_best_pair")
        if gb:
            print(
                f"Best symbol+threshold (among VALID): {gb['symbol']} @ {gb['buy_threshold']:.2f}  "
                f"PF={gb['profit_factor']:.3f}  exp={gb['expectancy_pct']:.4f}  "
                f"net_pnl%={gb['net_pnl_pct']:.2f}",
            )
        else:
            print("No VALID (symbol, threshold) pairs under current criteria.")
        if sweep_payload.get("best_per_symbol"):
            print("\nBest threshold per symbol (among VALID):")
            for b in sorted(sweep_payload["best_per_symbol"], key=_rank_key):
                print(
                    f"  {b['symbol']:<12} th={b['buy_threshold']:.2f}  PF={b['profit_factor']:.3f}  "
                    f"exp={b['expectancy_pct']:.4f}  trades={b['trades']}",
                )

    out_path = Path(args.output)
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
