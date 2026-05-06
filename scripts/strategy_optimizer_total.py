#!/usr/bin/env python3
"""Grid sweep: prob × risk on OHLCV 15m hold-out backtest (single XGB train, many sims)."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from rich.box import ASCII
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_DSA_PATH = Path(__file__).resolve().parent / "deep_strategy_audit.py"
_spec = importlib.util.spec_from_file_location("_deep_strategy_audit_exec", _DSA_PATH)
_dsa = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault(_spec.name, _dsa)
_spec.loader.exec_module(_dsa)

fetch_ohlcv_ccxt = _dsa.fetch_ohlcv_ccxt
build_dataset = _dsa.build_dataset
train_ts_cv = _dsa.train_ts_cv
run_backtest = _dsa.run_backtest
profit_factor = _dsa.profit_factor
max_drawdown = _dsa.max_drawdown

FEE_RATE = float(_dsa.FEE_RATE)
MIN_OHLC_ROWS = int(_dsa.MIN_OHLC_ROWS)
_LABEL_HORIZON = int(_dsa._LABEL_HORIZON)
_TRAIN_RATIO = float(_dsa._TRAIN_RATIO)
_FALLBACK_CONTRACT_SIZE = float(_dsa._FALLBACK_CONTRACT_SIZE)
SYMBOL_DEFAULT = str(_dsa.SYMBOL)

# Grid (user spec)
PROB_GRID = [round(x, 2) for x in np.arange(0.50, 0.801, 0.05)]
RISK_GRID = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]


def _pf_str(pf: float) -> str:
    if not math.isfinite(pf):
        return "inf" if pf > 0 else "nan"
    return f"{pf:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mass parameter sweep (single model train, test-only sim).")
    ap.add_argument("--symbol", default=SYMBOL_DEFAULT)
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--balance", type=float, default=10_000.0)
    ap.add_argument("--contract-size", type=float, default=_FALLBACK_CONTRACT_SIZE)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--train-ratio", type=float, default=_TRAIN_RATIO)
    args = ap.parse_args()

    bar_delta = _dsa._timeframe_to_delta(args.timeframe)
    round_trip_label = 2.0 * FEE_RATE
    console = Console(legacy_windows=False)

    console.print(
        f"[bold]strategy_optimizer_total[/] | {args.symbol} | {args.limit} velas {args.timeframe} | "
        f"train {args.train_ratio:.0%} / test {1 - args.train_ratio:.0%}\n"
        f"TTL 12h | SL min(2*ATR,5pct) HTF SMA200 1h | Trailing 2pct/2pct | fee/side {FEE_RATE}\n"
        f"Grid prob {escape(str(PROB_GRID))} | risk {escape(str(RISK_GRID))}"
    )

    raw = fetch_ohlcv_ccxt(args.symbol, args.timeframe, args.limit)
    full = build_dataset(raw, _LABEL_HORIZON, round_trip_label)

    split_i = int(len(full) * args.train_ratio)
    split_i = max(split_i, MIN_OHLC_ROWS + 100)
    split_i = min(split_i, len(full) - 200)
    train_df = full.iloc[:split_i].copy()
    feat_full = full.reset_index(drop=True)

    console.print(
        f"Filas: total={len(feat_full)} train={len(train_df)} test~{len(feat_full) - split_i}"
    )

    model = train_ts_cv(train_df, n_splits=args.cv_splits)

    rows_out: list[dict[str, float | int]] = []
    for prob in PROB_GRID:
        for risk in RISK_GRID:
            closed, equity_curve = run_backtest(
                feat_full,
                model,
                split_i,
                args.contract_size,
                bar_delta,
                args.balance,
                prob,
                risk,
            )
            nets = np.array([t.net_pnl for t in closed], dtype=float)
            n_tr = len(closed)
            wr = float((nets > 0).mean()) if n_tr else 0.0
            net_sum = float(nets.sum()) if n_tr else 0.0
            eq_series = [args.balance] + equity_curve
            mdd = max_drawdown(eq_series, args.balance)
            pf = profit_factor(closed)
            rows_out.append(
                {
                    "prob": prob,
                    "risk": risk,
                    "win_rate": wr,
                    "trades": n_tr,
                    "max_dd": mdd,
                    "profit_factor": float(pf) if math.isfinite(pf) else pf,
                    "net_pnl": net_sum,
                }
            )

    rows_out.sort(key=lambda r: float(r["net_pnl"]), reverse=True)

    table = Table(
        title="Resultados (NetPnL desc): azul MaxDD<10% | verde mejor NetPnL",
        box=ASCII,
        show_lines=False,
    )
    for col in ("Prob", "Risk", "WinRate", "Trades", "MaxDD", "ProfitFactor", "NetPnL"):
        table.add_column(col, justify="right")

    for i, r in enumerate(rows_out):
        prob = float(r["prob"])
        risk = float(r["risk"])
        wr = float(r["win_rate"])
        n_tr = int(r["trades"])
        mdd = float(r["max_dd"])
        pf_raw = r["profit_factor"]
        pf_v = float(pf_raw) if isinstance(pf_raw, (int, float)) else float("nan")
        net = float(r["net_pnl"])

        top_pnl = i == 0
        low_dd = mdd < 0.10 and not top_pnl
        style = "bold green" if top_pnl else ("bold blue" if low_dd else "")

        pf_disp = _pf_str(pf_v) if math.isfinite(pf_v) else ("inf" if pf_v > 0 else "nan")

        table.add_row(
            Text(f"{prob:.2f}", style=style),
            Text(f"{risk:.4f}", style=style),
            Text(f"{wr * 100:.2f}%", style=style),
            Text(str(n_tr), style=style),
            Text(f"{mdd * 100:.2f}%", style=style),
            Text(pf_disp, style=style),
            Text(f"{net:.2f}", style=style),
        )

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
