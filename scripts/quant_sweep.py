#!/usr/bin/env python3
"""Quantitative sweep: train per-symbol XGB on Binance OHLCV, rank by edge @ prob>=0.70."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from strategy.quant_features import (  # noqa: E402
    DEFAULT_TAKER_FEE_RATE,
    MIN_OHLC_ROWS,
    QUANT_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("quant_sweep")

# Producción = ETH/USDT único (main.py WATCHLIST); sweep default solo ETH.
TOP10: list[str] = ["ETH/USDT"]

BUY_THRESHOLD = 0.70
_TRAIN_RATIO = 0.80
_HORIZON_DEFAULT = 5


def _round_trip_cost() -> float:
    env = _REPO / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("QUANT_SWEEP_SPREAD_PROXY="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    break
    return 2.0 * DEFAULT_TAKER_FEE_RATE + 0.0005


def fetch_ohlcv_ccxt(symbol: str, timeframe: str = "15m", limit: int = 3000) -> pd.DataFrame:
    import ccxt  # noqa: PLC0415

    ex = ccxt.binance(
        {
            "enableRateLimit": True,
            "timeout": 60_000,
            "options": {"defaultType": "spot", "recvWindow": 10_000},
        }
    )
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def build_dataset(df: pd.DataFrame, horizon: int, cost: float) -> pd.DataFrame:
    feat = add_quant_features(df)
    feat["timestamp"] = pd.to_datetime(df["timestamp"].values, utc=True)
    feat["label"] = forward_return_label(feat["close"], horizon, cost)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"]).reset_index(drop=True)
    return feat


def train_ts_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> XGBClassifier:
    """Fit XGB on full train_df after optional TimeSeriesSplit sanity CV."""
    X = train_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int32)
    if len(X) < MIN_OHLC_ROWS + 50:
        raise ValueError(f"too few train rows: {len(X)}")

    # n_jobs=1: avoids Windows oversubscription / long stalls on some machines
    params = dict(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    tsc = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 500)))
    scores: list[float] = []
    for tr_idx, va_idx in tsc.split(X):
        if len(tr_idx) < 200:
            continue
        m = XGBClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[va_idx])
        acc = float((pred == y[va_idx]).mean())
        scores.append(acc)
    if scores:
        logger.info("TS-CV acc folds: %s (mean=%.4f)", ", ".join(f"{s:.4f}" for s in scores), float(np.mean(scores)))

    final = XGBClassifier(**params)
    final.fit(X, y)
    return final


def edge_audit(
    test_df: pd.DataFrame,
    model: XGBClassifier,
    threshold: float,
    horizon: int,
    cost: float,
) -> dict[str, float | int]:
    """Metrics only where predicted prob(class 1) >= threshold."""
    X = test_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    proba = model.predict_proba(X)[:, 1]
    ts = test_df["timestamp"] if "timestamp" in test_df.columns else None

    fwd_ret = test_df["close"].shift(-horizon) / test_df["close"] - 1.0
    pnl = fwd_ret - cost  # simulated long hold horizon bars

    mask = proba >= threshold
    n_sig = int(mask.sum())
    if n_sig == 0:
        return {
            "signals": 0,
            "win_rate": 0.0,
            "signals_per_week": 0.0,
            "profit_factor": float("nan"),
            "expectancy": float("nan"),
        }

    pnl_sel = pnl[mask].dropna()
    if pnl_sel.empty:
        return {
            "signals": n_sig,
            "win_rate": 0.0,
            "signals_per_week": 0.0,
            "profit_factor": float("nan"),
            "expectancy": float("nan"),
        }

    wins = pnl_sel[pnl_sel > 0]
    losses = pnl_sel[pnl_sel <= 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = gp / gl if gl > 1e-12 else float("inf") if gp > 0 else float("nan")
    wr = float((pnl_sel > 0).mean())
    exp = float(pnl_sel.mean())

    weeks = 1.0
    if ts is not None:
        t_valid = ts[mask]
        if len(t_valid) > 1:
            dt = (t_valid.max() - t_valid.min()).total_seconds() / (86400.0 * 7.0)
            weeks = max(dt, 1e-6)

    sig_week = n_sig / weeks
    return {
        "signals": n_sig,
        "win_rate": wr,
        "signals_per_week": float(sig_week),
        "profit_factor": float(pf) if math.isfinite(pf) else float("nan"),
        "expectancy": exp,
    }


def symbol_to_filename(sym: str, version: str = "v1") -> str:
    return f"{sym.replace('/', '_')}_{version}.json"


def patch_architecture(winner: str, model_basename: str) -> None:
    """Update WATCHLIST, model path, ALLOWED_SYMBOLS, SYMBOL_MAP."""
    main_py = _REPO / "main.py"
    ml_py = _REPO / "strategy" / "ml_predictor.py"
    mt5_py = _REPO / "execution" / "mt5_executor.py"

    main_txt = main_py.read_text(encoding="utf-8")
    main_txt = re.sub(
        r"^WATCHLIST:\s*list\[str\]\s*=\s*\[.*?\]",
        f'WATCHLIST: list[str] = ["{winner}"]',
        main_txt,
        count=1,
        flags=re.M,
    )
    main_txt = re.sub(
        r'^(_MODEL_PATH\s*=\s*Path\(__file__\)\.parent\s*/\s*"models"\s*/\s*")([^"]+)("\))',
        rf'\1{model_basename}\3',
        main_txt,
        count=1,
        flags=re.M,
    )
    main_py.write_text(main_txt, encoding="utf-8")

    ml_txt = ml_py.read_text(encoding="utf-8")
    ml_txt = re.sub(
        r"^ALLOWED_SYMBOLS:\s*tuple\[str,\s*\.\.\.\]\s*=\s*\(.*?\)",
        f'ALLOWED_SYMBOLS: tuple[str, ...] = ("{winner}",)',
        ml_txt,
        count=1,
        flags=re.M,
    )
    ml_txt = re.sub(
        r'^(_XGB_MODEL_PATH\s*=\s*Path\(__file__\)\.resolve\(\)\.parent\.parent\s*/\s*"models"\s*/\s*")([^"]+)("\))',
        rf'\1{model_basename}\3',
        ml_txt,
        count=1,
        flags=re.M,
    )
    ml_py.write_text(ml_txt, encoding="utf-8")

    # Ensure SYMBOL_MAP has winner (Admirals *USD-T pattern)
    mt5_txt = mt5_py.read_text(encoding="utf-8")
    base = winner.split("/")[0]
    admirals = f"{base}USD-T"
    needle = f'"{winner}"'
    if needle not in mt5_txt:
        insert_after = "SYMBOL_MAP: dict[str, str] = {"
        idx = mt5_txt.find(insert_after)
        if idx == -1:
            logger.warning("Could not find SYMBOL_MAP block - skip MT5 patch.")
        else:
            line = f'    "{winner}":    "{admirals}",\n'
            nl = mt5_txt.find("\n", idx) + 1
            mt5_txt = mt5_txt[:nl] + line + mt5_txt[nl:]
            mt5_py.write_text(mt5_txt, encoding="utf-8")


def run_one_symbol(symbol: str, args: argparse.Namespace) -> dict[str, object]:
    cost = _round_trip_cost()
    df = fetch_ohlcv_ccxt(symbol, timeframe=args.timeframe, limit=args.limit)
    full = build_dataset(df, args.horizon, cost)

    split_i = int(len(full) * _TRAIN_RATIO)
    split_i = max(split_i, MIN_OHLC_ROWS + 100)
    split_i = min(split_i, len(full) - 50)
    train_df = full.iloc[:split_i].copy()
    test_df = full.iloc[split_i:].reset_index(drop=True)

    model = train_ts_cv(train_df, n_splits=args.cv_splits)
    metrics = edge_audit(test_df, model, BUY_THRESHOLD, args.horizon, cost)

    return {
        "symbol": symbol,
        "model": model,
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "round_trip_cost": cost,
    }


def rank_key(row: dict[str, object]) -> float:
    m = row["metrics"]
    assert isinstance(m, dict)
    pf = m.get("profit_factor", float("nan"))
    if isinstance(pf, float) and math.isfinite(pf):
        return float(pf)
    return float("-inf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantitative sweep (default universe: ETH/USDT).")
    parser.add_argument("--timeframe", default="15m", help="ccxt timeframe (default 15m)")
    parser.add_argument("--limit", type=int, default=3000, help="candles per symbol")
    parser.add_argument("--horizon", type=int, default=_HORIZON_DEFAULT, help="forward bars for label & PnL")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--no-patch", action="store_true", help="do not modify main/ml_predictor/mt5")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    cost = _round_trip_cost()
    logger.info(
        "Round-trip friction model: %.6f (fees 2x%.5f + spread proxy)",
        cost,
        DEFAULT_TAKER_FEE_RATE,
    )

    rows: list[dict[str, object]] = []
    for sym in TOP10:
        logger.info("=== %s ===", sym)
        try:
            pack = run_one_symbol(sym, args)
            rows.append(pack)
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAILED %s: %s", sym, exc)
            rows.append(
                {
                    "symbol": sym,
                    "model": None,
                    "metrics": {
                        "signals": 0,
                        "win_rate": 0.0,
                        "signals_per_week": 0.0,
                        "profit_factor": float("nan"),
                        "expectancy": float("nan"),
                        "error": str(exc),
                    },
                    "train_rows": 0,
                    "test_rows": 0,
                    "round_trip_cost": cost,
                },
            )

    rows.sort(key=rank_key, reverse=True)

    table = Table(title="Quant sweep - ranked by Profit Factor (test, prob>=0.70)")
    table.add_column("Rank", justify="right")
    table.add_column("Symbol")
    table.add_column("PF", justify="right")
    table.add_column("Expectancy", justify="right")
    table.add_column("WinRate", justify="right")
    table.add_column("Sig/wk", justify="right")
    table.add_column("Signals", justify="right")

    for i, pack in enumerate(rows, start=1):
        sym = str(pack["symbol"])
        m = pack["metrics"]
        assert isinstance(m, dict)
        pf = m.get("profit_factor", float("nan"))
        pf_s = f"{float(pf):.3f}" if isinstance(pf, (int, float)) and math.isfinite(float(pf)) else "nan"
        ex = m.get("expectancy", float("nan"))
        ex_s = f"{float(ex):.6f}" if isinstance(ex, (int, float)) and math.isfinite(float(ex)) else "nan"
        wr = float(m.get("win_rate", 0.0))
        sw = float(m.get("signals_per_week", 0.0))
        ns = int(m.get("signals", 0))
        table.add_row(str(i), sym, pf_s, ex_s, f"{wr*100:.1f}%", f"{sw:.2f}", str(ns))

    console.print(table)

    # Winner: PF > 1.2 and expectancy > 0
    winner_pack: dict[str, object] | None = None
    for pack in rows:
        m = pack["metrics"]
        assert isinstance(m, dict)
        pf = float(m.get("profit_factor", float("nan")))
        ex = float(m.get("expectancy", float("nan")))
        if math.isfinite(pf) and math.isfinite(ex) and pf > 1.2 and ex > 0:
            winner_pack = pack
            break
    if winner_pack is None and rows:
        winner_pack = rows[0]
        console.print(
            "[yellow]No symbol passed PF>1.2 & positive expectancy - using top PF rank as operational pick.[/yellow]"
        )

    models_dir = _REPO / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if winner_pack and winner_pack.get("model") is not None:
        wsym = str(winner_pack["symbol"])
        fname = symbol_to_filename(wsym, "v1")
        out_path = models_dir / fname
        mdl = winner_pack["model"]
        assert isinstance(mdl, XGBClassifier)
        mdl.save_model(str(out_path))
        console.print(f"[green]Saved winner model -> {out_path}[/green]")
        report = {
            "winner": wsym,
            "model_file": str(out_path),
            "metrics": winner_pack["metrics"],
            "qualified_strict": bool(
                isinstance(winner_pack["metrics"], dict)
                and float(winner_pack["metrics"].get("profit_factor", 0)) > 1.2
                and float(winner_pack["metrics"].get("expectancy", -1)) > 0
            ),
            "ranking": [
                {
                    "symbol": str(r["symbol"]),
                    **{k: v for k, v in dict(r["metrics"]).items()},  # type: ignore[arg-type]
                }
                for r in rows
            ],
        }
        if args.json_out:
            args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        logs_dir = _REPO / "logs"
        logs_dir.mkdir(exist_ok=True)
        (logs_dir / "quant_sweep_last.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        if not args.no_patch:
            patch_architecture(wsym, fname)
            console.print("[green]Patched main.py, strategy/ml_predictor.py, execution/mt5_executor.py[/green]")
    else:
        console.print("[red]No model saved - all symbols failed.[/red]")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
