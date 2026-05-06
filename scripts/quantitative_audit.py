#!/usr/bin/env python3
"""Quantitative audit: trade journal, XGB artifact, risk/signal constants, optional Binance sweep, pytest."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _float_cell(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_ts(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


@dataclass
class ClosedTrade:
    symbol: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    ml_conf: float | None
    sentiment: float | None
    exit_reason: str
    pnl_usdt: float | None
    pnl_percent: float | None


def audit_trade_journal(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows_total": 0,
        "errors": [],
    }
    if not path.exists():
        return out

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    out["rows_total"] = len(rows)

    # Chronological replay: stack pending BUY per symbol
    pending: dict[str, dict[str, Any]] = {}
    closed: list[ClosedTrade] = []
    orphan_sells = 0

    def _sort_key(r: dict[str, str]) -> datetime:
        t = _parse_ts(r.get("timestamp", ""))
        if t is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        if t.tzinfo is None:
            return t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)

    sorted_rows = sorted(rows, key=_sort_key)

    for r in sorted_rows:
        action = (r.get("action") or "").strip().upper()
        sym = (r.get("symbol") or "").strip()
        ts = r.get("timestamp", "")
        if action == "BUY":
            pending[sym] = {
                "ts": ts,
                "price": _float_cell(r.get("execution_price", "")),
                "ml": _float_cell(r.get("ml_confidence_at_entry", "")),
                "sent": _float_cell(r.get("sentiment_score_at_entry", "")),
            }
        elif action == "SELL":
            if sym not in pending:
                orphan_sells += 1
                continue
            op = pending.pop(sym)
            ep = _float_cell(r.get("execution_price", ""))
            closed.append(
                ClosedTrade(
                    symbol=sym,
                    entry_ts=op["ts"],
                    exit_ts=ts,
                    entry_price=float(op["price"] or 0.0),
                    exit_price=float(ep or 0.0),
                    ml_conf=op.get("ml"),
                    sentiment=op.get("sent"),
                    exit_reason=(r.get("exit_reason") or "").strip(),
                    pnl_usdt=_float_cell(r.get("pnl_usdt", "")),
                    pnl_percent=_float_cell(r.get("pnl_percent", "")),
                )
            )

    out["orphan_sell_rows"] = orphan_sells
    out["open_positions_unmatched"] = list(pending.keys())
    out["closed_trades_count"] = len(closed)

    pnls = [c.pnl_usdt for c in closed if c.pnl_usdt is not None]
    pcts = [c.pnl_percent for c in closed if c.pnl_percent is not None]

    def _safe_mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    out["pnl_usdt"] = {
        "count": len(pnls),
        "sum": round(sum(pnls), 6) if pnls else 0.0,
        "mean": round(_safe_mean(pnls), 6) if pnls else 0.0,
        "median": round(sorted(pnls)[len(pnls) // 2], 6) if pnls else 0.0,
        "std": round(
            math.sqrt(sum((x - _safe_mean(pnls)) ** 2 for x in pnls) / max(len(pnls) - 1, 1)),
            6,
        )
        if len(pnls) > 1
        else 0.0,
        "min": round(min(pnls), 6) if pnls else 0.0,
        "max": round(max(pnls), 6) if pnls else 0.0,
        "win_rate_pct": round(len(wins) / len(pnls) * 100, 4) if pnls else 0.0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 6) if losses and sum(losses) != 0 else None,
        "expectancy": round(_safe_mean(pnls), 6) if pnls else 0.0,
    }

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    out["pnl_usdt"]["gross_profit"] = round(gross_profit, 6)
    out["pnl_usdt"]["gross_loss"] = round(gross_loss, 6)

    # Equity curve & max drawdown (USDT)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    out["equity_usdt"] = {
        "final_cumulative_pnl": round(equity, 6),
        "max_drawdown_usdt": round(max_dd, 6),
    }

    if pcts:
        m_pct = _safe_mean(pcts)
        var = sum((x - m_pct) ** 2 for x in pcts) / max(len(pcts) - 1, 1)
        std_pct = math.sqrt(var) if len(pcts) > 1 else 0.0
        out["pnl_percent_stats"] = {
            "count": len(pcts),
            "mean_pct": round(m_pct, 6),
            "std_pct": round(std_pct, 6),
            "sharpe_like_mean_over_std": round(m_pct / std_pct, 6) if std_pct > 1e-9 else None,
        }

    by_sym: dict[str, list[float]] = defaultdict(list)
    by_exit: dict[str, list[float]] = defaultdict(list)
    for c in closed:
        if c.pnl_usdt is not None:
            by_sym[c.symbol].append(c.pnl_usdt)
            by_exit[c.exit_reason or "UNKNOWN"].append(c.pnl_usdt)

    def _agg(name: str, d: dict[str, list[float]]) -> dict[str, Any]:
        r: dict[str, Any] = {}
        for k, vals in sorted(d.items()):
            w = sum(1 for x in vals if x > 0)
            r[k] = {
                "trades": len(vals),
                "sum_pnl": round(sum(vals), 6),
                "win_rate_pct": round(w / len(vals) * 100, 2) if vals else 0.0,
            }
        return r

    out["by_symbol"] = _agg("symbol", by_sym)
    out["by_exit_reason"] = _agg("exit", by_exit)

    # ML confidence buckets vs outcome
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: {"wins": 0, "n": 0})
    for c in closed:
        if c.pnl_usdt is None or c.ml_conf is None:
            continue
        b = int(c.ml_conf * 10) / 10.0  # 0.6, 0.7, ...
        key = f"{b:.1f}-{b+0.1:.1f}"
        buckets[key]["n"] += 1
        if c.pnl_usdt > 0:
            buckets[key]["wins"] += 1
    out["ml_confidence_buckets"] = {
        k: {
            "n": int(v["n"]),
            "win_rate_pct": round(v["wins"] / v["n"] * 100, 2) if v["n"] else 0.0,
        }
        for k, v in sorted(buckets.items())
    }

    out["closed_trades_sample"] = [asdict(c) for c in closed[:5]]
    return out


def audit_xgb_model(model_path: Path) -> dict[str, Any]:
    r: dict[str, Any] = {"path": str(model_path), "exists": model_path.exists()}
    if not model_path.exists():
        return r
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        r["parse_error"] = str(e)
        return r
    learner = data.get("learner", {})
    obj = learner.get("objective", {})
    gb = learner.get("gradient_booster", {})
    model = gb.get("model", {})
    trees = model.get("trees", [])
    r["objective"] = obj.get("name")
    r["booster"] = gb.get("name")
    lmp = learner.get("learner_model_param", {})
    r["num_feature"] = lmp.get("num_feature")
    r["base_score"] = lmp.get("base_score")
    r["tree_count"] = len(trees)
    if trees:
        depths: list[int] = []
        for t in trees:
            lc = t.get("left_children", [])
            n = len(lc)
            depth = [0] * n
            for i in range(n):
                l, rr = lc[i], t.get("right_children", [])[i]
                if l != -1:
                    depth[l] = depth[i] + 1
                if rr != -1:
                    depth[rr] = depth[i] + 1
            depths.append(max(depth) if depth else 0)
        r["max_depth_observed"] = max(depths)
        r["avg_depth"] = round(sum(depths) / len(depths), 4)
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415

        m = XGBClassifier()
        m.load_model(str(model_path))
        b = m.get_booster()
        r["feature_importance_gain"] = b.get_score(importance_type="gain")
        r["feature_importance_weight"] = b.get_score(importance_type="weight")
    except Exception as e:  # noqa: BLE001
        r["xgboost_import_error"] = str(e)
    try:
        import xgboost as xgb  # noqa: PLC0415

        r["xgboost_version"] = xgb.__version__
    except Exception:
        pass
    return r


def snapshot_constants(repo: Path) -> dict[str, Any]:
    """Read key numeric constants without importing heavy side effects."""
    out: dict[str, Any] = {}
    mp = repo / "strategy" / "ml_predictor.py"
    if mp.exists():
        txt = mp.read_text(encoding="utf-8")
        for name in (
            "BUY_PROB_THRESHOLD",
            "_SELL_PROB_THRESHOLD",
            "_SELL_SENTIMENT_THRESHOLD",
            "_ADX_TREND_THRESHOLD",
            "_ADX_RANGE_THRESHOLD",
            "_PREDICTION_HORIZON",
            "_SMA_PERIOD",
            "_RSI_PERIOD",
            "_MOMENTUM_PERIOD",
        ):
            m = re.search(rf"^{re.escape(name)}[^=]*=\s*([0-9.+-]+)", txt, re.M)
            if m:
                out[name] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    rm = repo / "risk" / "risk_manager.py"
    if rm.exists():
        txt = rm.read_text(encoding="utf-8")
        for name in (
            "INITIAL_SL",
            "ACTIVATION_PCT",
            "TRAILING_DISTANCE",
            "LEVERAGE",
            "RISK_PER_TRADE",
            "MAX_DAILY_LOSS_PCT",
            "MAX_POSITIONS",
            "MAX_PORTFOLIO_DD_PCT",
            "BASE_SL",
            "BASE_ACTIVATION_PCT",
            "BASE_TRAILING_DISTANCE",
            "_ACTIVATION_PCT_MIN",
            "_SENTIMENT_LOW_THRESHOLD",
            "_SENTIMENT_HIGH_THRESHOLD",
        ):
            m = re.search(rf"^{re.escape(name)}:\s*(?:float|int)\s*=\s*([0-9.]+)", txt, re.M)
            if m:
                val = m.group(1)
                out[name] = float(val) if "." in val else int(val)
    pe = repo / "execution" / "paper_executor.py"
    if pe.exists():
        txt = pe.read_text(encoding="utf-8")
        for name in (
            "_DEFAULT_TTL_HOURS",
            "_PROFIT_STAGNATION_TIMEOUT_HOURS",
            "_ML_EXIT_CONFIDENCE_FACTOR",
            "ATR_SL_MULTIPLIER",
            "ATR_TRAILING_MULTIPLIER",
            "_TAKER_FEE_RATE",
        ):
            m = re.search(rf"^{re.escape(name)}\s*:\s*float\s*=\s*([0-9.]+)", txt, re.M)
            if m:
                out[name] = float(m.group(1))
    bc = repo / "bot" / "constants.py"
    if bc.exists():
        txt = bc.read_text(encoding="utf-8")
        m = re.search(r"^NEWS_FILTER_VOLATILITY_THRESHOLD:\s*float\s*=\s*([0-9.]+)", txt, re.M)
        if m:
            out["NEWS_FILTER_VOLATILITY_THRESHOLD"] = float(m.group(1))
        m = re.search(r"^NEWS_FILTER_HOLD_MINUTES:\s*int\s*=\s*([0-9]+)", txt, re.M)
        if m:
            out["NEWS_FILTER_HOLD_MINUTES"] = int(m.group(1))
    se = repo / "bot" / "signal_emitter.py"
    if se.exists():
        m = re.search(r"async def signal_emitter\([\s\S]*?interval:\s*int\s*=\s*([0-9]+)", se.read_text(encoding="utf-8"))
        if m:
            out["signal_emitter_interval_s"] = int(m.group(1))
    return {"constants": out}


def run_buy_threshold_sweep(repo: Path) -> dict[str, Any]:
    """Optional: fetch Binance 15m, train XGB like run_backtest, sweep buy_threshold on test split."""
    result: dict[str, Any] = {"status": "skipped", "reason": ""}
    try:
        sys.path.insert(0, str(repo))
        from backtest.run_backtest import (  # noqa: PLC0415
            _FEATURE_COLS,
            _TRAIN_RATIO,
            build_features,
            fetch_ohlcv,
            simulate_backtest,
        )
        from xgboost import XGBClassifier  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        result["reason"] = f"import_failed: {e}"
        return result

    try:
        df = fetch_ohlcv("ETH/USDT")
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
        thresholds = [0.50, 0.55, 0.58, 0.62, 0.65, 0.68, 0.72]
        sweep = []
        for th in thresholds:
            st = simulate_backtest(test_df, model, buy_threshold=th, symbol="ETH/USDT")
            sweep.append({"buy_threshold": th, **st})
        result["status"] = "ok"
        result["symbol"] = "ETH/USDT"
        result["train_rows"] = int(len(train_df))
        result["test_rows"] = int(len(test_df))
        result["sweep"] = sweep
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["reason"] = str(e)
    return result


def run_pytest(repo: Path) -> dict[str, Any]:
    exe = sys.executable
    cmd = [
        exe,
        "-m",
        "pytest",
        "tests/",
        "-q",
        "--tb=no",
        "--ignore=tests/manual_order_test.py",
    ]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "exit_code": p.returncode,
            "stdout_tail": p.stdout[-4000:] if p.stdout else "",
            "stderr_tail": p.stderr[-2000:] if p.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -1, "error": str(e)}


def main() -> int:
    repo = _repo_root()
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = repo / "logs" / f"quant_audit_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repo": str(repo),
        "trade_journal": audit_trade_journal(repo / "logs" / "trade_journal.csv"),
        "xgb_model": audit_xgb_model(repo / "models" / "xgb_live.json"),
        "code_constants": snapshot_constants(repo),
        "pytest": run_pytest(repo),
    }

    if os.environ.get("AUDIT_SKIP_BINANCE", "").strip().lower() not in ("1", "true", "yes"):
        report["binance_threshold_sweep"] = run_buy_threshold_sweep(repo)
    else:
        report["binance_threshold_sweep"] = {"status": "skipped", "reason": "AUDIT_SKIP_BINANCE"}

    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"written": str(out_path), "summary": _human_summary(report)}, indent=2))
    return 0


def _human_summary(report: dict[str, Any]) -> dict[str, Any]:
    tj = report.get("trade_journal", {})
    pnl = tj.get("pnl_usdt", {})
    xm = report.get("xgb_model", {})
    sw = report.get("binance_threshold_sweep", {})
    return {
        "journal_closed_trades": tj.get("closed_trades_count"),
        "journal_win_rate_pct": pnl.get("win_rate_pct"),
        "journal_sum_pnl_usdt": pnl.get("sum"),
        "journal_max_dd_usdt": tj.get("equity_usdt", {}).get("max_drawdown_usdt"),
        "journal_profit_factor": pnl.get("profit_factor"),
        "xgb_trees": xm.get("tree_count"),
        "xgb_objective": xm.get("objective"),
        "binance_sweep_status": sw.get("status"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
