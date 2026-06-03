"""
GET /api/performance            — aggregate stats from trade_journal.csv
GET /api/performance/by-hour    — win rate bucketed by hour-of-day
GET /api/performance/by-dow     — win rate bucketed by day-of-week
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from dashboard.api.db.duckdb_client import query_trades

router = APIRouter()

_DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _parse_days(range_str: str) -> int:
    try:
        if range_str.endswith("d"):
            return int(range_str[:-1])
        return int(range_str)
    except Exception:
        return 30


def _compute_stats(trades: list[dict], symbol: Optional[str]) -> dict:
    if symbol and symbol.lower() != "all":
        trades = [t for t in trades if str(t.get("symbol", "")).upper() == symbol.upper()]

    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "pnl_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "by_symbol": [],
        }

    def _pnl(t: dict) -> float:
        for k in ("pnl_usd", "pnl", "profit_usd", "profit"):
            v = t.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 0.0

    pnls = [_pnl(t) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    total_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    # Sharpe (annualised, assuming daily returns)
    if len(pnls) > 1:
        mean_p = total_pnl / len(pnls)
        variance = sum((p - mean_p) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(variance) if variance > 0 else 0.0
        sharpe = (mean_p / std * math.sqrt(365)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Per-symbol breakdown
    sym_map: dict[str, list[float]] = {}
    for t, p in zip(trades, pnls):
        sym = str(t.get("symbol", "UNKNOWN"))
        sym_map.setdefault(sym, []).append(p)

    by_symbol = []
    for sym, sym_pnls in sym_map.items():
        sym_wins = [p for p in sym_pnls if p > 0]
        by_symbol.append({
            "symbol": sym,
            "trades": len(sym_pnls),
            "win_rate": round(len(sym_wins) / len(sym_pnls), 4),
            "pnl_usd": round(sum(sym_pnls), 4),
        })

    return {
        "trades": len(trades),
        "win_rate": round(win_rate, 4),
        "pnl_pct": round(total_pnl, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "by_symbol": by_symbol,
    }


@router.get("/api/performance")
async def get_performance(
    range: str = Query("30d"),
    symbol: Optional[str] = Query("all"),
) -> dict:
    days = _parse_days(range)
    trades = await query_trades(days=days)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _compute_stats, trades, symbol)


@router.get("/api/performance/by-hour")
async def get_performance_by_hour() -> list[dict]:
    trades = await query_trades(days=90)
    if not trades:
        return []

    buckets: dict[int, list[float]] = {h: [] for h in range(24)}

    def _pnl(t: dict) -> float:
        for k in ("pnl_usd", "pnl", "profit_usd", "profit"):
            v = t.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 0.0

    for t in trades:
        exit_time = t.get("exit_time", "")
        try:
            hour = int(str(exit_time)[11:13])
            buckets[hour].append(_pnl(t))
        except Exception:
            pass

    result = []
    for hour, pnls in sorted(buckets.items()):
        wins = [p for p in pnls if p > 0]
        result.append({
            "hour": hour,
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
            "count": len(pnls),
        })
    return result


@router.get("/api/performance/pnl-histogram")
async def get_pnl_histogram(bins: int = Query(default=20, ge=5, le=100), days: int = Query(default=90, ge=1, le=365)) -> dict:
    """Histogram of PnL per trade (USD). Returns bin edges + counts."""
    trades = await query_trades(days=days)
    if not trades:
        return {"bin_edges": [], "counts": [], "stats": {}}

    def _pnl(t: dict) -> float:
        for k in ("pnl_usd", "pnl", "profit_usd", "profit"):
            v = t.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 0.0

    pnls = [_pnl(t) for t in trades]
    pnls = [p for p in pnls if p == p]  # filter NaN
    if not pnls:
        return {"bin_edges": [], "counts": [], "stats": {}}

    lo, hi = min(pnls), max(pnls)
    if lo == hi:
        return {"bin_edges": [lo], "counts": [len(pnls)], "stats": {"mean": lo, "median": lo, "n": len(pnls)}}

    step = (hi - lo) / bins
    edges = [lo + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for p in pnls:
        idx = min(int((p - lo) / step), bins - 1)
        counts[idx] += 1

    pnls_sorted = sorted(pnls)
    median = pnls_sorted[len(pnls_sorted) // 2]
    mean = sum(pnls) / len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "bin_edges": [round(e, 2) for e in edges],
        "counts": counts,
        "stats": {
            "n": len(pnls),
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": round(lo, 2),
            "max": round(hi, 2),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
        },
    }


@router.get("/api/performance/rolling-sharpe")
async def get_rolling_sharpe(window: int = Query(default=20, ge=5, le=200), days: int = Query(default=90, ge=1, le=365)) -> list[dict]:
    """Rolling Sharpe ratio over last N trades. Annualized assuming ~250 trade days."""
    from datetime import datetime
    import math as _m

    trades = await query_trades(days=days)
    if not trades:
        return []

    def _pnl(t: dict) -> float:
        for k in ("pnl_usd", "pnl", "profit_usd", "profit"):
            v = t.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 0.0

    rows = []
    for t in trades:
        try:
            dt = datetime.fromisoformat(str(t.get("exit_time", "")).replace("Z", "+00:00"))
            rows.append((dt, _pnl(t)))
        except Exception:
            pass
    rows.sort(key=lambda x: x[0])
    if len(rows) < window:
        return []

    out: list[dict] = []
    for i in range(window, len(rows) + 1):
        sub = [p for _, p in rows[i - window:i]]
        mean = sum(sub) / len(sub)
        var = sum((x - mean) ** 2 for x in sub) / max(len(sub) - 1, 1)
        std = _m.sqrt(var)
        sharpe = (mean / std * _m.sqrt(252)) if std > 0 else 0.0
        out.append({
            "ts": rows[i - 1][0].isoformat(),
            "sharpe": round(sharpe, 3),
            "n": len(sub),
        })
    return out


@router.get("/api/performance/by-dow")
async def get_performance_by_dow() -> list[dict]:
    trades = await query_trades(days=90)
    if not trades:
        return []

    buckets: dict[int, list[float]] = {d: [] for d in range(7)}

    def _pnl(t: dict) -> float:
        for k in ("pnl_usd", "pnl", "profit_usd", "profit"):
            v = t.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return 0.0

    for t in trades:
        exit_time = t.get("exit_time", "")
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00"))
            dow = dt.weekday()  # 0=Monday
            buckets[dow].append(_pnl(t))
        except Exception:
            pass

    result = []
    for dow, pnls in sorted(buckets.items()):
        wins = [p for p in pnls if p > 0]
        result.append({
            "dow": dow,
            "day_name": _DOW_NAMES[dow],
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
            "count": len(pnls),
        })
    return result
