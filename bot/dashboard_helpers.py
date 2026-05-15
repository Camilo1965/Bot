"""Pure helpers for the Rich mega-dashboard (no MT5 / asyncio imports)."""

from __future__ import annotations

import os
from typing import Any
from zoneinfo import ZoneInfo


def display_timezone() -> tuple[ZoneInfo, str]:
    """Clock/report timezone (IANA). Defaults to Colombia (UTC−5, no DST).

    ``DASHBOARD_TIMEZONE`` overrides ``REPORT_TIMEZONE`` when set.
    """
    raw = (
        os.environ.get("DASHBOARD_TIMEZONE")
        or os.environ.get("REPORT_TIMEZONE")
        or "America/Bogota"
    ).strip()
    name = raw or "America/Bogota"
    try:
        return ZoneInfo(name), name
    except Exception:
        return ZoneInfo("America/Bogota"), "America/Bogota"


def dashboard_rsi_label() -> str:
    """Human label for RSI row (e.g. RSI 14 · 15m)."""
    tf = dashboard_rsi_timeframe()
    return f"RSI (14, velas {tf})"


def dashboard_rsi_timeframe() -> str:
    """Kline timeframe label used for dashboard RSI only (single homogeneous series)."""
    raw = os.environ.get("DASHBOARD_RSI_TF", "15m").strip().lower()
    return raw or "15m"


def pct_change_24h_vs_h1(h1_closes: list[float], mark_price: float) -> float | None:
    """Approximate 24h % vs H1 close ~24 bars ago (needs >=24 hourly closes)."""
    if len(h1_closes) < 24 or mark_price <= 0:
        return None
    ref = h1_closes[-24]
    if ref <= 0:
        return None
    return (mark_price - ref) / ref * 100.0


def mt5_dashboard_mark(
    state: dict[str, Any], sym: str, candle_close: float | None
) -> float | None:
    """Pick display/mark price: live MT5 quote if present, else candle-based fallback."""
    raw = state.get("mt5_last_quote")
    if not isinstance(raw, dict):
        return candle_close
    q = raw.get(sym)
    if not isinstance(q, dict):
        return candle_close
    bid = float(q.get("bid") or 0.0)
    ask = float(q.get("ask") or 0.0)
    mid = float(q.get("mid") or 0.0)
    mode = os.environ.get("MT5_DASHBOARD_PRICE", "mid").strip().lower()
    if mode == "bid" and bid > 0.0:
        return bid
    if mode == "ask" and ask > 0.0:
        return ask
    if mid > 0.0:
        return mid
    if bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    return candle_close


def compute_rsi(prices: list[float], period: int = 14) -> float | None:
    """Compute RSI for the last *period* bars using simple averages.

    Returns a float in [0, 100] or ``None`` when there are insufficient bars.
    Used only by the dashboard for display; the authoritative RSI used in
    signal generation comes from :class:`~strategy.feature_engineer.FeatureEngineer`.
    """
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    avg_gain = sum(d for d in recent if d > 0) / period
    avg_loss = sum(-d for d in recent if d < 0) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
