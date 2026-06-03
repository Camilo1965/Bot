"""data/orderbook_features.py — Live L2 order book imbalance feature.

Computes bid/ask volume imbalance from top N levels of the order book.
Imbalance > 0.6 → buy pressure; < 0.4 → sell pressure.

This is a LIVE-ONLY feature (no historical L2 snapshots available).
In backtest context, returns 0.5 (neutral) always.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("orderbook_features")

_OB_LEVELS = 10  # top N bid/ask levels to sum


def fetch_ob_imbalance(symbol: str, exchange: Any | None = None) -> float:
    """Fetch live order book and compute bid/(bid+ask) volume ratio.

    Returns float in [0, 1]: >0.6 = buy pressure, <0.4 = sell pressure, 0.5 = neutral.
    Falls back to 0.5 if fetch fails.
    """
    if exchange is None:
        try:
            import ccxt
            exchange = ccxt.binance()
        except Exception:
            return 0.5

    try:
        ob = exchange.fetch_order_book(symbol, limit=_OB_LEVELS)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return 0.5
        bid_vol = sum(float(row[1]) for row in bids[:_OB_LEVELS])
        ask_vol = sum(float(row[1]) for row in asks[:_OB_LEVELS])
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.5
        return float(bid_vol / total)
    except Exception as exc:
        logger.debug("[%s] OB fetch failed: %s", symbol, exc)
        return 0.5


def ob_imbalance_to_signal(imbalance: float, threshold: float = 0.15) -> int:
    """Convert imbalance to directional signal: +1 buy, -1 sell, 0 neutral.

    threshold: how far from 0.5 to consider significant.
    """
    if imbalance > 0.5 + threshold:
        return 1
    if imbalance < 0.5 - threshold:
        return -1
    return 0


# ── Feature column name ───────────────────────────────────────────────────────
OB_FEATURE_COLS: list[str] = ["ob_imbalance"]
