"""Normalize trading symbols for CSV journal and analytics (internal *BASE/USDT* form)."""

from __future__ import annotations

# Mirror broker suffixes from mt5_executor.SYMBOL_MAP (avoid import cycle).
_BROKER_TO_INTERNAL: dict[str, str] = {
    "BTCUSD-T": "BTC/USDT",
    "ETHUSD-T": "ETH/USDT",
    "SOLUSD-T": "SOL/USDT",
    "BNBUSD-T": "BNB/USDT",
    "LINKUSD-T": "LINK/USDT",
    "INJUSD-T": "INJ/USDT",
    "FETUSD-T": "FET/USDT",
    "RENDERUSD-T": "RENDER/USDT",
    "DOGEUSD-T": "DOGE/USDT",
    "PEPEUSD-T": "PEPE/USDT",
    "XAUUSD-T": "PAXG/USDT",
}


def journal_symbol(symbol: str) -> str:
    """Return canonical internal symbol for logging (e.g. ``ETH/USDT``)."""
    s = symbol.strip()
    if "/" in s:
        return s
    return _BROKER_TO_INTERNAL.get(s, s)
