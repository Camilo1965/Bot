"""
vibe.swarm_loops
~~~~~~~~~~~~~~~~

Background swarm-intelligence loop for VIBE-Trading (Fase 5).

Runs the ``crypto_trading_desk`` swarm preset periodically and stores the
parsed recommendation in ``shared_state["vibe_swarm_recommendation"]``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.swarm")

_SWARM_BACKOFF_BASE = 2.0
_SWARM_BACKOFF_MAX = 300.0
_SWARM_TIMEOUT = 90

# Keywords used to heuristically map swarm text output to a recommendation.
_STRONG_BUY_KW = {"strong buy", "aggressive long", "conviction buy", "accumulate"}
_BUY_KW = {"buy", "long", "bullish", "upside", "enter long"}
_EXIT_KW = {"exit", "close", "sell", "flat", "neutralize", "take profit"}
_REDUCE_KW = {"reduce", "trim", "scale out", "lighten", "de-risk"}


def _extract_text(data: Any) -> str:
    """Extract plain text from an MCP response dict."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data.lower()
    if isinstance(data, dict):
        parts: list[str] = []
        for item in data.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts).lower()
    return str(data).lower()


def parse_swarm_recommendation(result: dict | None) -> str:
    """Map a raw swarm result to one of five recommendation levels.

    Returns one of: ``STRONG_BUY``, ``BUY``, ``NEUTRAL``, ``REDUCE``, ``EXIT``.
    """
    if result is None:
        return "NEUTRAL"

    text = _extract_text(result)
    if not text:
        return "NEUTRAL"

    # Score each category by keyword count
    strong_buy = sum(1 for kw in _STRONG_BUY_KW if kw in text)
    buy = sum(1 for kw in _BUY_KW if kw in text)
    exit_s = sum(1 for kw in _EXIT_KW if kw in text)
    reduce = sum(1 for kw in _REDUCE_KW if kw in text)

    scores = {
        "STRONG_BUY": strong_buy,
        "BUY": buy,
        "EXIT": exit_s,
        "REDUCE": reduce,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "NEUTRAL"
    return best


async def crypto_desk_swarm_loop(
    client: VibeMCPClient,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = 3600,
    initial_delay_s: float = 120.0,
) -> None:
    """Hourly: run ``crypto_trading_desk`` swarm for each symbol and store recommendation."""
    if not client.available:
        logger.info("[VIBE SWARM] Client not available — swarm loop disabled.")
        return

    if initial_delay_s > 0:
        await asyncio.sleep(initial_delay_s)
    failures = 0

    while True:
        for symbol in watchlist:
            try:
                result = await client.run_swarm(
                    preset="crypto_trading_desk",
                    variables={"asset": symbol, "timeframe": "1h"},
                    timeout=_SWARM_TIMEOUT,
                )
                if result is not None:
                    rec = parse_swarm_recommendation(result)
                    shared_state.setdefault("vibe_swarm_recommendation", {})[symbol] = rec
                    logger.info("[VIBE SWARM] %s recommendation: %s", symbol, rec)
                    failures = 0
                else:
                    failures += 1
                    logger.warning("[VIBE SWARM] %s returned no result.", symbol)
            except asyncio.TimeoutError:
                failures += 1
                logger.warning("[VIBE SWARM] %s timed out after %ds.", symbol, _SWARM_TIMEOUT)
            except Exception as exc:
                failures += 1
                logger.warning("[VIBE SWARM] %s failed: %s", symbol, exc)

            if failures > 0:
                backoff = min(_SWARM_BACKOFF_BASE * (2 ** (failures - 1)), _SWARM_BACKOFF_MAX)
                logger.info("[VIBE SWARM] Backing off %.0fs after %d failures.", backoff, failures)
                await asyncio.sleep(backoff)

        await asyncio.sleep(interval_s)
