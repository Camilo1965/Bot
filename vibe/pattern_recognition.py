"""
vibe.pattern_recognition
~~~~~~~~~~~~~~~~~~~~~~~~
Detects technical patterns (candlestick, harmonic, SMC, Elliott Wave)
using Vibe-Trading's pattern MCP tool.

Runs on-demand or periodically, stores detected patterns in shared_state
for the signal emitter and dashboard to consume.
"""

from __future__ import annotations

import logging
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.pattern")


async def detect_patterns(
    client: VibeMCPClient,
    symbol: str,
    timeframe: str = "15m",
) -> dict[str, Any] | None:
    """Detect technical patterns for a symbol using Vibe-Trading's pattern tool.

    Returns a dict of detected patterns or None.
    """
    prompt = (
        f"Detect technical patterns for {symbol} on {timeframe} timeframe. "
        f"Include candlestick patterns, support/resistance levels, and trend structure."
    )
    return await client.pattern_recognition(symbol, prompt)