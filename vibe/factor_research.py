"""
vibe.factor_research
~~~~~~~~~~~~~~~~~~~~
Validates which of ClawdBot's quant features actually predict returns
using Vibe-Trading's factor_analysis MCP tool.

Produces IC/IR (Information Coefficient / Information Ratio) scores
that can be used to weight or prune features in the XGBoost model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.factor")

_CRYPTO_SOURCE = "ccxt"

_CRYPTO_FACTORS = [
    "close",
    "volume",
    "high",
    "low",
]


async def analyze_factors(
    client: VibeMCPClient,
    symbols: list[str],
    factor_name: str = "close",
    days_back: int = 90,
    source: str = _CRYPTO_SOURCE,
) -> dict[str, Any] | None:
    """Run IC/IR factor analysis for symbols.

    Returns factor analysis results or None.
    """
    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        result = await client.factor_analysis(
            codes=symbols,
            factor_name=factor_name,
            start_date=start_date,
            end_date=end_date,
            source=source,
        )
        if result:
            return result
    except Exception as exc:
        logger.warning("[VIBE] Factor analysis failed for %s: %s", symbols, exc)
    return None