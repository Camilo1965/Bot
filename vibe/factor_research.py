"""
vibe.factor_research
~~~~~~~~~~~~~~~~~~~~
Validates which of ClawdBot's 12 quant features actually predict returns
using Vibe-Trading's factor_analysis MCP tool.

Produces IC/IR (Information Coefficient / Information Ratio) scores
that can be used to weight or prune features in the XGBoost model.
"""

from __future__ import annotations

import logging
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.factor")


async def analyze_factors(
    client: VibeMCPClient,
    symbol: str,
) -> dict[str, Any] | None:
    """Run IC/IR factor analysis for a symbol's quant features.

    Returns factor analysis results or None.
    """
    prompt = (
        f"Analyze the predictive power of these factors for {symbol}: "
        f"RSI(14), MACD, ATR(14), Bollinger Bands %B, Bollinger Bandwidth, "
        f"Volume Delta, Log Return 1-period, Log Return 5-period, "
        f"Close vs SMA200(1h), Relative Volume. "
        f"Show ic, ir, and turnover for each factor over the last 90 days."
    )
    return await client.factor_analysis(prompt)