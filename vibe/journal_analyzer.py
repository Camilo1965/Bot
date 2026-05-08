"""
vibe.journal_analyzer
~~~~~~~~~~~~~~~~~~~~~
Analyzes ClawdBot's trade journal CSV using Vibe-Trading's
analyze_trade_journal MCP tool.

Detects behavioral biases: disposition effect, overtrading,
chasing momentum, anchoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.journal")

JOURNAL_PATH = Path("logs") / "trade_journal.csv"


async def analyze_journal(
    client: VibeMCPClient,
    journal_path: Path | None = None,
) -> dict[str, Any] | None:
    """Analyze the trade journal CSV for behavioral biases.

    Returns a dict with bias diagnostics, or None if Vibe is unavailable.
    """
    path = journal_path or JOURNAL_PATH
    if not path.is_file():
        logger.warning("[VIBE] Trade journal not found: %s", path)
        return None
    result = await client.analyze_trade_journal(str(path.resolve()))
    if result:
        logger.info("[VIBE] Journal analysis complete.")
    return result