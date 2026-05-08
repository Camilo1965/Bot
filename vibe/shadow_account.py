"""
vibe.shadow_account
~~~~~~~~~~~~~~~~~~~
Extracts strategy rules from ClawdBot's trade journal,
backtests a "shadow" portfolio that follows ideal rules,
and generates a comparison report showing how much PnL
is left on the table by early exits, missed signals, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.shadow")

JOURNAL_PATH = Path("logs") / "trade_journal.csv"


def _extract_shadow_id(result: dict[str, Any] | None) -> str | None:
    """Parse shadow_id from an extract_shadow_strategy MCP result."""
    if not result:
        return None
    try:
        if isinstance(result, dict) and "content" in result:
            for item in result["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    try:
                        data = json.loads(text)
                        return data.get("shadow_id")
                    except (json.JSONDecodeError, AttributeError):
                        idx = text.find("shadow_id")
                        if idx >= 0:
                            for segment in text[idx:].split():
                                if segment.startswith("shd_") or segment.startswith("shadow_"):
                                    clean = segment.strip('",`\'}:')
                                    return clean
                            rest = text[idx:]
                            for sep in [":", "="]:
                                if sep in rest:
                                    val = rest.split(sep, 1)[1].split()[0].strip('",`\'}:')
                                    if val:
                                        return val
        if isinstance(result, dict):
            return result.get("shadow_id")
    except Exception as exc:
        logger.warning("[VIBE] Could not parse shadow_id: %s", exc)
    return None


async def extract_and_backtest_shadow(
    client: VibeMCPClient,
    journal_path: Path | None = None,
) -> dict[str, Any] | None:
    """Extract shadow strategy from trade journal and run backtest.

    1. Calls extract_shadow_strategy on the journal CSV
    2. If a shadow_id is returned, runs shadow backtest
    3. Returns comparison report data
    """
    path = journal_path or JOURNAL_PATH
    if not path.is_file():
        logger.warning("[VIBE] Trade journal not found: %s", path)
        return None

    shadow_result = await client.extract_shadow_strategy(str(path.resolve()))
    if not shadow_result:
        logger.warning("[VIBE] Shadow strategy extraction returned no data.")
        return None

    shadow_id = _extract_shadow_id(shadow_result)
    if not shadow_id:
        logger.warning("[VIBE] Could not extract shadow_id from result.")
        return shadow_result

    backtest_result = await client.run_shadow_backtest(shadow_id=shadow_id)
    if backtest_result:
        return backtest_result

    return shadow_result