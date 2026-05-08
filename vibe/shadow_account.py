"""
vibe.shadow_account
~~~~~~~~~~~~~~~~~~~
Extracts strategy rules from ClawdBot's trade journal,
backtests a "shadow" portfolio that follows ideal rules,
and generates a comparison report showing how much PnL
is left on the table by early exits, missed signals, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.shadow")

JOURNAL_PATH = Path("logs") / "trade_journal.csv"


async def extract_and_backtest_shadow(
    client: VibeMCPClient,
    journal_path: Path | None = None,
) -> dict[str, Any] | None:
    """Extract shadow strategy from trade journal and run backtest.

    1. Calls extract_shadow_strategy on the journal CSV
    2. If a strategy is extracted, runs shadow backtest
    3. Returns comparison report data
    """
    path = journal_path or JOURNAL_PATH
    if not path.is_file():
        logger.warning("[VIBE] Trade journal not found: %s", path)
        return None

    shadow = await client.extract_shadow_strategy(str(path.resolve()))
    if not shadow:
        logger.warning("[VIBE] Shadow strategy extraction returned no data.")
        return None

    prompt = (
        "Backtest the shadow strategy extracted from the trade journal. "
        "Compare actual PnL vs ideal PnL. Highlight rule violations, "
        "early exits, and missed signals."
    )
    result = await client.run_shadow_backtest(prompt)
    return result