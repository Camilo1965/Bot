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

logger = logging.getLogger("clawdbot.vibe.journal")

JOURNAL_PATH = Path("logs") / "trade_journal.csv"
_MIN_UNIQUE_TRADES = 5


async def analyze_journal(
    client: Any,
    journal_path: Path | None = None,
) -> dict[str, Any] | None:
    """Analyze the trade journal CSV for behavioral biases.

    Returns a dict with bias diagnostics, or None if Vibe is unavailable.
    """
    path = journal_path or JOURNAL_PATH
    if not path.is_file():
        logger.debug("[VIBE] Trade journal not found (will analyze once trades exist): %s", path)
        return None

    # Pre-process: deduplicate to prevent MCP crash on duplicate trade IDs
    # (stress-test residue writes identical rows with trade_id='t1').
    try:
        import pandas as pd
        df = pd.read_csv(str(path), header=None, dtype=str)
        df = df.drop_duplicates()
        if len(df) < _MIN_UNIQUE_TRADES:
            logger.debug("[VIBE] Journal has only %d unique rows — skipping analysis.", len(df))
            return None
        clean_path = path.parent / "trade_journal_clean.csv"
        df.to_csv(str(clean_path), index=False, header=False)
        target = clean_path
    except Exception as exc:
        logger.warning("[VIBE] Journal pre-process failed: %s — using original file.", exc)
        target = path

    result = await client.analyze_trade_journal(target.resolve().as_posix())
    if result:
        logger.info("[VIBE] Journal analysis complete.")
    return result