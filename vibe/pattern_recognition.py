"""
vibe.pattern_recognition
~~~~~~~~~~~~~~~~~~~~~~~~
Detects technical patterns (candlestick, harmonic, SMC, Elliott Wave)
using Vibe-Trading's pattern_recognition MCP tool.

Exports OHLCV data from TimescaleDB to a run directory, then calls
the MCP tool with the run_dir path.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.pattern")

_RUN_DIR_BASE = Path("logs") / "vibe_runs"


async def _export_ohlcv_to_run_dir(symbol: str, run_dir: Path) -> Path | None:
    """Export OHLCV data from TimescaleDB to run_dir/artifacts/ohlcv_<symbol>.csv."""
    from database.db_manager import db

    rows = await db.fetch_market_data_ohlcv(symbol=symbol, limit=5000)
    if not rows:
        logger.warning("[VIBE] No DB data for %s — cannot export.", symbol)
        return None

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    safe_name = symbol.replace("/", "_")
    csv_path = artifacts_dir / f"ohlcv_{safe_name}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in rows:
            writer.writerow([
                row["timestamp"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            ])

    logger.info("[VIBE] Exported %d rows for %s → %s", len(rows), symbol, csv_path)
    return csv_path


async def detect_patterns(
    client: VibeMCPClient,
    symbol: str,
    timeframe: str = "15m",
) -> dict[str, Any] | None:
    """Detect technical patterns for a symbol using Vibe-Trading's pattern tool.

    Exports OHLCV data from the database to a run directory, then calls
    the pattern_recognition MCP tool with the run_dir path.

    Returns a dict of detected patterns or None.
    """
    _RUN_DIR_BASE.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = symbol.replace("/", "_")
    run_dir = _RUN_DIR_BASE / f"pattern_{safe_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = await _export_ohlcv_to_run_dir(symbol, run_dir)
    if not csv_path:
        return None

    return await client.pattern_recognition(run_dir=run_dir.resolve().as_posix())