"""
vibe.backtest
~~~~~~~~~~~~~
Runs Vibe-Trading backtest engine using historical data
from ClawdBot's own TimescaleDB or Vibe-Trading's data sources.

Two modes:
1. Prompt-based: "Backtest BTC-USDT RSI strategy last 30 days"
2. Data-based: Export ClawdBot's OHLCV data to CSV, then backtest
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.backtest")

_EXPORT_DIR = Path("logs") / "vibe_exports"


async def export_db_data_to_csv(symbol: str) -> Path | None:
    """Export market_data from TimescaleDB to CSV for Vibe-Trading consumption.

    Reads the last N rows from the market_data hypertable for the given symbol
    and writes to logs/vibe_exports/<symbol>_<timestamp>.csv
    """
    from database.db_manager import db

    rows = await db.fetch_market_data_ohlcv(symbol=symbol, limit=5000)
    if not rows:
        logger.warning("[VIBE] No DB data for %s — cannot export.", symbol)
        return None

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _EXPORT_DIR / f"{symbol.replace('/', '_')}_{ts}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
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

    logger.info("[VIBE] Exported %d rows for %s → %s", len(rows), symbol, path)
    return path


async def run_backtest(
    client: VibeMCPClient,
    prompt: str,
) -> dict[str, Any] | None:
    """Run a backtest using Vibe-Trading's backtest engine.

    The prompt should be a natural language description, e.g.:
    "Backtest BTC/USDT MACD crossover strategy, last 30 days, 15m timeframe"
    """
    return await client.backtest(prompt)