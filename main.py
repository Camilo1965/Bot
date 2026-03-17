"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import json
import sys
from datetime import datetime, timezone

from data_ingestion.websocket_client import run_websocket_client


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger with a JSON formatter and return it."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger("clawdbot")


async def main() -> None:
    logger = setup_logging()
    logger.info("ClawdBot starting up")

    await run_websocket_client()

    logger.info("ClawdBot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
