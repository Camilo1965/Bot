"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs the Binance WebSocket client and the crypto-news collector concurrently;
each incoming BTC trade is logged together with the latest sentiment score.
"""

from __future__ import annotations

import asyncio
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any

from data_ingestion.news_client import run_news_client
from data_ingestion.websocket_client import BinanceWebSocketClient
from strategy.sentiment_analyzer import SentimentAnalyzer


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


async def sentiment_processor(
    news_queue: asyncio.Queue[list[str]],
    state: dict[str, Any],
    analyzer: SentimentAnalyzer,
) -> None:
    """Consume headline batches from *news_queue*, update shared *state*."""
    while True:
        headlines = await news_queue.get()
        score = analyzer.score_headlines(headlines)
        state["sentiment"] = score
        logger = logging.getLogger("clawdbot.sentiment")
        logger.info(
            "Sentiment updated – headlines=%d  score=%.4f",
            len(headlines),
            score,
        )
        news_queue.task_done()


async def market_consumer(
    market_queue: asyncio.Queue[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    """Log BTC trade prices alongside the latest sentiment score."""
    logger = logging.getLogger("clawdbot.market")
    while True:
        message = await market_queue.get()
        if message.get("type") == "trade":
            price = message.get("price")
            sentiment = state.get("sentiment", 0.0)
            logger.info(
                "BTC/USDT price=%.2f  sentiment_score=%.4f",
                price,
                sentiment,
            )
        else:
            logger.info("Received message: %s", message)
        market_queue.task_done()


async def main() -> None:
    logger = setup_logging()
    logger.info("ClawdBot starting up")

    market_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    news_queue: asyncio.Queue[list[str]] = asyncio.Queue()
    shared_state: dict[str, Any] = {"sentiment": 0.0}

    analyzer = SentimentAnalyzer()
    ws_client = BinanceWebSocketClient(queue=market_queue)

    await asyncio.gather(
        ws_client.run(),
        run_news_client(news_queue),
        sentiment_processor(news_queue, shared_state, analyzer),
        market_consumer(market_queue, shared_state),
    )

    logger.info("ClawdBot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
