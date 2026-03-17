"""
data_ingestion.websocket_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connects to Binance via ccxt.pro WebSockets and feeds live L2 order-book
snapshots and trade events for BTC/USDT into an asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
RECONNECT_DELAY = 5  # seconds between reconnection attempts


class BinanceWebSocketClient:
    """Streams L2 order-book and trades for *symbol* into *queue*."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        symbol: str = SYMBOL,
    ) -> None:
        self.queue = queue
        self.symbol = symbol
        self._exchange: ccxtpro.binance | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start both stream coroutines and keep them alive on errors."""
        while True:
            self._exchange = ccxtpro.binance({"enableRateLimit": True})
            tasks: list[asyncio.Task[None]] = []
            try:
                logger.info(
                    "Connecting to Binance WebSocket streams for %s", self.symbol
                )
                tasks = [
                    asyncio.create_task(self._watch_order_book()),
                    asyncio.create_task(self._watch_trades()),
                ]
                await asyncio.gather(*tasks)
            except (ccxtpro.NetworkError, ccxtpro.ExchangeError) as exc:
                logger.warning(
                    "WebSocket error (%s): %s – reconnecting in %ds",
                    type(exc).__name__,
                    exc,
                    RECONNECT_DELAY,
                )
            except asyncio.CancelledError:
                logger.info("WebSocket client cancelled – shutting down.")
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Unexpected error (%s): %s – reconnecting in %ds",
                    type(exc).__name__,
                    exc,
                    RECONNECT_DELAY,
                )
            finally:
                # Cancel any still-running sibling tasks before closing the
                # exchange so the stream coroutines can't dereference a
                # closed/None exchange object.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                await self._close_exchange()

            await asyncio.sleep(RECONNECT_DELAY)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _watch_order_book(self) -> None:
        """Continuously receive L2 order-book snapshots and push to queue."""
        while self._exchange is not None:
            order_book = await self._exchange.watch_order_book(self.symbol)
            message: dict[str, Any] = {
                "type": "order_book",
                "symbol": self.symbol,
                "bids": order_book["bids"][:5],
                "asks": order_book["asks"][:5],
                "timestamp": order_book.get("timestamp"),
            }
            await self.queue.put(message)

    async def _watch_trades(self) -> None:
        """Continuously receive trade events and push to queue."""
        while self._exchange is not None:
            trades = await self._exchange.watch_trades(self.symbol)
            for trade in trades:
                message: dict[str, Any] = {
                    "type": "trade",
                    "symbol": self.symbol,
                    "id": trade.get("id"),
                    "price": trade.get("price"),
                    "amount": trade.get("amount"),
                    "side": trade.get("side"),
                    "timestamp": trade.get("timestamp"),
                }
                await self.queue.put(message)

    async def _close_exchange(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:  # noqa: BLE001
                pass
            self._exchange = None


async def message_consumer(queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Print every message arriving on *queue* to verify the connection."""
    while True:
        message = await queue.get()
        logger.info("Received message: %s", message)
        queue.task_done()


async def run_websocket_client() -> None:
    """Entry point: create queue, start producer and consumer coroutines."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    client = BinanceWebSocketClient(queue=queue)
    await asyncio.gather(
        client.run(),
        message_consumer(queue),
    )
