"""
data_ingestion.websocket_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connects to Binance via ccxt.pro WebSockets and feeds live L2 order-book
snapshots and trade events for all symbols in the watchlist into an
asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)

SYMBOL = "BTC/USDT"
RECONNECT_DELAY = 5  # seconds between reconnection attempts
OHLCV_TIMEFRAME = "15m"  # candle timeframe used for ML feature calculation


class BinanceWebSocketClient:
    """Streams L2 order-book and trades for every symbol in *watchlist* into *queue*."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        watchlist: list[str] | None = None,
        symbol: str = SYMBOL,
    ) -> None:
        self.queue = queue
        # *watchlist* takes priority; fall back to the legacy *symbol* parameter
        self.watchlist: list[str] = watchlist if watchlist is not None else [symbol]
        self._exchange: ccxtpro.binance | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start stream coroutines for every watchlist symbol and keep them alive on errors."""
        while True:
            self._exchange = ccxtpro.binance({"enableRateLimit": True})
            tasks: list[asyncio.Task[None]] = []
            try:
                logger.info(
                    "Connecting to Binance WebSocket streams for %s", self.watchlist
                )
                for sym in self.watchlist:
                    tasks.append(asyncio.create_task(self._watch_order_book(sym)))
                    tasks.append(asyncio.create_task(self._watch_trades(sym)))
                    tasks.append(asyncio.create_task(self._watch_ohlcv(sym)))
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

    async def _watch_order_book(self, symbol: str) -> None:
        """Continuously receive L2 order-book snapshots for *symbol* and push to queue.

        [ELITE] Calculates the Order Book Imbalance (OBI) ratio from the top-5
        levels: ``obi = (bid_volume - ask_volume) / (bid_volume + ask_volume)``.
        A positive value signals more buying pressure; negative more selling.
        """
        while self._exchange is not None:
            order_book = await self._exchange.watch_order_book(symbol)
            bids = order_book["bids"][:5]
            asks = order_book["asks"][:5]

            # [ELITE] Order Book Imbalance calculation
            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            total_vol = bid_vol + ask_vol
            obi_ratio = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

            message: dict[str, Any] = {
                "type": "order_book",
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "timestamp": order_book.get("timestamp"),
                "obi": obi_ratio,
            }
            await self.queue.put(message)

    async def _watch_trades(self, symbol: str) -> None:
        """Continuously receive trade events for *symbol* and push to queue."""
        while self._exchange is not None:
            trades = await self._exchange.watch_trades(symbol)
            for trade in trades:
                message: dict[str, Any] = {
                    "type": "trade",
                    "symbol": symbol,
                    "id": trade.get("id"),
                    "price": trade.get("price"),
                    "amount": trade.get("amount"),
                    "side": trade.get("side"),
                    "timestamp": trade.get("timestamp"),
                }
                await self.queue.put(message)

    async def _watch_ohlcv(self, symbol: str, timeframe: str = OHLCV_TIMEFRAME) -> None:
        """Watch 15-minute OHLCV klines for *symbol* and push candle messages to queue.

        Each message contains the most recent candle's close price and timestamp.
        Downstream consumers should deduplicate by timestamp so only one price
        entry is added per completed candle.
        """
        while self._exchange is not None:
            ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)
            if ohlcv:
                candle = ohlcv[-1]  # [timestamp, open, high, low, close, volume]
                message: dict[str, Any] = {
                    "type": "kline",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
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
