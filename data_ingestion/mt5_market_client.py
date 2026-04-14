"""
data_ingestion.mt5_market_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poll-based MT5 market-data client that emits trade/order-book/kline-like
messages compatible with the bot's existing market queue consumer.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from execution.mt5_executor import MT5Executor, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_M15

logger = logging.getLogger(__name__)

_TF_MAP: dict[str, int] = {
    "15m": TIMEFRAME_M15,
    "1h": TIMEFRAME_H1,
    "4h": TIMEFRAME_H4,
}


class MT5MarketDataClient:
    """Poll MT5 ticks/candles and publish normalized queue messages."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        executor: MT5Executor,
        watchlist: list[str],
        tick_interval_s: float = 1.0,
        kline_interval_s: float = 10.0,
    ) -> None:
        self.queue = queue
        self.executor = executor
        self.watchlist = watchlist
        self.tick_interval_s = tick_interval_s
        self.kline_interval_s = kline_interval_s
        self._last_kline_ts: dict[str, dict[str, int]] = {
            sym: {"15m": -1, "1h": -1, "4h": -1} for sym in watchlist
        }

    async def run(self) -> None:
        """Run tick and kline pollers concurrently."""
        await asyncio.gather(
            self._tick_loop(),
            self._kline_loop(),
        )

    async def _tick_loop(self) -> None:
        while True:
            for sym in self.watchlist:
                try:
                    tick = await self.executor.fetch_tick(sym)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[MT5 FEED] tick fetch failed for %s: %s", sym, exc)
                    continue
                if not tick:
                    continue
                bid = float(tick.get("bid") or 0.0)
                ask = float(tick.get("ask") or 0.0)
                last = float(tick.get("last") or ((bid + ask) / 2.0 if bid and ask else 0.0))
                if last <= 0.0 and bid > 0.0 and ask > 0.0:
                    last = (bid + ask) / 2.0
                if last <= 0.0:
                    continue
                ts = tick.get("time")
                ts_ms = (
                    int(ts.timestamp() * 1000)
                    if isinstance(ts, datetime)
                    else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
                )

                await self.queue.put(
                    {
                        "type": "trade",
                        "symbol": sym,
                        "id": f"mt5-{sym}-{ts_ms}",
                        "price": last,
                        "amount": 0.0,
                        "side": None,
                        "timestamp": ts_ms,
                    }
                )

                # MT5 does not expose level-2 book by default; emit synthetic top-of-book.
                if bid > 0.0 and ask > 0.0:
                    await self.queue.put(
                        {
                            "type": "order_book",
                            "symbol": sym,
                            "bids": [[bid, 1.0]],
                            "asks": [[ask, 1.0]],
                            "timestamp": ts_ms,
                        }
                    )
            await asyncio.sleep(self.tick_interval_s)

    async def _kline_loop(self) -> None:
        while True:
            for sym in self.watchlist:
                for tf_name, tf_value in _TF_MAP.items():
                    try:
                        df = await self.executor.fetch_candles(
                            symbol=sym,
                            timeframe=tf_value,
                            count=2,
                            start_pos=0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("[MT5 FEED] kline fetch failed for %s/%s: %s", sym, tf_name, exc)
                        continue
                    if df is None or df.empty:
                        continue
                    row = df.iloc[-1]
                    ts = row.get("time")
                    if not isinstance(ts, datetime):
                        continue
                    ts_ms = int(ts.timestamp() * 1000)
                    if ts_ms == self._last_kline_ts[sym][tf_name]:
                        continue
                    self._last_kline_ts[sym][tf_name] = ts_ms

                    await self.queue.put(
                        {
                            "type": "kline",
                            "symbol": sym,
                            "timeframe": tf_name,
                            "timestamp": ts_ms,
                            "open": float(row.get("open", 0.0)),
                            "high": float(row.get("high", 0.0)),
                            "low": float(row.get("low", 0.0)),
                            "close": float(row.get("close", 0.0)),
                            "volume": float(row.get("tick_volume", 0.0)),
                        }
                    )
            await asyncio.sleep(self.kline_interval_s)
