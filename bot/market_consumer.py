"""Consumes normalized market messages from the asyncio queue."""

from __future__ import annotations

import asyncio
import collections
import logging
from datetime import datetime, timezone
from typing import Any

from database.db_manager import db
from execution.paper_executor import PaperExecutor

logger = logging.getLogger("clawdbot.market")

async def market_consumer(
    queue: asyncio.Queue[dict[str, Any]],
    state: dict[str, Any],
    paper_executor: PaperExecutor,
) -> None:
    _count = 0
    while True:
        msg = await queue.get()
        _count += 1
        sym = msg["symbol"]
        price = msg["price"]

        # Update last price in state
        state["prices"][sym].append(price)
        state["last_market_message_at"] = datetime.now(tz=timezone.utc)

        # [MT5] Last quote update for dashboard/PnL
        if "bid" in msg and "ask" in msg:
            state["mt5_last_quote"][sym] = {
                "bid": msg["bid"],
                "ask": msg["ask"],
                "mid": price,
            }

        # [ELITE] Deduplicate and append to OHLCV buffers
        kline_ts = msg.get("timestamp")
        if kline_ts and kline_ts != state["last_kline_ts"].get(sym):
            state["last_kline_ts"][sym] = kline_ts
            state["highs"][sym].append(msg["high"])
            state["lows"][sym].append(msg["low"])
            vol_deques = state.setdefault("volumes", {})
            if sym not in vol_deques:
                vol_deques[sym] = collections.deque(maxlen=1000)
            vol_deques[sym].append(float(msg.get("volume", 0.0) or 0.0))
            
            if "timeframe" in msg and "close" in msg:
                tf = msg["timeframe"]
                if tf in ("1h", "4h"):
                    state["htf_closes"][sym][tf].append(msg["close"])
                    state["htf_opens"][sym][tf].append(msg["open"])

            # Persist to DB
            try:
                await db.insert_market_data(
                    bucket=kline_ts,
                    symbol=sym,
                    open=msg.get("open", price),
                    high=msg.get("high", price),
                    low=msg.get("low", price),
                    close=msg.get("close", price),
                    volume=msg.get("volume", 0.0),
                )
            except Exception as exc:
                logger.warning("DB insert failed: %s", exc)

        # Check and move trailing stops
        if sym in paper_executor.open_positions:
            await paper_executor.check_and_close(sym, price)

        queue.task_done()
