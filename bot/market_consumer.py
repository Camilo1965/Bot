"""Consumes normalized market messages from the asyncio queue."""

from __future__ import annotations

import asyncio
import collections
import logging
from datetime import datetime, timezone
from typing import Any

from bot.dashboard_helpers import dashboard_rsi_timeframe
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
        # MT5 feed emits trade (price), order_book (bids/asks), kline (OHLC) - not all have "price".
        mtype = msg.get("type", "trade")
        if mtype == "kline":
            price = float(msg.get("close") or msg.get("open") or 0.0)
        elif mtype == "order_book":
            bids = msg.get("bids") or []
            asks = msg.get("asks") or []
            if bids and asks:
                price = (float(bids[0][0]) + float(asks[0][0])) / 2.0
            else:
                queue.task_done()
                continue
        elif "price" in msg:
            price = float(msg["price"])
        else:
            queue.task_done()
            continue

        if price <= 0.0:
            queue.task_done()
            continue

        # Update last price in state
        state["last_price"] = price  # Store latest price for real-time checks
        state["last_market_message_at"] = datetime.now(tz=timezone.utc)

        # Buffer update logic: update technical indicator buffers based on configured timeframe
        from strategy.ml_predictor import get_symbol_config
        cfg = get_symbol_config(sym)
        target_tf = cfg.get("timeframe", "15m")

        if mtype == "kline" and msg.get("timeframe") == target_tf:
            state["prices"][sym].append(price)
            state["highs"][sym].append(float(msg.get("high", price)))
            state["lows"][sym].append(float(msg.get("low", price)))
            
            vol_deques = state.setdefault("volumes", {})
            if sym not in vol_deques:
                vol_deques[sym] = collections.deque(maxlen=1000)
            vol_deques[sym].append(float(msg.get("volume", 0.0) or 0.0))

        _rsi_tf = dashboard_rsi_timeframe()
        if (
            mtype == "kline"
            and msg.get("timeframe") == _rsi_tf
            and float(msg.get("close") or 0.0) > 0.0
        ):
            rsi_buf = state.get("dashboard_rsi_closes", {}).get(sym)
            if rsi_buf is not None:
                rsi_buf.append(float(msg["close"]))

        # [MT5] Last quote update for dashboard/PnL
        if "bid" in msg and "ask" in msg:
            state["mt5_last_quote"][sym] = {
                "bid": msg["bid"],
                "ask": msg["ask"],
                "mid": price,
            }

        # New closed candle: only kline messages carry OHLC; ticks must not write bogus highs/lows.
        kline_ts = msg.get("timestamp")
        if (
            mtype == "kline"
            and kline_ts
            and kline_ts != state["last_kline_ts"].get(sym)
        ):
            state["last_kline_ts"][sym] = kline_ts
            
            if "timeframe" in msg and "close" in msg:
                tf = msg["timeframe"]
                if tf in ("1h", "4h"):
                    state["htf_closes"][sym][tf].append(msg["close"])
                    state["htf_opens"][sym][tf].append(msg["open"])

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
            # Push ratcheted SL/TP to MT5 for live positions (trailing-stop sync)
            if sym in paper_executor.open_positions:
                sync_fn = getattr(paper_executor, "_sync_exchange_stops", None)
                if sync_fn is not None:
                    try:
                        await sync_fn(sym, paper_executor.open_positions[sym], price, None)
                    except Exception:
                        pass

        queue.task_done()
