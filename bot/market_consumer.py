"""Consumes normalized market messages from the asyncio queue."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from bot.constants import DEBUG_LOG_HINT

from database.db_manager import db

from execution.mt5_executor import MT5Executor, fetch_mt5_wallet_snapshot
from execution.paper_executor import PaperExecutor


async def market_consumer(
    market_queue: asyncio.Queue[dict[str, Any]],
    state: dict[str, Any],
    paper_executor: PaperExecutor,
) -> None:
    logger = logging.getLogger("clawdbot.market")
    _last_price_log: dict[str, float] = {}  # symbol → last log timestamp (monotonic)
    _last_check_close: dict[str, float] = {} # symbol → last check_and_close timestamp (monotonic)
    _last_wallet_snap: float = 0.0
    
    # Tick sampling interval: 100ms (10 Hz) is enough for stop-loss checks 
    # and prevents CPU/IPC bottleneck when MT5 sends micro-ticks.
    CHECK_CLOSE_INTERVAL_S = 0.1 

    while True:
        message = await market_queue.get()
        _now_mono = time.monotonic()
        state["last_market_message_at"] = datetime.now(tz=timezone.utc)
        symbol: str = message.get("symbol", "BTC/USDT")

        if message.get("type") == "trade":
            if isinstance(paper_executor, MT5Executor) and paper_executor._live:
                # Wallet snapshot is expensive (MT5 IPC), throttle to 1 second.
                if _now_mono - _last_wallet_snap >= 1.0:
                    _last_wallet_snap = _now_mono
                    snap = fetch_mt5_wallet_snapshot()
                    if snap:
                        state["mt5_wallet"] = snap
            
            price = message.get("price")
            b_raw, a_raw = message.get("bid"), message.get("ask")
            if b_raw is not None and a_raw is not None:
                b_f, a_f = float(b_raw), float(a_raw)
                if b_f > 0.0 and a_f > 0.0:
                    mid_q = float(message.get("mid") or (b_f + a_f) / 2.0)
                    ts_ms = message.get("timestamp")
                    state.setdefault("mt5_last_quote", {})[symbol] = {
                        "bid": b_f,
                        "ask": a_f,
                        "mid": mid_q,
                        "ts_ms": int(ts_ms) if isinstance(ts_ms, (int, float)) else 0,
                    }
            
            # Guard against None sentiment (startup race) to avoid TypeError.
            sentiment = state.get("sentiment") or 0.0

            # Demoted to DEBUG – throttled to once per 60 s per symbol.
            if _now_mono - _last_price_log.get(symbol, 0.0) >= 60.0:
                _last_price_log[symbol] = _now_mono
                prices_buf = state["prices"].get(symbol, [])
                logger.debug(
                    "%s price=%.2f  sentiment_score=%.4f | Buffer: %d/500",
                    symbol, price, sentiment, len(prices_buf),
                )

            if price is not None:
                # ── TICK SAMPLING ─────────────────────────────────────────────
                # Only check exit conditions at most once per CHECK_CLOSE_INTERVAL_S.
                # This drastically reduces MT5 IPC calls and DB overhead.
                if (_now_mono - _last_check_close.get(symbol, 0.0)) >= CHECK_CLOSE_INTERVAL_S:
                    _last_check_close[symbol] = _now_mono
                    try:
                        pnl = await paper_executor.check_and_close(
                            float(price),
                            symbol=symbol,
                            current_atr=state.get("atrs", {}).get(symbol),
                        )
                        if pnl is not None:
                            logger.info(
                                "💰 [CIERRE] %s | PnL: %.4f USDT | Tick Trade",
                                symbol, pnl
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("⚠️ [ALERTA] check_and_close failed: %s %s", exc, DEBUG_LOG_HINT)

        elif message.get("type") == "kline":
            # ... (timeframe logic remains same) ...
            timeframe: str = message.get("timeframe", "15m")
            close_price = message.get("close")
            open_price = message.get("open")
            high_price = message.get("high")
            low_price = message.get("low")
            candle_ts = message.get("timestamp")

            if timeframe == "15m":
                if close_price is not None and candle_ts is not None:
                    last_ts = state["last_kline_ts"].get(symbol)
                    if candle_ts != last_ts:
                        state["last_kline_ts"][symbol] = candle_ts
                        prices_dict: dict[str, deque[float]] = state["prices"]
                        if symbol in prices_dict:
                            prices_dict[symbol].append(float(close_price))
                        if high_price is not None and symbol in state.get("highs", {}):
                            state["highs"][symbol].append(float(high_price))
                        if low_price is not None and symbol in state.get("lows", {}):
                            state["lows"][symbol].append(float(low_price))
                        logger.debug(
                            "📊 [KLINE] %s – new 15m candle close=%.2f",
                            symbol, float(close_price)
                        )

            elif timeframe in ("1h", "4h"):
                if close_price is not None and candle_ts is not None:
                    htf_last = state.get("htf_last_ts", {}).get(symbol, {})
                    last_htf_ts = htf_last.get(timeframe)
                    if candle_ts != last_htf_ts:
                        htf_last[timeframe] = candle_ts
                        htf_closes = state.get("htf_closes", {}).get(symbol, {})
                        htf_opens = state.get("htf_opens", {}).get(symbol, {})
                        if timeframe in htf_closes and close_price is not None:
                            htf_closes[timeframe].append(float(close_price))
                        if timeframe in htf_opens and open_price is not None:
                            htf_opens[timeframe].append(float(open_price))

        elif message.get("type") == "order_book":
            bids: list[Any] = message.get("bids", [])
            asks: list[Any] = message.get("asks", [])
            raw_ts = message.get("timestamp")
            ts = (
                datetime.fromtimestamp(raw_ts / 1000, tz=timezone.utc)
                if isinstance(raw_ts, (int, float))
                else datetime.now(tz=timezone.utc)
            )
            if bids and asks:
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                state.setdefault("mt5_last_quote", {})[symbol] = {
                    "bid": float(bids[0][0]),
                    "ask": float(asks[0][0]),
                    "mid": mid_price,
                    "ts_ms": int(raw_ts) if isinstance(raw_ts, (int, float)) else 0,
                }

                bid_vol = sum(float(b[1]) for b in bids[:5])
                ask_vol = sum(float(a[1]) for a in asks[:5])
                obi_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
                state.get("obi_ratios", {})[symbol] = obi_ratio
                
                # ── ASYNC DB PERSISTENCE ──────────────────────────────────────
                # Non-critical market ticks are buffered in background.
                asyncio.create_task(db.insert_market_tick(
                    symbol=symbol,
                    best_bid=float(bids[0][0]),
                    bid_volume=float(bids[0][1]),
                    best_ask=float(asks[0][0]),
                    ask_volume=float(asks[0][1]),
                    timestamp=ts,
                ))

                # ── TICK SAMPLING FOR ORDER BOOK ──────────────────────────────
                if (_now_mono - _last_check_close.get(symbol, 0.0)) >= CHECK_CLOSE_INTERVAL_S:
                    _last_check_close[symbol] = _now_mono
                    try:
                        pnl = await paper_executor.check_and_close(
                            mid_price,
                            symbol=symbol,
                            timestamp=ts,
                            current_atr=state.get("atrs", {}).get(symbol),
                        )
                        if pnl is not None:
                            logger.info(
                                "💰 [CIERRE] %s | PnL: %.4f USDT | OrderBook Tick",
                                symbol, pnl
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("⚠️ [ALERTA] check_and_close (order_book) failed: %s", exc)
        
        market_queue.task_done()
