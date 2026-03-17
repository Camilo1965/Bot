"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs the Binance WebSocket client and the crypto-news collector concurrently;
each incoming BTC trade is logged together with the latest sentiment score.
Every order-book snapshot and scored headline is persisted to TimescaleDB.

An ML predictor (XGBoost) is warm-started from historical market data at
startup and emits a BUY / SELL / HOLD signal every 60 seconds.

When the signal is **BUY** the :class:`~risk.risk_manager.RiskManager`
sizes a position via the Half-Kelly Criterion and the
:class:`~execution.paper_executor.PaperExecutor` simulates the trade entry.
Open positions are monitored on every market tick and closed automatically
when the static Stop-Loss (1.5 %) or Take-Profit (3 %) is hit.

A dashboard task prints the total simulated PnL and current balance every
5 minutes.
"""

from __future__ import annotations

import asyncio
import logging
import json
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from data_ingestion.news_client import DEFAULT_FEED_URL, run_news_client
from data_ingestion.websocket_client import BinanceWebSocketClient
from database.db_manager import close_db, db, init_db
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager
from strategy.ml_predictor import MLPredictor
from strategy.sentiment_analyzer import SentimentAnalyzer

load_dotenv()


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
    source: str = "cointelegraph",
) -> None:
    """Consume headline batches from *news_queue*, update shared *state*, and
    persist each scored headline to TimescaleDB."""
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
        for headline in headlines:
            headline_score = analyzer.score_headline(headline)
            ts = datetime.now(tz=timezone.utc)
            try:
                await db.insert_sentiment(
                    headline=headline,
                    sentiment_score=headline_score,
                    source=source,
                    timestamp=ts,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB insert_sentiment failed: %s", exc)
        news_queue.task_done()


async def market_consumer(
    market_queue: asyncio.Queue[dict[str, Any]],
    state: dict[str, Any],
    paper_executor: PaperExecutor,
) -> None:
    """Log BTC trade prices alongside the latest sentiment score, persist
    order-book snapshots to TimescaleDB, maintain a rolling price buffer,
    and check open paper positions for SL/TP exits on every tick.

    Parameters
    ----------
    market_queue:
        Queue of market messages produced by the WebSocket client.
    state:
        Shared application state dict (``prices``, ``sentiment``).
    paper_executor:
        :class:`~execution.paper_executor.PaperExecutor` instance used to
        monitor and close open positions when SL/TP thresholds are hit.
    """
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
            # Check whether the open position has hit SL or TP
            if price is not None and paper_executor.open_position is not None:
                try:
                    pnl = await paper_executor.check_and_close(float(price))
                    if pnl is not None:
                        logger.info(
                            "Position closed on trade tick  pnl=%.4f  total_pnl=%.4f",
                            pnl,
                            paper_executor.total_pnl,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("check_and_close failed: %s", exc)
        elif message.get("type") == "order_book":
            symbol: str = message.get("symbol", "")
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
                state["prices"].append(mid_price)
                try:
                    await db.insert_market_tick(
                        symbol=symbol,
                        best_bid=float(bids[0][0]),
                        bid_volume=float(bids[0][1]),
                        best_ask=float(asks[0][0]),
                        ask_volume=float(asks[0][1]),
                        timestamp=ts,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DB insert_market_tick failed: %s", exc)
                # Also check SL/TP on order-book mid-price updates
                if paper_executor.open_position is not None:
                    try:
                        pnl = await paper_executor.check_and_close(mid_price, ts)
                        if pnl is not None:
                            logger.info(
                                "Position closed on order-book tick  pnl=%.4f  total_pnl=%.4f",
                                pnl,
                                paper_executor.total_pnl,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("check_and_close (order_book) failed: %s", exc)
        else:
            logger.info("Received message: %s", message)
        market_queue.task_done()


async def signal_emitter(
    state: dict[str, Any],
    predictor: MLPredictor,
    paper_executor: PaperExecutor,
    interval: int = 60,
) -> None:
    """Emit a BUY / SELL / HOLD AI signal every *interval* seconds.

    On a BUY signal the :class:`~execution.paper_executor.PaperExecutor`
    attempts to open a paper trade sized by the Half-Kelly Criterion.

    Parameters
    ----------
    state:
        Shared application state dict (``prices``, ``sentiment``).
    predictor:
        Trained :class:`~strategy.ml_predictor.MLPredictor` instance.
    paper_executor:
        :class:`~execution.paper_executor.PaperExecutor` used to open
        simulated trades when the signal is BUY.
    interval:
        Seconds between signal evaluations.  Defaults to 60.
    """
    logger = logging.getLogger("clawdbot.signal")
    while True:
        await asyncio.sleep(interval)
        prices: list[float] = list(state["prices"])
        sentiment: float = state.get("sentiment", 0.0)
        signal = predictor.generate_signal(prices, sentiment)
        win_prob: float = predictor.predict_proba(prices, sentiment) or 0.0
        logger.info(
            "AI Signal: %s  prices_in_buffer=%d  sentiment=%.4f  win_prob=%.4f",
            signal,
            len(prices),
            sentiment,
            win_prob,
        )

        if signal == "BUY" and prices:
            entry_price = prices[-1]
            try:
                opened = await paper_executor.try_open_trade(
                    entry_price=entry_price,
                    win_probability=win_prob,
                )
                if not opened:
                    logger.info(
                        "BUY signal ignored – position already open or insufficient balance."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Paper trade open failed: %s", exc)


async def dashboard_logger(
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    interval: int = 300,
) -> None:
    """Print a summary of the simulated trading account every *interval* seconds.

    Logs the total realised PnL, current simulated balance, and whether a
    position is currently open.

    Parameters
    ----------
    paper_executor:
        :class:`~execution.paper_executor.PaperExecutor` instance whose
        ``total_pnl`` and ``open_position`` attributes are reported.
    risk_manager:
        :class:`~risk.risk_manager.RiskManager` instance whose ``balance``
        is reported.
    interval:
        Seconds between dashboard log lines.  Defaults to 300 (5 minutes).
    """
    logger = logging.getLogger("clawdbot.dashboard")
    while True:
        await asyncio.sleep(interval)
        logger.info(
            "DASHBOARD ── Total PnL: %.4f USDT  |  Balance: %.2f USDT  |  Open position: %s",
            paper_executor.total_pnl,
            risk_manager.balance,
            "YES" if paper_executor.open_position is not None else "NO",
        )


async def main() -> None:
    logger = setup_logging()
    logger.info("ClawdBot starting up")

    await init_db()

    market_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    news_queue: asyncio.Queue[list[str]] = asyncio.Queue()
    # Rolling buffer of mid-prices (up to 500 ticks) for the ML predictor
    shared_state: dict[str, Any] = {
        "sentiment": 0.0,
        "prices": deque(maxlen=500),
    }

    analyzer = SentimentAnalyzer()
    predictor = MLPredictor()
    ws_client = BinanceWebSocketClient(queue=market_queue)

    risk_manager = RiskManager(initial_balance=10_000.0)
    paper_executor = PaperExecutor(db=db, risk_manager=risk_manager)

    # Warm-start the ML model from historical data already stored in TimescaleDB
    try:
        historical_prices = await db.fetch_market_data(symbol="BTC/USDT", limit=500)
        if historical_prices:
            shared_state["prices"].extend(historical_prices)
            predictor.warm_start(prices=historical_prices)
        else:
            logger.info("No historical market data found – model will train on live data.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not warm-start ML model: %s", exc)

    try:
        await asyncio.gather(
            ws_client.run(),
            run_news_client(news_queue),
            sentiment_processor(news_queue, shared_state, analyzer, source=DEFAULT_FEED_URL),
            market_consumer(market_queue, shared_state, paper_executor),
            signal_emitter(shared_state, predictor, paper_executor),
            dashboard_logger(paper_executor, risk_manager),
        )
    finally:
        await close_db()

    logger.info("ClawdBot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
