"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs the Binance WebSocket client (for all symbols in WATCHLIST) and the
crypto-news collector concurrently; each incoming trade is logged together
with the latest sentiment score.  Every order-book snapshot and scored
headline is persisted to TimescaleDB.

An ML predictor (XGBoost) is warm-started from historical market data at
startup and emits a BUY / SELL / HOLD signal for each symbol independently.

When the signal is BUY the RiskManager sizes a position via the Half-Kelly
Criterion (capped to 1/max_positions of the portfolio) and the PaperExecutor
simulates the trade entry.  Up to max_positions trades may be open
simultaneously, one per symbol.
"""

from __future__ import annotations

import asyncio
import logging
import json
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

from data_ingestion.news_client import DEFAULT_FEED_URL, run_news_client
from data_ingestion.websocket_client import BinanceWebSocketClient
from database.db_manager import close_db, db, init_db
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager
from strategy.ml_predictor import MLPredictor
from strategy.sentiment_analyzer import SentimentAnalyzer

load_dotenv()

# ── Multi-asset watchlist ─────────────────────────────────────────────────────
WATCHLIST: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# ── [PRO] News Filter parameters ─────────────────────────────────────────────
# Maximum allowed sentiment swing within the 10-minute observation window.
_NEWS_FILTER_VOLATILITY_THRESHOLD: float = 0.4
# Duration (minutes) of the global HOLD period triggered by the filter.
_NEWS_FILTER_HOLD_MINUTES: int = 30

# ── [PRO] Weekly Re-trainer ───────────────────────────────────────────────────
_RETRAINER_DATA_LIMIT: int = 10_000   # price rows to fetch for retraining
_MODEL_PATH = Path(__file__).parent / "models" / "xgb_live.json"

# ── [ELITE] Funding Rate fetcher ──────────────────────────────────────────────
_FUNDING_RATE_INTERVAL_HOURS: int = 4   # how often to refresh funding rates


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
    logger = logging.getLogger("clawdbot.market")
    ticks = 0
    while True:
        message = await market_queue.get()
        symbol: str = message.get("symbol", "BTC/USDT")

        if message.get("type") == "trade":
            price = message.get("price")
            sentiment = state.get("sentiment", 0.0)
            ticks += 1

            # Solo imprimimos el precio cada 50 ticks para no inundar la pantalla
            if ticks % 50 == 0:
                prices_buf = state["prices"].get(symbol, [])
                logger.info(
                    "%s price=%.2f  sentiment_score=%.4f | Buffer: %d/500",
                    symbol, price, sentiment, len(prices_buf),
                )

            if price is not None:
                try:
                    pnl = await paper_executor.check_and_close(float(price), symbol=symbol)
                    if pnl is not None:
                        logger.info(
                            "Position closed on trade tick  symbol=%s  pnl=%.4f  total_pnl=%.4f",
                            symbol,
                            pnl,
                            paper_executor.total_pnl,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("check_and_close failed: %s", exc)

        elif message.get("type") == "kline":
            # ----------------------------------------------------------------
            # Update the prices buffer with 15-minute candle close prices.
            # Deduplicate by timestamp so each completed candle contributes
            # exactly one entry – matching the backtester's OHLCV data.
            # ----------------------------------------------------------------
            close_price = message.get("close")
            candle_ts = message.get("timestamp")
            if close_price is not None and candle_ts is not None:
                last_ts = state["last_kline_ts"].get(symbol)
                if candle_ts != last_ts:
                    state["last_kline_ts"][symbol] = candle_ts
                    prices_dict: dict[str, deque[float]] = state["prices"]
                    if symbol in prices_dict:
                        prices_dict[symbol].append(float(close_price))
                    logger.info(
                        "📊 [KLINE] %s – new 15m candle close=%.2f | Buffer: %d/%d",
                        symbol,
                        float(close_price),
                        len(prices_dict.get(symbol, [])),
                        prices_dict[symbol].maxlen if symbol in prices_dict else 0,
                    )

        elif message.get("type") == "order_book":
            bids: list[Any] = message.get("bids", [])
            asks: list[Any] = message.get("asks", [])
            raw_ts = message.get("timestamp")
            ts = (
                datetime.fromtimestamp(raw_ts / 1000, tz=timezone.utc)
                if isinstance(raw_ts, (int, float))
                else datetime.now(tz=timezone.utc)
            )

            # [ELITE] Store the latest OBI ratio in shared state
            obi = message.get("obi", 0.0)
            state["obi"][symbol] = obi

            if bids and asks:
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2.0
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

                try:
                    pnl = await paper_executor.check_and_close(mid_price, symbol=symbol, timestamp=ts)
                    if pnl is not None:
                        logger.info(
                            "Position closed on order-book tick  symbol=%s  pnl=%.4f  total_pnl=%.4f",
                            symbol,
                            pnl,
                            paper_executor.total_pnl,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("check_and_close (order_book) failed: %s", exc)
        market_queue.task_done()


async def signal_emitter(
    state: dict[str, Any],
    predictor: MLPredictor,
    paper_executor: PaperExecutor,
    watchlist: list[str],
    interval: int = 15,
) -> None:
    logger = logging.getLogger("clawdbot.signal")
    while True:
        await asyncio.sleep(interval)
        sentiment: float = state.get("sentiment", 0.0)

        # ------------------------------------------------------------------
        # [PRO] Advanced News Filter
        # ------------------------------------------------------------------
        # Record the current sentiment reading with its timestamp.
        now = datetime.now(tz=timezone.utc)
        sentiment_history: deque[tuple[datetime, float]] = state["sentiment_history"]
        sentiment_history.append((now, sentiment))

        # Prune entries older than the 10-minute observation window.
        cutoff = now - timedelta(minutes=10)
        while sentiment_history and sentiment_history[0][0] < cutoff:
            sentiment_history.popleft()

        # Check for high-volatility sentiment fluctuation.
        hold_until: datetime | None = state.get("news_hold_until")
        if len(sentiment_history) >= 2:
            scores = [s for _, s in sentiment_history]
            if max(scores) - min(scores) > _NEWS_FILTER_VOLATILITY_THRESHOLD:
                new_hold_until = now + timedelta(minutes=_NEWS_FILTER_HOLD_MINUTES)
                # Extend the HOLD window on every new trigger.
                if hold_until is None or new_hold_until > hold_until:
                    state["news_hold_until"] = new_hold_until
                    logger.info(
                        "[PRO] News Filter triggered – sentiment swing %.4f > %.2f "
                        "in the last 10 min. Global HOLD until %s.",
                        max(scores) - min(scores),
                        _NEWS_FILTER_VOLATILITY_THRESHOLD,
                        new_hold_until.isoformat(),
                    )

        # Honour the active HOLD period: skip all signal evaluation.
        hold_until = state.get("news_hold_until")
        if hold_until is not None and now < hold_until:
            remaining = int((hold_until - now).total_seconds() / 60)
            logger.info(
                "[PRO] News Filter active – global HOLD in effect (%d min remaining).",
                remaining,
            )
            continue
        elif hold_until is not None and now >= hold_until:
            # Clear the expired HOLD.
            state["news_hold_until"] = None

        for symbol in watchlist:
            prices: list[float] = list(state["prices"].get(symbol, []))

            if len(prices) < 50:
                logger.info(
                    "⏳ [AI WARMUP] %s – Recopilando datos... (%d/50 ticks necesarios)",
                    symbol,
                    len(prices),
                )
                continue

            obi_ratio: float = state["obi"][symbol]
            funding_rate: float = state["funding_rates"][symbol]

            signal = predictor.generate_signal(prices, sentiment, obi_ratio, funding_rate)
            win_prob: float = predictor.predict_proba(prices, sentiment, obi_ratio, funding_rate) or 0.0

            logger.info(
                "🧠 [AI THOUGHT] %s – Signal: %s | Confidence: %.2f%% | Prices in buffer: %d | Sentiment: %.4f",
                symbol,
                signal,
                win_prob * 100,
                len(prices),
                sentiment,
            )

            if signal == "BUY" and prices:
                entry_price = prices[-1]
                try:
                    opened = await paper_executor.try_open_trade(
                        entry_price=entry_price,
                        win_probability=win_prob,
                        symbol=symbol,
                    )
                    if not opened:
                        logger.info(
                            "⚠️ BUY signal ignored for %s – position already open, "
                            "max positions reached, or insufficient balance.",
                            symbol,
                        )
                    else:
                        logger.info(
                            "🚀 [TRADE OPENED] Comprando %s simulado a %.2f",
                            symbol,
                            entry_price,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Paper trade open failed for %s: %s", symbol, exc)


async def weekly_retrainer(
    predictor: MLPredictor,
    watchlist: list[str],
    model_path: Path,
) -> None:
    """[PRO] Background task – re-trains the ML model every Sunday at 00:00 UTC.

    On each trigger the function:

    1. Fetches the latest :data:`_RETRAINER_DATA_LIMIT` price rows from
       TimescaleDB for every symbol in *watchlist*.
    2. Concatenates the price series and calls :meth:`MLPredictor.warm_start`
       to re-fit the model.
    3. Saves the updated model to *model_path* (``models/xgb_live.json``).
    4. Hot-reloads the saved file back into *predictor* so the running process
       immediately uses the fresh model without a restart.
    """
    log = logging.getLogger("clawdbot.retrainer")

    def _seconds_until_next_sunday_midnight() -> float:
        """Return the number of seconds until the next Sunday 00:00 UTC."""
        now = datetime.now(tz=timezone.utc)
        # isoweekday(): Monday=1 … Sunday=7.  Compute days until the next
        # Sunday, then set the time component to 00:00:00 on that date.
        days_until_sunday = (7 - now.isoweekday()) % 7
        candidate = (now + timedelta(days=days_until_sunday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # If the computed candidate is already in the past (e.g. it is Sunday
        # at 01:00 so candidate fell at Sunday 00:00), advance by one week.
        if candidate < now:
            candidate += timedelta(weeks=1)
        return (candidate - now).total_seconds()

    while True:
        wait_secs = _seconds_until_next_sunday_midnight()
        log.info(
            "[PRO] Weekly Re-trainer sleeping %.1f hours until Sunday 00:00 UTC.",
            wait_secs / 3600,
        )
        await asyncio.sleep(wait_secs)

        log.info("[PRO] Weekly Re-training started – fetching latest market data.")
        all_prices: list[float] = []
        for sym in watchlist:
            try:
                prices = await db.fetch_market_data(symbol=sym, limit=_RETRAINER_DATA_LIMIT)
                all_prices.extend(prices)
                log.info("[PRO] Fetched %d prices for %s.", len(prices), sym)
            except Exception as exc:  # noqa: BLE001
                log.warning("[PRO] Could not fetch data for %s: %s", sym, exc)

        if len(all_prices) < 50:
            log.warning(
                "[PRO] Re-training skipped – only %d price samples available (need ≥ 50).",
                len(all_prices),
            )
            continue

        log.info("[PRO] Re-training model on %d total price samples.", len(all_prices))
        success = predictor.warm_start(prices=all_prices)
        if not success:
            log.warning("[PRO] Re-training failed (warm_start returned False).")
            continue

        saved = predictor.save_model(model_path)
        if not saved:
            log.warning("[PRO] Re-training complete but model could not be saved to %s.", model_path)
            continue

        # Hot-reload: load the freshly saved model back so inference picks it
        # up immediately (validates that the file round-trips correctly).
        reloaded = predictor.load_model(model_path)
        if reloaded:
            log.info("[PRO] Weekly Re-training complete – model hot-reloaded from %s.", model_path)
        else:
            log.warning(
                "[PRO] Re-training complete but hot-reload from %s failed.", model_path
            )


async def funding_rate_fetcher(
    state: dict[str, Any],
    watchlist: list[str],
    interval_hours: int = _FUNDING_RATE_INTERVAL_HOURS,
) -> None:
    """[ELITE] Background task – refresh Binance perpetual funding rates every *interval_hours*.

    Funding rates act as a Greed/Fear indicator:
    * A high positive rate signals market greed → the ML predictor applies a BUY penalty.
    * A negative rate signals market fear → can be used to boost BUY confidence.

    Rates are stored in ``state["funding_rates"]`` keyed by symbol.
    """
    log = logging.getLogger("clawdbot.funding")
    exchange = ccxt_async.binance({"enableRateLimit": True})
    try:
        while True:
            log.info(
                "[ELITE] Funding Rate Fetcher – refreshing rates for %s.", watchlist
            )
            for sym in watchlist:
                try:
                    # fetch_funding_rate returns a dict with 'fundingRate' key
                    data = await exchange.fetch_funding_rate(sym)
                    rate = float(data.get("fundingRate") or 0.0)
                    state["funding_rates"][sym] = rate
                    log.info("[ELITE] Funding rate %s = %.6f", sym, rate)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[ELITE] Could not fetch funding rate for %s: %s", sym, exc)
            await asyncio.sleep(interval_hours * 3600)
    finally:
        await exchange.close()


async def dashboard_logger(
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    interval: int = 60,
) -> None:
    logger = logging.getLogger("clawdbot.dashboard")
    while True:
        await asyncio.sleep(interval)
        open_symbols = list(paper_executor.open_positions.keys())
        logger.info(
            "📊 [DASHBOARD] Total PnL: %.4f USDT  |  Balance: %.2f USDT  |  "
            "Open positions: %d/%d  |  Symbols: %s",
            paper_executor.total_pnl,
            risk_manager.balance,
            len(open_symbols),
            risk_manager.max_positions,
            open_symbols if open_symbols else "none",
        )


async def main() -> None:
    logger = setup_logging()
    logger.info("🚀 ClawdBot starting up...")

    await init_db()

    market_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    news_queue: asyncio.Queue[list[str]] = asyncio.Queue()
    shared_state: dict[str, Any] = {
        "sentiment": 0.0,
        "prices": {symbol: deque(maxlen=500) for symbol in WATCHLIST},
        # [PRO] News Filter state
        "sentiment_history": deque(),  # stores (datetime, float) tuples
        "news_hold_until": None,       # datetime | None
        # Per-symbol last seen kline timestamp for deduplication
        "last_kline_ts": {symbol: None for symbol in WATCHLIST},
        # [ELITE] Order Book Imbalance per symbol
        "obi": {symbol: 0.0 for symbol in WATCHLIST},
        # [ELITE] Funding rates per symbol (updated every 4 h)
        "funding_rates": {symbol: 0.0 for symbol in WATCHLIST},
    }

    analyzer = SentimentAnalyzer()
    predictor = MLPredictor()
    ws_client = BinanceWebSocketClient(queue=market_queue, watchlist=WATCHLIST)

    risk_manager = RiskManager(initial_balance=10_000.0)
    paper_executor = PaperExecutor(db=db, risk_manager=risk_manager)

    # ------------------------------------------------------------------
    # [ELITE] Fast REST Warmup – download last 100 15m candles per symbol
    # ------------------------------------------------------------------
    model_loaded = predictor.load_model(_MODEL_PATH)
    if model_loaded:
        logger.info("✅ Pre-trained model loaded from %s.", _MODEL_PATH)

    logger.info("[ELITE] Fast REST Warmup – fetching last 100 15m candles for each symbol...")
    rest_exchange = ccxt_async.binance({"enableRateLimit": True})
    try:
        for sym in WATCHLIST:
            try:
                ohlcv = await rest_exchange.fetch_ohlcv(sym, timeframe="15m", limit=100)
                if ohlcv:
                    close_prices = [float(candle[4]) for candle in ohlcv]
                    shared_state["prices"][sym].extend(close_prices)
                    logger.info(
                        "[ELITE] Warmup complete for %s – %d candles loaded into price buffer.",
                        sym,
                        len(close_prices),
                    )
                    # Train the model once using the first available symbol's data
                    if not predictor.is_trained:
                        predictor.warm_start(prices=close_prices)
                        logger.info(
                            "[ELITE] ML model warm-started with %d REST candles for %s.",
                            len(close_prices),
                            sym,
                        )
                else:
                    logger.warning("[ELITE] No OHLCV data returned for %s.", sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ELITE] REST warmup failed for %s: %s", sym, exc)
    finally:
        await rest_exchange.close()

    try:
        await asyncio.gather(
            ws_client.run(),
            run_news_client(news_queue),
            sentiment_processor(news_queue, shared_state, analyzer, source=DEFAULT_FEED_URL),
            market_consumer(market_queue, shared_state, paper_executor),
            signal_emitter(shared_state, predictor, paper_executor, watchlist=WATCHLIST, interval=15),
            dashboard_logger(paper_executor, risk_manager, interval=60),
            weekly_retrainer(predictor, watchlist=WATCHLIST, model_path=_MODEL_PATH),
            funding_rate_fetcher(shared_state, watchlist=WATCHLIST),
        )
    finally:
        await close_db()

    logger.info("🛑 ClawdBot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())