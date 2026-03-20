"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs the Binance WebSocket client (for all symbols in WATCHLIST) and a
Gemini-powered sentiment refresher concurrently; each incoming trade is
logged together with the latest sentiment score.  Every order-book snapshot
and scored headline is persisted to TimescaleDB.

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
import os
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

from data_ingestion.funding_rate_client import FundingRateClient
from data_ingestion.news_scraper import fetch_crypto_headlines
from data_ingestion.websocket_client import BinanceWebSocketClient
from database.db_manager import close_db, db, init_db
from execution.binance_executor import create_exchange, fetch_open_positions, fetch_total_wallet_balance
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager
from strategy.ml_predictor import MLPredictor, compute_htf_trend
from strategy.sentiment_llm import get_gemini_sentiment

load_dotenv()

# ── ANSI colour helpers (no extra dependency) ─────────────────────────────────
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _check_env() -> None:
    """Validate required environment variables before the bot starts.

    * Verifies that a ``.env`` file exists next to this module.
    * Confirms that ``GEMINI_API_KEY`` is set and non-empty.

    On failure a human-readable, colorized message is printed to *stderr* and
    the process exits with code 1 so no messy traceback reaches the user.
    """
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print(
            f"\n{_BOLD}{_RED}⚠️  ERROR:{_RESET}{_RED} .env file not found.{_RESET}\n"
            f"  Please copy {_YELLOW}.env.example{_RESET} to {_YELLOW}.env{_RESET}"
            " and fill in your credentials:\n"
            f"    cp .env.example .env\n"
            f"  Then set {_BOLD}GEMINI_API_KEY{_RESET} inside .env.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print(
            f"\n{_BOLD}{_RED}⚠️  ERROR:{_RESET}{_RED} GEMINI_API_KEY is missing or empty.{_RESET}\n"
            f"  Open your {_YELLOW}.env{_RESET} file and add:\n"
            f"    {_BOLD}GEMINI_API_KEY=your_gemini_api_key_here{_RESET}\n"
            f"  You can obtain a free key at "
            f"{_YELLOW}https://aistudio.google.com/app/apikey{_RESET}\n",
            file=sys.stderr,
        )
        sys.exit(1)


_check_env()

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


async def gemini_sentiment_refresher(
    state: dict[str, Any],
    interval: int = 1800,
) -> None:
    """[LLM] Background task – refreshes the sentiment score every 30 minutes.

    Fetches the latest crypto headlines from CoinTelegraph and CoinDesk via RSS
    (using :func:`~data_ingestion.news_scraper.fetch_crypto_headlines`) and
    passes them to :func:`~strategy.sentiment_llm.get_gemini_sentiment` to
    obtain a single aggregated score.  The result is cached in
    ``state["sentiment"]`` and persisted to TimescaleDB so the GUI Dashboard's
    Sentiment Gauge always reflects the most recent value.
    """
    log = logging.getLogger("clawdbot.gemini")
    while True:
        try:
            loop = asyncio.get_event_loop()
            # Run blocking I/O in a thread executor to avoid stalling the loop.
            headlines: list[str] = await loop.run_in_executor(
                None, fetch_crypto_headlines
            )
            if headlines:
                score: float = await loop.run_in_executor(
                    None, get_gemini_sentiment, headlines
                )
                state["sentiment"] = score
                log.info(
                    "[LLM] Gemini sentiment updated – headlines=%d  score=%.4f",
                    len(headlines),
                    score,
                )
                ts = datetime.now(tz=timezone.utc)
                try:
                    # Store one aggregated DB row per refresh cycle rather than
                    # one row per headline.  This keeps the news_sentiment table
                    # lean while still making the latest Gemini score visible to
                    # the GUI Dashboard's Sentiment Gauge.
                    await db.insert_sentiment(
                        headline=f"[Gemini batch: {len(headlines)} headlines]",
                        sentiment_score=score,
                        source="gemini-1.5-flash",
                        timestamp=ts,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("[LLM] DB insert_sentiment failed: %s", exc)
            else:
                log.warning("[LLM] No headlines fetched – sentiment unchanged.")
        except asyncio.CancelledError:
            log.info("gemini_sentiment_refresher cancelled – shutting down.")
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[LLM] Gemini refresher error (%s): %s – retrying in %ds",
                type(exc).__name__,
                exc,
                interval,
            )
        await asyncio.sleep(interval)


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
            # Route kline messages by timeframe.
            # 15m candles update the primary price/high/low buffers used by
            # the ML model.  1H and 4H candles update the HTF buffers used by
            # the Multi-Timeframe Analysis (MTA) trend filter.
            # All timeframes are deduplicated by timestamp.
            # ----------------------------------------------------------------
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
                        # [ELITE] Store high / low for ADX/ATR features
                        if high_price is not None and symbol in state.get("highs", {}):
                            state["highs"][symbol].append(float(high_price))
                        if low_price is not None and symbol in state.get("lows", {}):
                            state["lows"][symbol].append(float(low_price))
                        logger.info(
                            "📊 [KLINE] %s – new 15m candle close=%.2f | Buffer: %d/%d",
                            symbol,
                            float(close_price),
                            len(prices_dict.get(symbol, [])),
                            prices_dict[symbol].maxlen if symbol in prices_dict else 0,
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
                        logger.info(
                            "📊 [HTF KLINE] %s – new %s candle close=%.2f",
                            symbol,
                            timeframe.upper(),
                            float(close_price),
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
            if bids and asks:
                mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2.0

                # [ELITE] Order Book Imbalance: total bid volume vs ask volume
                bid_vol = sum(float(b[1]) for b in bids[:5])
                ask_vol = sum(float(a[1]) for a in asks[:5])
                obi_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
                obi_ratios: dict[str, float] = state.get("obi_ratios", {})
                obi_ratios[symbol] = obi_ratio
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

            # [ELITE] Gather regime and funding inputs
            highs: list[float] = list(state.get("highs", {}).get(symbol, []))
            lows: list[float] = list(state.get("lows", {}).get(symbol, []))
            obi_ratio: float = state.get("obi_ratios", {}).get(symbol, 1.0)
            funding_rate: float = state.get("funding_rates", {}).get(symbol, 0.0)

            # [MTA] Compute higher-timeframe trend statuses
            closes_4h: list[float] = list(
                state.get("htf_closes", {}).get(symbol, {}).get("4h", [])
            )
            opens_4h: list[float] = list(
                state.get("htf_opens", {}).get(symbol, {}).get("4h", [])
            )
            closes_1h: list[float] = list(
                state.get("htf_closes", {}).get(symbol, {}).get("1h", [])
            )
            opens_1h: list[float] = list(
                state.get("htf_opens", {}).get(symbol, {}).get("1h", [])
            )

            trend_4h = compute_htf_trend(closes_4h, opens_4h or None)
            trend_1h = compute_htf_trend(closes_1h, opens_1h or None)
            trend_15m = compute_htf_trend(prices, None)

            # Update shared state and persist to DB
            state.setdefault("htf_trend", {}).setdefault(symbol, {})
            state["htf_trend"][symbol] = {
                "4h": trend_4h,
                "1h": trend_1h,
                "15m": trend_15m,
            }
            for tf, trend in (("4h", trend_4h), ("1h", trend_1h), ("15m", trend_15m)):
                try:
                    await db.upsert_htf_trend(symbol, tf, trend)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[MTA] DB upsert_htf_trend failed for %s/%s: %s", symbol, tf, exc)

            logger.info(
                "🔭 [MTA RADAR] %s – 4H=%s | 1H=%s | 15M=%s",
                symbol, trend_4h, trend_1h, trend_15m,
            )

            signal = predictor.generate_signal(
                prices,
                sentiment,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
                funding_rate=funding_rate,
                htf_trend_4h=trend_4h,
                htf_trend_1h=trend_1h,
            )
            win_prob: float = predictor.predict_proba(
                prices,
                sentiment,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
            ) or 0.0

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


async def preload_historical_data(
    state: dict[str, Any],
    watchlist: list[str],
    timeframe: str = "15m",
    limit: int = 1000,
) -> None:
    """[ELITE] Fast REST Warmup – populate price/high/low buffers before WebSocket starts.

    Fetches *limit* historical OHLCV candles for every symbol in *watchlist*
    via the Binance REST API (ccxt.async_support) and appends the close, high,
    and low values to the respective shared-state deques.  This means that when
    the first live WebSocket kline arrives the buffer is already pre-filled,
    bypassing the ``[AI WARMUP]`` phase entirely.

    Also fetches 1H and 4H candles (up to *limit* each) to pre-fill the
    higher-timeframe buffers used by the Multi-Timeframe Analysis (MTA) trend
    filter, providing a deep history for fractal analysis.
    """
    log = logging.getLogger("clawdbot.preload")
    exchange = ccxt_async.binance({"enableRateLimit": True})
    try:
        for symbol in watchlist:
            # ── 15m primary buffer ─────────────────────────────────────────
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                # ohlcv rows: [timestamp, open, high, low, close, volume]
                for candle in ohlcv:
                    _, open_price, high, low, close, _volume = candle
                    if close is not None and symbol in state["prices"]:
                        state["prices"][symbol].append(float(close))
                    if high is not None and symbol in state.get("highs", {}):
                        state["highs"][symbol].append(float(high))
                    if low is not None and symbol in state.get("lows", {}):
                        state["lows"][symbol].append(float(low))
                log.info(
                    "[ELITE] Preloaded %d historical candles for %s.",
                    len(ohlcv),
                    symbol,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[ELITE] Could not preload historical candles for %s: %s",
                    symbol,
                    exc,
                )

            # ── 1H and 4H HTF buffers ──────────────────────────────────────
            for htf in ("1h", "4h"):
                try:
                    htf_ohlcv = await exchange.fetch_ohlcv(
                        symbol, timeframe=htf, limit=limit
                    )
                    htf_closes = state.get("htf_closes", {}).get(symbol, {})
                    htf_opens = state.get("htf_opens", {}).get(symbol, {})
                    for candle in htf_ohlcv:
                        _, open_price, _high, _low, close, _volume = candle
                        if close is not None and htf in htf_closes:
                            htf_closes[htf].append(float(close))
                        if open_price is not None and htf in htf_opens:
                            htf_opens[htf].append(float(open_price))
                    log.info(
                        "[MTA] Preloaded %d %s candles for %s.",
                        len(htf_ohlcv),
                        htf.upper(),
                        symbol,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "[MTA] Could not preload %s candles for %s: %s",
                        htf.upper(),
                        symbol,
                        exc,
                    )
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
    shared_state: dict[str, Any] = {
        "sentiment": 0.0,
        "prices": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        # [ELITE] OHLCV buffers for ADX / ATR computation
        "highs": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        "lows": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        # [ELITE] Latest Order Book Imbalance ratio per symbol
        "obi_ratios": {symbol: 1.0 for symbol in WATCHLIST},
        # [ELITE] Latest perpetual-futures funding rate per symbol
        "funding_rates": {symbol: 0.0 for symbol in WATCHLIST},
        # [PRO] News Filter state
        "sentiment_history": deque(),  # stores (datetime, float) tuples
        "news_hold_until": None,       # datetime | None
        # Per-symbol last seen kline timestamp for deduplication
        "last_kline_ts": {symbol: None for symbol in WATCHLIST},
        # [MTA] Higher-timeframe OHLCV buffers (1h and 4h)
        "htf_closes": {
            symbol: {"1h": deque(maxlen=1000), "4h": deque(maxlen=1000)}
            for symbol in WATCHLIST
        },
        "htf_opens": {
            symbol: {"1h": deque(maxlen=1000), "4h": deque(maxlen=1000)}
            for symbol in WATCHLIST
        },
        "htf_last_ts": {
            symbol: {"1h": None, "4h": None}
            for symbol in WATCHLIST
        },
        # [MTA] Computed trend status per symbol/timeframe
        "htf_trend": {
            symbol: {"1h": "neutral", "4h": "neutral", "15m": "neutral"}
            for symbol in WATCHLIST
        },
    }

    predictor = MLPredictor()
    ws_client = BinanceWebSocketClient(queue=market_queue, watchlist=WATCHLIST)
    funding_rate_client = FundingRateClient(symbols=WATCHLIST, state=shared_state)

    # ── Execution mode: Testnet / Live / Paper ───────────────────────────────
    use_testnet: bool = os.environ.get("USE_BINANCE_TESTNET", "False").strip().lower() in (
        "true", "1", "yes"
    )
    paper_trading: bool = os.environ.get("PAPER_TRADING", "True").strip().lower() in (
        "true", "1", "yes"
    )
    api_key: str = os.environ.get("EXCHANGE_API_KEY", "").strip()
    api_secret: str = os.environ.get("EXCHANGE_SECRET", "").strip()

    exchange_client: ccxt_async.binanceusdm | None = None
    initial_balance: float = 10_000.0

    if use_testnet:
        # USE_BINANCE_TESTNET takes priority over PAPER_TRADING.
        if not api_key or not api_secret:
            logger.warning(
                "USE_BINANCE_TESTNET=True but EXCHANGE_API_KEY/EXCHANGE_SECRET are not "
                "set – falling back to paper trading with %.2f USDT.",
                initial_balance,
            )
        else:
            exchange_client = create_exchange(api_key, api_secret, testnet=True)
            fetched: float | None = await fetch_total_wallet_balance(exchange_client)
            if fetched is not None:
                initial_balance = fetched
                logger.info(
                    "✅ [TESTNET] Binance Futures Testnet total wallet balance fetched: %.2f",
                    initial_balance,
                )
            else:
                logger.warning(
                    "[TESTNET] Could not fetch balance from Binance Testnet – "
                    "using default %.2f USDT.",
                    initial_balance,
                )
            logger.info(
                "🔗 [TESTNET] Orders will be sent to Binance Futures Testnet API."
            )
    elif not paper_trading:
        # Live trading (non-testnet).
        if not api_key or not api_secret:
            logger.warning(
                "PAPER_TRADING=False but EXCHANGE_API_KEY/EXCHANGE_SECRET are not "
                "set – running in pure paper trading mode."
            )
        else:
            exchange_client = create_exchange(api_key, api_secret, testnet=False)
            fetched = await fetch_total_wallet_balance(exchange_client)
            if fetched is not None:
                initial_balance = fetched
                logger.info(
                    "✅ [LIVE] Binance Futures live total wallet balance fetched: %.2f",
                    initial_balance,
                )
            else:
                logger.warning(
                    "[LIVE] Could not fetch balance from Binance live API – "
                    "using default %.2f USDT.",
                    initial_balance,
                )
            logger.info(
                "🔗 [LIVE] Orders will be sent to Binance Futures live API."
            )
    else:
        logger.info(
            "📝 [PAPER] Running in pure paper trading mode (internal simulation) "
            "with %.2f USDT.",
            initial_balance,
        )

    risk_manager = RiskManager(initial_balance=initial_balance)
    paper_executor = PaperExecutor(db=db, risk_manager=risk_manager, exchange=exchange_client)

    # ------------------------------------------------------------------
    # [SYNC] Re-sync open positions from Binance on restart
    # ------------------------------------------------------------------
    # When the bot restarts it must not assume all positions are closed.
    # Fetch the actual open positions from Binance and restore them in
    # the local PaperExecutor/RiskManager state so that:
    #   - `can_open_position()` returns the correct headroom.
    #   - The per-symbol duplicate guard in `try_open_trade()` works.
    # NOTE: the balance was fetched from Binance above and already reflects
    # the margin tied up in open positions, so no further deduction is made.
    if exchange_client is not None:
        try:
            existing_positions = await fetch_open_positions(exchange_client)
            synced = 0
            for pos_info in existing_positions:
                sym: str = pos_info["symbol"]
                if sym not in WATCHLIST:
                    logger.info(
                        "🔄 [SYNC] Skipping position for %s (not in watchlist).",
                        sym,
                    )
                    continue
                restored = paper_executor.restore_position(
                    symbol=sym,
                    entry_price=pos_info["entry_price"],
                    position_size=pos_info["position_size"],
                )
                if restored:
                    synced += 1
                    logger.info(
                        "🔄 [SYNC] Restored open position: %s @ %.2f (notional=%.2f USDT, side=%s)",
                        sym,
                        pos_info["entry_price"],
                        pos_info["position_size"],
                        pos_info["side"],
                    )
            if synced > 0:
                logger.info(
                    "✅ [SYNC] Re-synced %d open position(s) from Binance "
                    "(Open positions: %d/%d).",
                    synced,
                    risk_manager.open_count,
                    risk_manager.max_positions,
                )
            else:
                logger.info("🔄 [SYNC] No existing open positions found on Binance.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️ [SYNC] Could not sync open positions from Binance: %s", exc)

    # ------------------------------------------------------------------
    # Attempt to load a pre-trained model; fall back to warm-start
    # ------------------------------------------------------------------
    model_loaded = predictor.load_model(_MODEL_PATH)
    if model_loaded:
        logger.info("✅ Pre-trained model loaded from %s.", _MODEL_PATH)
    else:
        logger.info("ℹ️ No pre-trained model found – will warm-start from historical DB data.")
        for sym in WATCHLIST:
            try:
                historical_prices = await db.fetch_market_data(symbol=sym, limit=1000)
                if historical_prices:
                    shared_state["prices"][sym].extend(historical_prices)
                    # Train the model once using the first available symbol's data
                    if not predictor.is_trained:
                        predictor.warm_start(prices=historical_prices)
                        logger.info(
                            "✅ ML model warm-started with %d historical prices for %s.",
                            len(historical_prices),
                            sym,
                        )
                    else:
                        logger.info(
                            "✅ Historical prices loaded for %s (%d ticks).",
                            sym,
                            len(historical_prices),
                        )
                else:
                    logger.info("ℹ️ No historical market data found for %s.", sym)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not warm-start for %s: %s", sym, exc)

    # ------------------------------------------------------------------
    # [ELITE] Fast REST Warmup – pre-fill buffers before WebSocket starts
    # ------------------------------------------------------------------
    await preload_historical_data(shared_state, WATCHLIST)

    try:
        await asyncio.gather(
            ws_client.run(),
            gemini_sentiment_refresher(shared_state),
            market_consumer(market_queue, shared_state, paper_executor),
            signal_emitter(shared_state, predictor, paper_executor, watchlist=WATCHLIST, interval=15),
            dashboard_logger(paper_executor, risk_manager, interval=60),
            weekly_retrainer(predictor, watchlist=WATCHLIST, model_path=_MODEL_PATH),
            funding_rate_client.run(),
        )
    finally:
        if exchange_client is not None:
            await exchange_client.close()
        await close_db()

    logger.info("🛑 ClawdBot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())