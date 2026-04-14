"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs an MT5 market-data poller (ticks + multi-timeframe candles) and a
Gemini-powered sentiment refresher concurrently; each incoming trade is
logged together with the latest sentiment score.  Every synthetic top-of-book
snapshot and scored headline is persisted to TimescaleDB.

An ML predictor (XGBoost) is warm-started from historical market data at
startup and emits a BUY / SELL / HOLD signal for each symbol independently.

When the signal is BUY the RiskManager sizes a position via the Half-Kelly
Criterion (capped to 1/max_positions of the portfolio) and the PaperExecutor
simulates the trade entry.  Up to max_positions trades may be open
simultaneously, one per symbol.

Before first live run, execute ``preflight.py`` (same venv as ``main.py``) to
verify PostgreSQL/TimescaleDB, MT5 login, and Telegram.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box as rich_box
from rich.logging import RichHandler

from data_ingestion.mt5_market_client import MT5MarketDataClient
from data_ingestion.news_scraper import fetch_crypto_headlines
from database.db_manager import close_db, db, init_db
from execution.mt5_executor import (
    MT5Executor,
    TIMEFRAME_H1,
    TIMEFRAME_H4,
    TIMEFRAME_M15,
    fetch_mt5_account_balance,
    initialize_mt5,
    shutdown_mt5,
)
from execution.paper_executor import PaperExecutor
from risk.risk_manager import LEVERAGE as _RISK_LEVERAGE, RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD, BUY_SENTIMENT_THRESHOLD, MLPredictor, compute_htf_trend
from strategy.feature_engineer import FeatureEngineer
from strategy.sentiment_llm import get_gemini_sentiment
from utils.telegram_notifier import send_telegram_alert

load_dotenv()

# ── ANSI colour helpers (no extra dependency) ─────────────────────────────────
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# ── Mega-Dashboard state ─────────────────────────────────────────────────────
_BOT_START_TIME: datetime | None = None
_DASHBOARD_EVENTS: deque[str] = deque(maxlen=5)
_GEMINI_ENABLED: bool = False


def _check_env() -> None:
    """Validate environment variables before the bot starts.

    * Verifies that a ``.env`` file exists next to this module.
    * Detects whether ``GEMINI_API_KEY`` is set and non-empty.
    * Runs in degraded mode when Gemini credentials are missing.
    """
    global _GEMINI_ENABLED  # noqa: PLW0603
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print(
            f"\n{_BOLD}{_YELLOW}⚠️  WARNING:{_RESET}{_YELLOW} .env file not found.{_RESET}\n"
            f"  Please copy {_YELLOW}.env.example{_RESET} to {_YELLOW}.env{_RESET}"
            " and fill in your credentials:\n"
            f"    cp .env.example .env\n"
            "  Running with current process environment variables.\n",
            file=sys.stderr,
        )

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    _GEMINI_ENABLED = bool(gemini_key)
    if not _GEMINI_ENABLED:
        print(
            f"\n{_BOLD}{_YELLOW}⚠️  WARNING:{_RESET}{_YELLOW} GEMINI_API_KEY is missing or empty.{_RESET}\n"
            f"  Open your {_YELLOW}.env{_RESET} file and add:\n"
            f"    {_BOLD}GEMINI_API_KEY=your_gemini_api_key_here{_RESET}\n"
            f"  You can obtain a free key at "
            f"{_YELLOW}https://aistudio.google.com/app/apikey{_RESET}\n"
            "  Bot will run in degraded mode with neutral sentiment.\n",
            file=sys.stderr,
        )


_check_env()

# ── Multi-asset watchlist ─────────────────────────────────────────────────────
# Keep this aligned with symbols that exist in *Market Watch* on your MT5 broker
# (see SYMBOL_MAP in execution/mt5_executor.py). Many brokers only offer majors.
WATCHLIST: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
]

# ── [PRO] News Filter parameters ─────────────────────────────────────────────
# Maximum allowed sentiment swing within the 10-minute observation window.
_NEWS_FILTER_VOLATILITY_THRESHOLD: float = 0.4
# Duration (minutes) of the global HOLD period triggered by the filter.
_NEWS_FILTER_HOLD_MINUTES: int = 30

# ── [PRO] Weekly Re-trainer ───────────────────────────────────────────────────
_RETRAINER_DATA_LIMIT: int = 10_000   # price rows to fetch for retraining
_MODEL_PATH = Path(__file__).parent / "models" / "xgb_live.json"

# Hint appended to warning messages that have more detail in the debug log.
_DEBUG_LOG_HINT = "(Revisa logs/last_session.log o bot_debug.log para detalles)"

# Set once per process in :func:`setup_logging` — also embedded in JSON lines.
_LOG_SESSION_ID: str = ""


class _DropDashboardDebugFilter(logging.Filter):
    """Avoid filling ``bot_debug.log`` with per-second dashboard DEBUG heartbeats."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "clawdbot.dashboard" and record.levelno == logging.DEBUG:
            return False
        return True


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects (includes ``session_id``)."""

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._session_id = session_id

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "session_id": self._session_id,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """Human-readable coloured formatter for the console (stdout).

    Levels are colour-coded:
    * WARNING  → yellow
    * ERROR / CRITICAL → bold red
    * INFO / DEBUG → default terminal colour
    """

    _LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG:    _RESET,
        logging.INFO:     _RESET,
        logging.WARNING:  _YELLOW,
        logging.ERROR:    _RED + _BOLD,
        logging.CRITICAL: _RED + _BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, _RESET)
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{color}{ts} | {msg}{_RESET}"


class _DashboardEventHandler(logging.Handler):
    """Captures INFO+ log messages for the dashboard events panel.

    Appends a short timestamped line to the module-level
    :data:`_DASHBOARD_EVENTS` deque so the mega-dashboard can display
    the last few operational events without interfering with other handlers.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
            msg = record.getMessage()
            if len(msg) > 78:
                msg = msg[:75] + "..."
            level_markup = {
                logging.WARNING:  "[yellow]⚠[/yellow]",
                logging.ERROR:    "[red]✖[/red]",
                logging.CRITICAL: "[bold red]‼[/bold red]",
            }.get(record.levelno, "[dim]•[/dim]")
            _DASHBOARD_EVENTS.append(f"[dim]{ts}[/dim] {level_markup} {msg}")
        except Exception:  # noqa: BLE001
            pass


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure dual-channel logging and return the root *clawdbot* logger.

    * **File** (``bot_debug.log``): JSON lines at ``DEBUG`` with
      ``session_id`` on every record.  Rotates at 5 MB (3 backups).
      Per-second ``clawdbot.dashboard`` DEBUG heartbeats are filtered out.

    * **File** (``logs/last_session.log``): human-readable ``INFO+`` only,
      **truncated on each process start** — use this to see *what happened
      this run* without parsing huge JSON.

    * **Console** (stdout): :class:`~logging.StreamHandler` at ``INFO``.

    * **Audit** (``audit.log``): ``clawdbot.audit`` only, ``propagate=False``.
    """
    global _LOG_SESSION_ID  # noqa: PLW0603

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    _LOG_SESSION_ID = uuid.uuid4().hex[:12]
    _logs_dir = Path(__file__).parent / "logs"
    _logs_dir.mkdir(exist_ok=True)

    # ── File handler: full DEBUG log in JSON (session-scoped, no dashboard spam)
    log_file = Path(__file__).parent / "bot_debug.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFormatter(session_id=_LOG_SESSION_ID))
    file_handler.addFilter(_DropDashboardDebugFilter())
    root.addHandler(file_handler)

    # ── Human-readable session log (this run only) ─────────────────────────
    session_log = _logs_dir / "last_session.log"
    session_handler = logging.FileHandler(session_log, mode="w", encoding="utf-8")
    session_handler.setLevel(logging.INFO)
    session_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(session_handler)

    # ── Console handler: filtered INFO + WARNING/ERROR ────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_ConsoleFormatter())
    root.addHandler(console_handler)

    # ── Dashboard event handler: capture INFO+ events for TUI panel ───────────
    dashboard_event_handler = _DashboardEventHandler()
    dashboard_event_handler.setLevel(logging.INFO)
    root.addHandler(dashboard_event_handler)

    # ── Audit handler: pipe-delimited risk telemetry → audit.log ─────────────
    audit_log_file = Path(__file__).parent / "audit.log"
    audit_file_handler = logging.FileHandler(audit_log_file, encoding="utf-8")
    audit_file_handler.setLevel(logging.DEBUG)
    audit_file_handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger = logging.getLogger("clawdbot.audit")
    audit_logger.setLevel(logging.DEBUG)
    audit_logger.propagate = False  # never reaches console or bot_debug.log
    audit_logger.addHandler(audit_file_handler)

    # Write a column header once per session so the file is self-describing.
    audit_logger.info(
        "# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%"
    )

    boot = logging.getLogger("clawdbot")
    boot.info(
        "SESSION_START session_id=%s pid=%s | full JSON: bot_debug.log | esta sesión: logs/last_session.log",
        _LOG_SESSION_ID,
        os.getpid(),
    )
    return boot


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
                is_first_reading = state.get("sentiment") is None
                state["sentiment"] = score
                if is_first_reading:
                    log.info(
                        "[LLM] Primera lectura de sentimiento: %.2f (Ignorando cálculo de swing por inicio de sistema).",
                        score,
                    )
                else:
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
                        source="gemini-2.5-flash",
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
    _last_price_log: dict[str, float] = {}  # symbol → last log timestamp (monotonic)
    while True:
        message = await market_queue.get()
        state["last_market_message_at"] = datetime.now(tz=timezone.utc)
        symbol: str = message.get("symbol", "BTC/USDT")

        if message.get("type") == "trade":
            price = message.get("price")
            # Guard against None sentiment (startup race) to avoid TypeError.
            sentiment = state.get("sentiment") or 0.0

            # Demoted to DEBUG – throttled to once per 60 s per symbol.
            _now = time.monotonic()
            if _now - _last_price_log.get(symbol, 0.0) >= 60.0:
                _last_price_log[symbol] = _now
                prices_buf = state["prices"].get(symbol, [])
                logger.debug(
                    "%s price=%.2f  sentiment_score=%.4f | Buffer: %d/500",
                    symbol, price, sentiment, len(prices_buf),
                )

            if price is not None:
                try:
                    pnl = await paper_executor.check_and_close(
                        float(price),
                        symbol=symbol,
                        current_atr=state.get("atrs", {}).get(symbol),
                    )
                    if pnl is not None:
                        logger.debug(
                            "Position closed on trade tick  symbol=%s  pnl=%.4f  total_pnl=%.4f",
                            symbol,
                            pnl,
                            paper_executor.total_pnl,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("⚠️ [ALERTA] check_and_close failed: %s %s", exc, _DEBUG_LOG_HINT)

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
                        logger.debug(
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
                        logger.debug(
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
                    logger.warning("⚠️ [ALERTA] DB insert_market_tick failed: %s %s", exc, _DEBUG_LOG_HINT)

                try:
                    pnl = await paper_executor.check_and_close(
                        mid_price,
                        symbol=symbol,
                        timestamp=ts,
                        current_atr=state.get("atrs", {}).get(symbol),
                    )
                    if pnl is not None:
                        logger.debug(
                            "Position closed on order-book tick  symbol=%s  pnl=%.4f  total_pnl=%.4f",
                            symbol,
                            pnl,
                            paper_executor.total_pnl,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("⚠️ [ALERTA] check_and_close (order_book) failed: %s %s", exc, _DEBUG_LOG_HINT)
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
        sentiment: float | None = state.get("sentiment")

        # ------------------------------------------------------------------
        # [PRO] Advanced News Filter
        # ------------------------------------------------------------------
        # Record the current sentiment reading with its timestamp.
        now = datetime.now(tz=timezone.utc)
        sentiment_history: deque[tuple[datetime, float]] = state["sentiment_history"]
        # Only append to history once a real Gemini score is available.
        # When sentiment is None (first boot, before first Gemini call) we skip
        # the append so the deque stays empty and no artificial swing is triggered.
        if sentiment is not None:
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
            logger.debug(
                "[PRO] News Filter active – global HOLD in effect (%d min remaining).",
                remaining,
            )
            continue
        elif hold_until is not None and now >= hold_until:
            # Clear the expired HOLD.
            state["news_hold_until"] = None

        for symbol in watchlist:
            prices: list[float] = list(state["prices"].get(symbol, []))

            # Resolve the effective sentiment score for signal generation.
            # Use neutral 0.0 until the first real Gemini reading is available.
            effective_sentiment: float = sentiment if sentiment is not None else 0.0

            if len(prices) < 50:
                logger.debug(
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

            # [ATR] Compute ATR_14 from the 15m OHLCV buffers and cache in state
            current_atr: float | None = FeatureEngineer.compute_atr(highs, lows, prices)
            if current_atr is not None:
                state.setdefault("atrs", {})[symbol] = current_atr
                logger.debug(
                    "📐 [ATR] %s – ATR_14=%.4f", symbol, current_atr
                )
            else:
                current_atr = state.get("atrs", {}).get(symbol)

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

            logger.debug(
                "🔭 [MTA RADAR] %s – 4H=%s | 1H=%s | 15M=%s",
                symbol, trend_4h, trend_1h, trend_15m,
            )

            signal = predictor.generate_signal(
                prices,
                effective_sentiment,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
                funding_rate=funding_rate,
                htf_trend_4h=trend_4h,
                htf_trend_1h=trend_1h,
            )
            win_prob: float = predictor.predict_proba(
                prices,
                effective_sentiment,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
            ) or 0.0

            # Store the latest ML confidence so dashboard_logger can display it.
            state["ml_probs"][symbol] = win_prob

            logger.debug(
                "🧠 [AI THOUGHT] %s – Signal: %s | Confidence: %.2f%% | Prices in buffer: %d | Sentiment: %.4f",
                symbol,
                signal,
                win_prob * 100,
                len(prices),
                effective_sentiment,
            )

            # ------------------------------------------------------------------
            # [SMART EXIT] Layers 1 + 4 – ML Exhaustion & TTL
            # ------------------------------------------------------------------
            # For every open position evaluate whether the current ML signal
            # warrants an early exit (trend reversal) or the TTL has elapsed.
            # This runs in the same signal-emitter cycle (every ~15 s) so it
            # never blocks the event loop with additional network calls.
            if symbol in paper_executor.open_positions and prices:
                current_price = prices[-1]
                try:
                    smart_pnl = await paper_executor.check_ml_exit(
                        current_price=current_price,
                        ml_signal=signal,
                        ml_probability=win_prob if win_prob > 0.0 else None,
                        symbol=symbol,
                        min_confidence=BUY_PROB_THRESHOLD,
                    )
                    if smart_pnl is not None:
                        logger.debug(
                            "Smart exit triggered  symbol=%s  pnl=%.4f  total_pnl=%.4f",
                            symbol,
                            smart_pnl,
                            paper_executor.total_pnl,
                        )
                        # Position was closed – skip the BUY entry logic below.
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "⚠️ [ALERTA] check_ml_exit failed for %s: %s %s",
                        symbol,
                        exc,
                        _DEBUG_LOG_HINT,
                    )

            if signal == "BUY" and prices:
                entry_price = prices[-1]
                try:
                    opened = await paper_executor.try_open_trade(
                        entry_price=entry_price,
                        win_probability=win_prob,
                        symbol=symbol,
                        sentiment_score=effective_sentiment,
                        current_atr=current_atr,
                    )
                    if not opened:
                        logger.debug(
                            "BUY signal ignored for %s – position already open, "
                            "max positions reached, or insufficient balance.",
                            symbol,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("⚠️ [ALERTA] Paper trade open failed for %s: %s %s", symbol, exc, _DEBUG_LOG_HINT)


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


async def preload_historical_data_mt5(
    state: dict[str, Any],
    watchlist: list[str],
    executor: MT5Executor,
    limit: int = 1000,
) -> None:
    """Preload 15m/1h/4h candles from MT5 to warm up buffers."""
    log = logging.getLogger("clawdbot.preload")
    for symbol in watchlist:
        try:
            df_15m = await executor.fetch_candles(
                symbol=symbol,
                timeframe=TIMEFRAME_M15,
                count=limit,
                start_pos=0,
            )
            if df_15m is not None and not df_15m.empty:
                for _, row in df_15m.iterrows():
                    close = row.get("close")
                    high = row.get("high")
                    low = row.get("low")
                    if close is not None and symbol in state["prices"]:
                        state["prices"][symbol].append(float(close))
                    if high is not None and symbol in state.get("highs", {}):
                        state["highs"][symbol].append(float(high))
                    if low is not None and symbol in state.get("lows", {}):
                        state["lows"][symbol].append(float(low))
                log.info("[MT5] Preloaded %d 15m candles for %s.", len(df_15m), symbol)
            else:
                log.warning("[MT5] No 15m candles available for %s.", symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("[MT5] Could not preload 15m candles for %s: %s", symbol, exc)

        for tf_name, tf_value in (("1h", TIMEFRAME_H1), ("4h", TIMEFRAME_H4)):
            try:
                df_htf = await executor.fetch_candles(
                    symbol=symbol,
                    timeframe=tf_value,
                    count=limit,
                    start_pos=0,
                )
                if df_htf is None or df_htf.empty:
                    log.warning("[MT5] No %s candles available for %s.", tf_name.upper(), symbol)
                    continue
                htf_closes = state.get("htf_closes", {}).get(symbol, {})
                htf_opens = state.get("htf_opens", {}).get(symbol, {})
                for _, row in df_htf.iterrows():
                    close = row.get("close")
                    open_price = row.get("open")
                    if close is not None and tf_name in htf_closes:
                        htf_closes[tf_name].append(float(close))
                    if open_price is not None and tf_name in htf_opens:
                        htf_opens[tf_name].append(float(open_price))
                log.info("[MT5] Preloaded %d %s candles for %s.", len(df_htf), tf_name.upper(), symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[MT5] Could not preload %s candles for %s: %s",
                    tf_name.upper(),
                    symbol,
                    exc,
                )


def _compute_rsi(prices: list[float], period: int = 14) -> float | None:
    """Compute RSI for the last *period* bars using simple averages.

    Returns a float in [0, 100] or ``None`` when there are insufficient bars.
    Used only by the dashboard for display; the authoritative RSI used in
    signal generation comes from :class:`~strategy.feature_engineer.FeatureEngineer`.
    """
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    avg_gain = sum(d for d in recent if d > 0) / period
    avg_loss = sum(-d for d in recent if d < 0) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def generate_dashboard(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    watchlist: list[str],
) -> Group:
    """Build and return a Rich mega-dashboard layout with system health,
    market data, and risk/events panels.

    The returned :class:`~rich.console.Group` stacks three sections:

    1. **Header panel** – uptime, API latency, model names, sentiment.
    2. **Market table** – per-symbol price, 24 h %, ATR volatility, RSI,
       HTF trend, open position, unrealised PnL, and AI action.
       Rows for open positions are highlighted with a neon-green background.
    3. **Footer columns** – risk & wallet panel (balance, available margin,
       session PnL, max drawdown) beside a live-events log.
    """
    now_utc = datetime.now(tz=timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    sentiment: float | None = state.get("sentiment")
    ml_probs: dict[str, float] = state.get("ml_probs", {})
    news_hold_until: datetime | None = state.get("news_hold_until")
    global_hold = (
        (sentiment is not None and sentiment < BUY_SENTIMENT_THRESHOLD)
        or (news_hold_until is not None and now_utc < news_hold_until)
    )

    # ── Header Panel: System Health ──────────────────────────────────────────
    if _BOT_START_TIME is not None:
        elapsed = now_utc - _BOT_START_TIME
        total_secs = int(elapsed.total_seconds())
        h, rem = divmod(total_secs, 3600)
        m, s = divmod(rem, 60)
        uptime_str = f"{h:02d}h {m:02d}m {s:02d}s"
    else:
        uptime_str = "—"

    latency_ms: float = state.get("api_latency_ms", 0.0)
    lat_color = "bright_green" if latency_ms < 100 else "yellow" if latency_ms < 500 else "bright_red"
    lat_str = f"{latency_ms:.0f} ms" if latency_ms > 0 else "—"

    sentiment_val_str = f"{sentiment:.4f}" if sentiment is not None else "—"
    s_color = "bright_green" if (sentiment or 0) >= 0.55 else "yellow" if (sentiment or 0) >= 0.35 else "bright_red"
    news_str = "[bright_red]⛔ HOLD[/bright_red]" if global_hold else "[bright_green]✅ OK[/bright_green]"

    header_text = Text(justify="left")
    header_text.append("🤖  ClawdBot  –  Mega Dashboard  |  ", style="bold cyan")
    header_text.append(now_str, style="white")
    header_text.append("\n")
    header_text.append("⏱ Uptime: ", style="dim")
    header_text.append(uptime_str, style="bold white")
    header_text.append("   │   🌐 Latencia API: ", style="dim")
    header_text.append(lat_str, style=f"bold {lat_color}")
    header_text.append("   │   🧠 Modelo: ", style="dim")
    header_text.append("XGBoost_v3 + Gemini-2.5-Flash", style="bold magenta")
    header_text.append("   │   🔮 Sentimiento IA: ", style="dim")
    header_text.append(sentiment_val_str, style=f"bold {s_color}")
    header_text.append("   │   Noticias: ", style="dim")
    header_text.append_text(Text.from_markup(news_str))

    header_panel = Panel(
        header_text,
        title="[bold cyan]⚡ SYSTEM HEALTH[/bold cyan]",
        border_style="cyan",
        box=rich_box.ROUNDED,
    )

    # ── Market Data Table ────────────────────────────────────────────────────
    table = Table(
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        show_footer=False,
        border_style="bright_blue",
        title="[bold cyan]📊 MARKET DATA[/bold cyan]",
        title_style="bold cyan",
    )
    table.add_column("Símbolo", style="bold white", no_wrap=True)
    table.add_column("Precio", justify="right")
    table.add_column("24h %", justify="right")
    table.add_column("Vol (ATR)", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("ML Conf.", justify="right")
    table.add_column("Tendencia", justify="center")
    table.add_column("Posición", justify="center")
    table.add_column("PnL Pos.", justify="right")
    table.add_column("Acción IA", justify="center")

    _trend_color = {"bullish": "bright_green", "bearish": "bright_red", "neutral": "yellow"}

    for sym in watchlist:
        prices = list(state["prices"].get(sym, []))
        price: float | None = prices[-1] if prices else None
        price_str = f"{price:,.2f}" if price is not None else "[dim]N/A[/dim]"

        # 24 h % change – 96 bars × 15 m = 24 h
        change_24h_str = "[dim]—[/dim]"
        if price is not None and len(prices) >= 96:
            price_24h = prices[-96]
            if price_24h > 0:
                pct = (price - price_24h) / price_24h * 100
                ch_color = "bright_green" if pct >= 0 else "bright_red"
                sign = "+" if pct >= 0 else ""
                change_24h_str = f"[{ch_color}]{sign}{pct:.2f}%[/{ch_color}]"

        # Vol (ATR)
        atr_val: float | None = state.get("atrs", {}).get(sym)
        atr_str = f"{atr_val:.2f}" if atr_val is not None else "[dim]—[/dim]"

        # RSI
        rsi_val = _compute_rsi(prices) if len(prices) >= 15 else None
        if rsi_val is not None:
            rsi_color = "bright_red" if rsi_val >= 70 else "bright_green" if rsi_val <= 30 else "white"
            rsi_str = f"[{rsi_color}]{rsi_val:.1f}[/{rsi_color}]"
        else:
            rsi_str = "[dim]—[/dim]"

        # ML probability (kept for Acción IA and ML Conf. column)
        prob = ml_probs.get(sym, 0.0)

        # ML Conf. display
        prob_pct = prob * 100
        if prob_pct > 55:
            ml_conf_str = f"[bold green]{prob_pct:.1f}%[/bold green]"
        elif prob_pct > 50:
            ml_conf_str = f"[bold yellow]{prob_pct:.1f}%[/bold yellow]"
        else:
            ml_conf_str = f"[dim]{prob_pct:.1f}%[/dim]"

        # HTF trend
        htf_trend = state.get("htf_trend", {}).get(sym, {})
        t15 = htf_trend.get("15m", "neutral")
        t1h = htf_trend.get("1h", "neutral")
        t4h = htf_trend.get("4h", "neutral")
        trend_str = (
            f"[{_trend_color.get(t15, 'white')}]{t15[:4]}[/] / "
            f"[{_trend_color.get(t1h, 'white')}]{t1h[:4]}[/] / "
            f"[{_trend_color.get(t4h, 'white')}]{t4h[:4]}[/]"
        )

        # Position & unrealised PnL
        pos = paper_executor.open_positions.get(sym)
        row_style = ""
        if pos and price is not None:
            qty = pos.position_size / pos.entry_price
            unrealized_pnl = (price - pos.entry_price) * qty
            pnl_color = "bright_green" if unrealized_pnl >= 0 else "bright_red"
            pos_str = f"[bright_cyan]LONG @{pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str = f"[{pnl_color}]{unrealized_pnl:+.2f}[/{pnl_color}]"
            row_style = "on dark_green"
        elif pos:
            pos_str = f"[bright_cyan]LONG @{pos.entry_price:,.2f}[/bright_cyan]"
            pnl_str = "[dim]N/A[/dim]"
            row_style = "on dark_green"
        else:
            pos_str = "[dim]—[/dim]"
            pnl_str = "[dim]—[/dim]"

        # Acción IA
        if pos:
            accion_str = "[bright_blue]TRADING 🔵[/bright_blue]"
        elif prob >= BUY_PROB_THRESHOLD:
            accion_str = "[bright_green]BUY! 🟢[/bright_green]"
        else:
            accion_str = "[bright_yellow]HOLD 🟡[/bright_yellow]"

        table.add_row(
            sym.split("/")[0],
            price_str,
            change_24h_str,
            atr_str,
            rsi_str,
            ml_conf_str,
            trend_str,
            pos_str,
            pnl_str,
            accion_str,
            style=row_style,
        )

    # ── Footer: Risk & Wallet + Events Log ──────────────────────────────────
    n_pos = len(paper_executor.open_positions)
    pending_closes = sum(
        1 for p in paper_executor.open_positions.values() if getattr(p, "close_pending", False)
    )
    balance = risk_manager.balance
    total_pnl = paper_executor.total_pnl
    max_drawdown: float = state.get("max_drawdown", 0.0)

    # Approximate used margin (position_size already in USDT notional / LEVERAGE)
    used_margin = sum(
        pos.position_size / _RISK_LEVERAGE for pos in paper_executor.open_positions.values()
    )
    available_margin = balance - used_margin
    margin_color = (
        "bright_green" if available_margin > balance * 0.2
        else "yellow" if available_margin > 0
        else "bright_red"
    )
    pnl_color_r = "bright_green" if total_pnl >= 0 else "bright_red"
    dd_color = "bright_red" if max_drawdown < -(balance * 0.05) else "yellow" if max_drawdown < 0 else "bright_green"

    risk_text = Text()
    risk_text.append("💰 Balance:           ", style="dim")
    risk_text.append(f"{balance:>12,.2f} USDT\n", style="bold white")
    risk_text.append("📊 Margen Disponible: ", style="dim")
    risk_text.append(f"{available_margin:>12,.2f} USDT\n", style=f"bold {margin_color}")
    risk_text.append("📈 PnL Sesión:        ", style="dim")
    risk_text.append(f"{total_pnl:>+12.4f} USDT\n", style=f"bold {pnl_color_r}")
    risk_text.append("📉 Max Drawdown:      ", style="dim")
    risk_text.append(f"{max_drawdown:>+12.4f} USDT\n", style=f"bold {dd_color}")
    risk_text.append("🔄 Posiciones:        ", style="dim")
    risk_text.append(f"{n_pos}/{risk_manager.max_positions}\n", style="bold white")
    risk_text.append("⏳ Cierres pendientes:", style="dim")
    pending_style = "bold bright_red" if pending_closes > 0 else "bold bright_green"
    risk_text.append(f"{pending_closes}", style=pending_style)

    risk_panel = Panel(
        risk_text,
        title="[bold yellow]💼 RIESGO & WALLET[/bold yellow]",
        border_style="yellow",
        box=rich_box.ROUNDED,
    )

    # Events log panel
    events_text = Text()
    events = list(_DASHBOARD_EVENTS)
    if events:
        for evt in events[-3:]:
            events_text.append_text(Text.from_markup(evt + "\n"))
    else:
        events_text.append_text(Text.from_markup("[dim]Sin eventos recientes…[/dim]"))

    events_panel = Panel(
        events_text,
        title="[bold green]📋 ÚLTIMOS EVENTOS[/bold green]",
        border_style="green",
        box=rich_box.ROUNDED,
    )

    # Operations panel (live positions and pending-close status)
    ops_text = Text()
    if paper_executor.open_positions:
        for sym, pos in list(paper_executor.open_positions.items())[:4]:
            px_buf = state.get("prices", {}).get(sym, [])
            mark = float(px_buf[-1]) if px_buf else pos.entry_price
            qty = pos.position_size / pos.entry_price if pos.entry_price > 0 else 0.0
            unrl = (mark - pos.entry_price) * qty
            pnl_color = "bright_green" if unrl >= 0 else "bright_red"
            pending = getattr(pos, "close_pending", False)
            status = "[red]PENDING_CLOSE[/red]" if pending else "[green]OPEN[/green]"
            ops_text.append_text(
                Text.from_markup(
                    f"{sym:<10} {status}  PnL [{pnl_color}]{unrl:+.2f}[/{pnl_color}]\n"
                )
            )
    else:
        ops_text.append_text(Text.from_markup("[dim]Sin posiciones abiertas[/dim]"))

    ops_panel = Panel(
        ops_text,
        title="[bold magenta]🧾 OPERACIONES EN VIVO[/bold magenta]",
        border_style="magenta",
        box=rich_box.ROUNDED,
    )

    footer = Columns([risk_panel, ops_panel, events_panel], equal=True)

    return Group(header_panel, table, footer)


async def dashboard_logger(
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    state: dict[str, Any],
    live: Live,
    interval: int = 1,
) -> None:
    """Refresh the Rich Live dashboard every *interval* seconds.

    On each tick the function:

    * Calls :func:`generate_dashboard` to build a fresh :class:`~rich.table.Table`
      and pushes it to the running :class:`~rich.live.Live` context so the
      terminal display is updated in-place without any scrolling.
    * Writes per-position risk telemetry (trailing stop, ATR, high-watermark)
      exclusively to ``audit.log`` via the ``clawdbot.audit`` logger – this
      output never reaches the console.
    * Emits a DEBUG heartbeat to ``bot_debug.log`` for post-session analysis.

    Important log events (errors, trade opens/closes) continue to reach the
    console via the :class:`~rich.logging.RichHandler` that is attached to the
    root logger in :func:`main`, which renders them above the live table.
    """
    logger_dash = logging.getLogger("clawdbot.dashboard")
    audit = logging.getLogger("clawdbot.audit")
    while True:
        await asyncio.sleep(interval)

        # ── Refresh the live TUI table ────────────────────────────────────────
        live.update(generate_dashboard(state, paper_executor, risk_manager, WATCHLIST))

        # ── Detailed heartbeat → bot_debug.log only (DEBUG) ──────────────────
        n_positions = len(paper_executor.open_positions)
        ml_probs: dict[str, float] = state.get("ml_probs", {})
        logger_dash.debug(
            "📊 [DASHBOARD] Total PnL: %.4f USDT  |  Balance: %.2f USDT  |  "
            "Open positions: %d/%d  |  Symbols: %s",
            paper_executor.total_pnl,
            risk_manager.balance,
            n_positions,
            risk_manager.max_positions,
            list(paper_executor.open_positions.keys()) or "none",
        )

        # ── Audit telemetry: per-position risk data → audit.log only ─────────
        hora = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        for sym, pos in paper_executor.open_positions.items():
            prices_buf = state["prices"].get(sym, [])
            if not prices_buf:
                continue
            mark_price: float = prices_buf[-1]
            if pos.entry_price == 0.0:
                continue
            # Use the live ratcheted stop stored on the position object – this
            # is updated by check_and_close() on every price tick, so it always
            # reflects the current in-memory value rather than a recomputed
            # snapshot that could diverge from what the exit logic actually uses.
            stop_calculated: float = pos.current_stop_loss
            atr_val: float | None = state.get("atrs", {}).get(sym)
            # DISTANCIA_% = ((PRECIO_ACTUAL - STOP_CALCULADO) / PRECIO_ACTUAL) * 100
            distancia_pct = (
                (mark_price - stop_calculated) / mark_price * 100
                if mark_price > 0
                else 0.0
            )
            atr_str = f"{atr_val:.4f}" if atr_val is not None else "N/A"
            xgb_conf: float = ml_probs.get(sym, 0.0)
            audit.info(
                "%s | %-10s | %13.4f | %11.4f | %8s | %14.4f | %8.4f | %11.2f%%",
                hora,
                sym,
                mark_price,
                pos.peak_price,
                atr_str,
                stop_calculated,
                xgb_conf,
                distancia_pct,
            )

            # ── Track max drawdown across all open positions ──────────────────
            qty = pos.position_size / pos.entry_price
            unrealized = (mark_price - pos.entry_price) * qty
            current_dd = state.get("max_drawdown", 0.0)
            if unrealized < current_dd:
                state["max_drawdown"] = unrealized


async def position_sync_loop(
    paper_executor: PaperExecutor,
    interval: int = 60,
) -> None:
    """Periodically reconcile local position state with the live exchange.

    On each tick the coroutine:

    1. Logs ``[SISTEMA] 🔄 Sincronizando posiciones…``
    2. Calls :meth:`~execution.paper_executor.PaperExecutor.sync_positions_with_exchange`
       which removes any *ghost* positions (in memory but gone on the exchange) and
       cancels their orphan orders.
    3. Logs ``[SISTEMA] ✅ Sincronización completa. Posiciones reales: {count}``.

    When the executor is running in pure paper-trading mode (no live exchange
    client and not a live MT5Executor) the coroutine exits immediately so it
    never occupies a task slot needlessly.
    """
    is_live_mt5 = isinstance(paper_executor, MT5Executor) and paper_executor._live
    if paper_executor._exchange is None and not is_live_mt5:
        return  # Nothing to sync in pure paper-trading mode.

    logger = logging.getLogger("clawdbot.sync")
    while True:
        await asyncio.sleep(interval)
        logger.debug("[SISTEMA] 🔄 Sincronizando posiciones con el exchange...")
        try:
            real_count = await paper_executor.sync_positions_with_exchange()
            log_fn = logger.debug if real_count == 0 else logger.info
            log_fn(
                "[SISTEMA] ✅ Sincronización completa. Posiciones reales: %d.",
                real_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "⚠️ [ALERTA] Position sync failed: %s %s",
                exc,
                _DEBUG_LOG_HINT,
            )


async def health_monitor_loop(
    state: dict[str, Any],
    market_queue: asyncio.Queue[dict[str, Any]],
    paper_executor: PaperExecutor,
    interval: int = 30,
) -> None:
    """Periodic health checks for feed freshness and processing backlog."""
    logger = logging.getLogger("clawdbot.health")
    stale_alert_sent = False
    backlog_alert_sent = False
    pending_alert_streak = 0
    while True:
        await asyncio.sleep(interval)
        now = datetime.now(tz=timezone.utc)
        last_msg: datetime | None = state.get("last_market_message_at")
        queue_size = market_queue.qsize()
        open_positions = len(paper_executor.open_positions)
        stale_seconds: float | None = None
        if last_msg is not None:
            stale_seconds = (now - last_msg).total_seconds()

        logger.debug(
            "[HEALTH] queue=%d open_positions=%d stale_seconds=%s",
            queue_size,
            open_positions,
            f"{stale_seconds:.1f}" if stale_seconds is not None else "n/a",
        )

        if queue_size > 2000:
            logger.warning(
                "[HEALTH] market queue backlog is high (%d messages). "
                "Consumer may be lagging.",
                queue_size,
            )
            if not backlog_alert_sent:
                backlog_alert_sent = True
                try:
                    await db.insert_health_event(
                        level="WARNING",
                        component="health_monitor",
                        event_code="QUEUE_BACKLOG_HIGH",
                        message=f"Queue backlog high: {queue_size}",
                        payload_json=json.dumps({"queue_size": queue_size}),
                    )
                except Exception:
                    pass
        else:
            backlog_alert_sent = False
        if stale_seconds is not None and stale_seconds > 120:
            logger.warning(
                "[HEALTH] market feed appears stale (%.1fs without messages).",
                stale_seconds,
            )
            if not stale_alert_sent:
                stale_alert_sent = True
                try:
                    await db.insert_health_event(
                        level="WARNING",
                        component="health_monitor",
                        event_code="MARKET_FEED_STALE",
                        message=f"Market feed stale for {stale_seconds:.1f}s",
                        payload_json=json.dumps({"stale_seconds": stale_seconds}),
                    )
                except Exception:
                    pass
                asyncio.create_task(
                    send_telegram_alert(
                        "⚠️ *HEALTH ALERT* Feed de mercado sin mensajes recientes (>120s)."
                    )
                )
        else:
            stale_alert_sent = False

        pending_count = sum(
            1 for p in paper_executor.open_positions.values() if getattr(p, "close_pending", False)
        )
        if pending_count > 0:
            pending_alert_streak += 1
            if pending_alert_streak >= 3:
                pending_alert_streak = 0
                details = []
                for sym, pos in paper_executor.open_positions.items():
                    if getattr(pos, "close_pending", False):
                        details.append(f"{sym}:{getattr(pos, 'last_close_error', 'unknown')}")
                try:
                    await db.insert_health_event(
                        level="ERROR",
                        component="reconciler",
                        event_code="PENDING_CLOSE_PERSISTENT",
                        message=f"Pending closes persist: {pending_count}",
                        payload_json=json.dumps({"pending": details}),
                    )
                except Exception:
                    pass
                asyncio.create_task(
                    send_telegram_alert(
                        "🚨 *RECONCILER ALERT* Cierres pendientes persistentes: "
                        f"{pending_count}\n" + "\n".join(details[:5])
                    )
                )
        else:
            pending_alert_streak = 0


async def close_pending_reconciler_loop(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    interval: int = 20,
) -> None:
    """Retry loop for positions marked as close_pending."""
    logger = logging.getLogger("clawdbot.reconciler")
    while True:
        await asyncio.sleep(interval)
        pending = [
            sym
            for sym, pos in paper_executor.open_positions.items()
            if getattr(pos, "close_pending", False)
        ]
        if not pending:
            continue
        latest_prices: dict[str, float] = {}
        prices_state: dict[str, deque[float]] = state.get("prices", {})
        for sym in pending:
            buf = prices_state.get(sym, deque())
            if buf:
                latest_prices[sym] = float(buf[-1])
        if not latest_prices:
            logger.debug("[RECONCILER] pending closes exist but no latest prices available.")
            continue
        try:
            closed = await paper_executor.retry_close_pending_positions(latest_prices)
            if closed > 0:
                logger.info(
                    "[RECONCILER] Successfully closed %d pending position(s).",
                    closed,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RECONCILER] retry loop failed: %s", exc)


async def main() -> None:
    logger = setup_logging()

    # ── Ensure logs/ directory exists for trade journal and audit files ────────
    _logs_dir = Path(__file__).parent / "logs"
    _logs_dir.mkdir(exist_ok=True)

    # ── Rich console & Live dashboard setup ───────────────────────────────────
    # Replace the plain StreamHandler with a RichHandler so that any log
    # messages emitted while the Live table is active are rendered above the
    # live area rather than being mixed into the raw terminal stream.
    # Only WARNING+ events (trade alerts, errors) are forwarded to the console;
    # everything else continues to be captured by the rotating file handler
    # (bot_debug.log) and the dedicated audit logger (audit.log).
    _rich_console = Console()
    root_logger = logging.getLogger()
    for _h in list(root_logger.handlers):
        if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
            root_logger.removeHandler(_h)
    _rich_handler = RichHandler(
        console=_rich_console,
        show_path=False,
        rich_tracebacks=False,
        level=logging.WARNING,
    )
    root_logger.addHandler(_rich_handler)

    logger.info("🚀 ClawdBot starting up...")

    # ── Telegram startup notification ─────────────────────────────────────────
    asyncio.create_task(send_telegram_alert("🚀 *ClawdBot* ha iniciado correctamente."))

    # ── Record bot start time for uptime display ──────────────────────────────
    global _BOT_START_TIME
    _BOT_START_TIME = datetime.now(tz=timezone.utc)

    await init_db()

    market_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    shared_state: dict[str, Any] = {
        "sentiment": None if _GEMINI_ENABLED else 0.0,
        "prices": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        # [ELITE] OHLCV buffers for ADX / ATR computation
        "highs": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        "lows": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        # [ATR] Latest ATR_14 value per symbol (updated each signal cycle; None = not yet computed)
        "atrs": {symbol: None for symbol in WATCHLIST},
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
        # [ML] Latest XGBoost win-probability per symbol (0.0 = not yet computed)
        "ml_probs": {symbol: 0.0 for symbol in WATCHLIST},
        # [DASHBOARD] Mega-dashboard telemetry
        "api_latency_ms": 0.0,   # REST/WS round-trip latency in milliseconds
        "max_drawdown": 0.0,     # most negative unrealised PnL seen this session
        "last_market_message_at": None,  # datetime | None
    }

    predictor = MLPredictor()

    # ── Execution mode: MT5-first (Binance removed) ─────────────────────────
    execution_mode: str = os.environ.get("EXECUTION_MODE", "mt5").strip().lower()
    initial_balance: float = 10_000.0
    _mt5_initialized: bool = False

    if execution_mode == "mt5":
        # ── MetaTrader 5 execution path ──────────────────────────────────
        mt5_login_raw: str = os.environ.get("MT5_LOGIN", "").strip()
        mt5_password: str = os.environ.get("MT5_PASSWORD", "").strip()
        mt5_server: str = os.environ.get("MT5_SERVER", "").strip()

        missing = [k for k, v in [
            ("MT5_LOGIN", mt5_login_raw),
            ("MT5_PASSWORD", mt5_password),
            ("MT5_SERVER", mt5_server),
        ] if not v]
        if missing:
            logger.warning(
                "⚠️ [MT5] Missing environment variable(s): %s – "
                "falling back to paper trading with %.2f USDT.",
                ", ".join(missing),
                initial_balance,
            )
            execution_mode = "paper"
        else:
            try:
                mt5_login: int = int(mt5_login_raw)
            except ValueError:
                logger.warning(
                    "⚠️ [MT5] MT5_LOGIN='%s' is not a valid integer – "
                    "falling back to paper trading.",
                    mt5_login_raw,
                )
                execution_mode = "paper"
            else:
                logger.info(
                    "🔌 [MT5] Connecting to MetaTrader 5 | server=%s | login=%d",
                    mt5_server,
                    mt5_login,
                )
                _mt5_initialized = initialize_mt5(
                    account=mt5_login,
                    password=mt5_password,
                    server=mt5_server,
                )
                if not _mt5_initialized:
                    logger.warning(
                        "⚠️ [MT5] initialize_mt5() failed – "
                        "falling back to paper trading with %.2f USDT.",
                        initial_balance,
                    )
                    execution_mode = "paper"
                else:
                    # Fetch the real account balance from the connected MT5 terminal
                    # so that RiskManager is seeded with the actual capital, not the
                    # hardcoded 10 000 USDT default.
                    mt5_balance = fetch_mt5_account_balance()
                    if mt5_balance is not None and mt5_balance > 0.0:
                        initial_balance = mt5_balance
                        logger.info(
                            "✅ [MT5] Account balance fetched: %.2f USDT",
                            initial_balance,
                        )
                    else:
                        logger.warning(
                            "⚠️ [MT5] Could not fetch account balance – "
                            "using default %.2f USDT.",
                            initial_balance,
                        )

    if execution_mode != "mt5":
        logger.info(
            "📝 [PAPER] EXECUTION_MODE=%s (MT5 not selected). "
            "Running in paper mode with %.2f USDT.",
            execution_mode,
            initial_balance,
        )

    risk_manager = RiskManager(initial_balance=initial_balance)

    if execution_mode == "mt5" and _mt5_initialized:
        paper_executor: PaperExecutor = MT5Executor(
            db=db,
            risk_manager=risk_manager,
            live=True,
        )
        logger.info(
            "✅ [MT5] MT5Executor initialised in LIVE mode – "
            "orders will be sent to MetaTrader 5."
        )
        asyncio.create_task(
            send_telegram_alert(
                f"✅ *ClawdBot [MT5 LIVE]* conectado\n"
                f"Servidor: `{mt5_server}` | Login: `{mt5_login}`\n"
                f"Balance: *{initial_balance:,.2f} USDT*"
            )
        )
    else:
        paper_executor = PaperExecutor(db=db, risk_manager=risk_manager, exchange=None)

    mt5_market_client: MT5MarketDataClient | None = None
    if execution_mode == "mt5" and _mt5_initialized and isinstance(paper_executor, MT5Executor):
        mt5_market_client = MT5MarketDataClient(
            queue=market_queue,
            executor=paper_executor,
            watchlist=WATCHLIST,
        )

    logger.info(
        "SESSION_CONFIG session_id=%s execution_mode=%s mt5_initialized=%s "
        "market_feed=%s initial_balance=%.2f gemini_enabled=%s watchlist=%s",
        _LOG_SESSION_ID,
        execution_mode,
        _mt5_initialized,
        mt5_market_client is not None,
        initial_balance,
        _GEMINI_ENABLED,
        WATCHLIST,
    )

    # ------------------------------------------------------------------
    # [SYNC] Re-sync open positions from exchange on restart
    # ------------------------------------------------------------------
    # When the bot restarts it must not assume all positions are closed.
    # Fetch the actual open positions and restore them in the local
    # PaperExecutor/RiskManager state so that:
    #   - `can_open_position()` returns the correct headroom.
    #   - The per-symbol duplicate guard in `try_open_trade()` works.
    # NOTE: the balance was fetched from the exchange above and already
    # reflects the margin tied up in open positions, so no further
    # deduction is made.
    if execution_mode == "mt5" and _mt5_initialized:
        # ------------------------------------------------------------------
        # [MT5] Restore positions: load saved local state first, then
        # reconcile against live MT5 positions to remove any ghosts.
        # ------------------------------------------------------------------
        paper_executor.load_state()
        try:
            real_count = await paper_executor.sync_positions_with_exchange()
            if real_count > 0:
                logger.info(
                    "✅ [MT5 SYNC] Startup reconcile complete – "
                    "%d live position(s) confirmed on MT5 "
                    "(Open positions: %d/%d).",
                    real_count,
                    risk_manager.open_count,
                    risk_manager.max_positions,
                )
            else:
                logger.info(
                    "🔄 [MT5 SYNC] No live MT5 positions at startup – "
                    "starting fresh session."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "⚠️ [MT5 SYNC] Startup position sync failed: %s – "
                "local state.json will be used as-is.",
                exc,
            )
    else:
        # ------------------------------------------------------------------
        # [PAPER] Restore open positions from local state.json (paper trading)
        # ------------------------------------------------------------------
        # In pure paper-trading mode there is no exchange to query.
        # Load the last saved state so that positions survive a bot restart.
        _paper_restored = paper_executor.load_state()
        if _paper_restored == 0:
            logger.info(
                "📝 [PAPER] No previous state.json found – starting fresh session."
            )

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
    # [MT5] Warmup – pre-fill buffers from MT5 candle history
    # ------------------------------------------------------------------
    if execution_mode == "mt5" and _mt5_initialized and isinstance(paper_executor, MT5Executor):
        await preload_historical_data_mt5(shared_state, WATCHLIST, paper_executor)
    else:
        logger.warning(
            "[MT5] Historical preload skipped because MT5 is not active."
        )

    # ------------------------------------------------------------------
    # [AUDIT] Decision pipeline diagnostics – log active thresholds and
    # confirm that session state was cleanly initialised on this startup.
    # ------------------------------------------------------------------
    logger.info(
        "🔍 [AUDIT] Decision pipeline thresholds: "
        "ML_BUY_PROB≥%.2f | SENTIMENT_BUY≥%.2f | "
        "NEWS_FILTER_SWING>%.2f → HOLD_%dmin",
        BUY_PROB_THRESHOLD,
        BUY_SENTIMENT_THRESHOLD,
        _NEWS_FILTER_VOLATILITY_THRESHOLD,
        _NEWS_FILTER_HOLD_MINUTES,
    )
    logger.info(
        "🔍 [AUDIT] Session state reset: "
        "sentiment=%s  news_hold_until=None  max_drawdown=0.0  "
        "trading_halted=%s",
        shared_state.get("sentiment"),
        risk_manager.is_trading_halted(),
    )

    # ── Start Rich Live dashboard ─────────────────────────────────────────────
    # The Live context renders a fixed TUI table that refreshes every second.
    # Important log events (WARNING+) are forwarded via RichHandler and appear
    # above the live area so they are never lost.  Full DEBUG logs continue to
    # be written to bot_debug.log and audit.log as before.
    _live = Live(
        generate_dashboard(shared_state, paper_executor, risk_manager, WATCHLIST),
        console=_rich_console,
        refresh_per_second=1,
        auto_refresh=True,
    )
    _live.start(refresh=True)

    if mt5_market_client is None:
        logger.error("❌ [MT5] Market feed client not available; cannot start bot loop.")
        await close_db()
        return

    run_tasks: list[asyncio.Future[Any] | asyncio.Task[Any] | Any] = [
        mt5_market_client.run(),
        market_consumer(market_queue, shared_state, paper_executor),
        signal_emitter(shared_state, predictor, paper_executor, watchlist=WATCHLIST, interval=15),
        dashboard_logger(paper_executor, risk_manager, shared_state, _live, interval=1),
        weekly_retrainer(predictor, watchlist=WATCHLIST, model_path=_MODEL_PATH),
        position_sync_loop(paper_executor, interval=60),
        health_monitor_loop(shared_state, market_queue, paper_executor, interval=30),
        close_pending_reconciler_loop(shared_state, paper_executor, interval=20),
    ]
    if _GEMINI_ENABLED:
        run_tasks.append(gemini_sentiment_refresher(shared_state))
    else:
        logger.warning(
            "[LLM] Gemini sentiment refresher disabled (GEMINI_API_KEY missing). "
            "Using neutral sentiment baseline."
        )

    try:
        await asyncio.gather(*run_tasks)
    except Exception as _critical_exc:  # noqa: BLE001
        logger.critical(
            "🚨 [CRITICAL] Bot loop terminated unexpectedly: %s – check bot_debug.log.",
            _critical_exc,
            exc_info=True,
        )
        try:
            await send_telegram_alert(
                f"🚨 *ERROR CRÍTICO* – ClawdBot se ha detenido inesperadamente.\n"
                f"Causa: `{type(_critical_exc).__name__}: {str(_critical_exc)[:200]}`"
            )
        except Exception:  # noqa: BLE001
            pass  # never let the Telegram call block the shutdown path
        raise
    finally:
        _live.stop()
        if _mt5_initialized:
            shutdown_mt5()
        await close_db()

    logger.info("🛑 ClawdBot shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())