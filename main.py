"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs an MT5 market-data poller (ticks + multi-timeframe candles) and a
Gemini-powered sentiment refresher concurrently; each incoming trade is
logged together with the latest sentiment score.  Every synthetic top-of-book
snapshot and scored headline is persisted to TimescaleDB.

An ML predictor (XGBoost) is warm-started from historical market data at
startup and emits a BUY / SELL / HOLD signal for each symbol independently.

When the signal is BUY the :class:`~risk.risk_manager.RiskManager` sizes the
position using a **fixed fractional** rule (``balance × RISK_PER_TRADE ×
LEVERAGE``, capped per ``max_positions`` — see ``risk/risk_manager.py``), not
the Kelly criterion (that remains a possible future enhancement). The executor
then opens the trade. Up to ``max_positions`` trades may be open at once, one
per symbol.

**Long-only execution:** shorts are not opened. A model **SELL** means “bearish
guidance”; exits for open LONGs are handled by :meth:`~execution.paper_executor.PaperExecutor.check_ml_exit`
(smart reversal / TTL) and by mechanical stops in :meth:`~execution.paper_executor.PaperExecutor.check_and_close`.

Before first live run, execute ``preflight.py`` (same venv as ``main.py``) to
verify PostgreSQL/TimescaleDB, MT5 login, and Telegram.
"""

from __future__ import annotations

import asyncio
import contextlib
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
from rich.console import Console
from rich.live import Live
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
    fetch_mt5_wallet_snapshot,
    initialize_mt5,
    shutdown_mt5,
)
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD, BUY_SENTIMENT_THRESHOLD, MLPredictor
from strategy.sentiment_llm import get_gemini_sentiment
from utils.telegram_notifier import (
    install_asyncio_critical_telegram_alerts,
    send_priority_telegram_alert,
    send_telegram_alert,
    telegram_command_poller,
)

from bot import state as dash_state
from bot.constants import (
    NEWS_FILTER_HOLD_MINUTES as _NEWS_FILTER_HOLD_MINUTES,
    NEWS_FILTER_VOLATILITY_THRESHOLD as _NEWS_FILTER_VOLATILITY_THRESHOLD,
)
from bot.dashboard import generate_dashboard
from bot.web_server import start_web_dashboard
from bot.loops import (
    close_pending_reconciler_loop,
    dashboard_logger,
    health_monitor_loop,
    monthly_report_loop,
    position_sync_loop,
    weekly_report_loop,
)
from bot.market_consumer import market_consumer
from bot.mt5_preload import preload_historical_data_mt5
from bot.signal_emitter import signal_emitter
from bot.weekly_retrainer import weekly_retrainer

load_dotenv()

# ── ANSI colour helpers (no extra dependency) ─────────────────────────────────
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# Dashboard event buffer and start time live in :mod:`bot.state`.
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

# News filter constants: :mod:`bot.constants` (imported as _NEWS_FILTER_* for audit logs).

_MODEL_PATH = Path(__file__).parent / "models" / "xgb_live.json"

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

    Appends a short timestamped line to :data:`bot.state.dashboard_events` so
    the mega-dashboard can display
    the last few operational events without interfering with other handlers.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            
            # Excluir actualizaciones periódicas que generan ruido visual en el dashboard
            # según lo solicitado: "solo elimina la actualziacion que hace cuando esta mostrando el servidor web"
            if any(k in msg for k in ("Sincronizando posiciones", "Sincronización completa", "[MT5 FEED]", "[LLM]")):
                if record.levelno < logging.WARNING:
                    return
            
            # Excluir logs de acceso al servidor web (cada fetch de la API)
            if record.name == "aiohttp.access" or "GET /api/state" in msg:
                return

            ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
            if len(msg) > 78:
                msg = msg[:75] + "..."
            level_markup = {
                logging.WARNING:  "[yellow]⚠[/yellow]",
                logging.ERROR:    "[red]✖[/red]",
                logging.CRITICAL: "[bold red]‼[/bold red]",
            }.get(record.levelno, "[dim]•[/dim]")
            dash_state.dashboard_events.append(f"[dim]{ts}[/dim] {level_markup} {msg}")
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
    interval: int = 900,
) -> None:
    """[LLM] Background task – refreshes the sentiment score every 15 minutes.

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

    install_asyncio_critical_telegram_alerts()

    logger.info("🚀 ClawdBot starting up...")

    # ── Telegram startup notification ─────────────────────────────────────────
    startup_alert_task: asyncio.Task[bool] | None = asyncio.create_task(
        send_telegram_alert("🚀 *ClawdBot* ha iniciado correctamente.")
    )

    # ── Record bot start time for uptime display ──────────────────────────────
    dash_state.bot_start_time = datetime.now(tz=timezone.utc)

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
        # Live bid/ask/mid from MT5 ticks (dashboard + mark PnL); not the ML candle buffer.
        "mt5_last_quote": {},
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
        _tick_iv = float(os.environ.get("MT5_TICK_INTERVAL_S", "0.25"))
        _tick_iv = max(0.05, min(_tick_iv, 5.0))
        _kline_iv = float(os.environ.get("MT5_KLINE_POLL_S", "5.0"))
        _kline_iv = max(1.0, min(_kline_iv, 120.0))
        mt5_market_client = MT5MarketDataClient(
            queue=market_queue,
            executor=paper_executor,
            watchlist=WATCHLIST,
            shared_state=shared_state,
            tick_interval_s=_tick_iv,
            kline_interval_s=_kline_iv,
        )
        logger.info(
            "[MT5 FEED] tick_interval=%.2fs  kline_poll=%.1fs",
            _tick_iv,
            _kline_iv,
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
    if isinstance(paper_executor, MT5Executor) and paper_executor._live:
        _w0 = fetch_mt5_wallet_snapshot()
        if _w0:
            shared_state["mt5_wallet"] = _w0

    _live = Live(
        generate_dashboard(shared_state, paper_executor, risk_manager, WATCHLIST),
        console=_rich_console,
        refresh_per_second=1,
        auto_refresh=True,
    )
    _live.start(refresh=True)

    if mt5_market_client is None:
        logger.error("❌ [MT5] Market feed client not available; cannot start bot loop.")
        try:
            await send_priority_telegram_alert(
                "🚨 *NO ARRANCA EL BOT* — sin *market feed* MT5.\n"
                "Revisa terminal MT5, credenciales y `EXECUTION_MODE`.",
                priority="critical",
                dedup_key="startup:no_market_feed",
                force=True,
            )
        except Exception:  # noqa: BLE001
            pass
        await close_db()
        return

    run_tasks: list[asyncio.Future[Any] | asyncio.Task[Any] | Any] = [
        mt5_market_client.run(),
        market_consumer(market_queue, shared_state, paper_executor),
        signal_emitter(shared_state, predictor, paper_executor, watchlist=WATCHLIST, interval=15),
        dashboard_logger(
            paper_executor,
            risk_manager,
            shared_state,
            _live,
            watchlist=WATCHLIST,
            interval=1,
        ),
        weekly_retrainer(predictor, watchlist=WATCHLIST, model_path=_MODEL_PATH),
        position_sync_loop(
            paper_executor,
            interval=max(5, min(int(float(os.environ.get("MT5_POSITION_SYNC_S", "20"))), 300)),
        ),
        health_monitor_loop(shared_state, market_queue, paper_executor, interval=30),
        close_pending_reconciler_loop(shared_state, paper_executor, interval=20),
        telegram_command_poller(shared_state, paper_executor, risk_manager, interval=5),
        weekly_report_loop(),
        monthly_report_loop(),
        start_web_dashboard(shared_state, paper_executor, risk_manager, WATCHLIST, port=8080),
    ]
    if _GEMINI_ENABLED:
        run_tasks.append(gemini_sentiment_refresher(shared_state))
    else:
        logger.warning(
            "[LLM] Gemini sentiment refresher disabled (GEMINI_API_KEY missing). "
            "Using neutral sentiment baseline."
        )
    running_tasks: list[asyncio.Task[Any]] = [asyncio.create_task(task) for task in run_tasks]

    try:
        await asyncio.gather(*running_tasks)
    except Exception as _critical_exc:  # noqa: BLE001
        logger.critical(
            "🚨 [CRITICAL] Bot loop terminated unexpectedly: %s – check bot_debug.log.",
            _critical_exc,
            exc_info=True,
        )
        try:
            await send_priority_telegram_alert(
                f"🚨 *ERROR CRÍTICO* – loop principal parado.\n"
                f"Causa: `{type(_critical_exc).__name__}: {str(_critical_exc)[:200]}`",
                priority="critical",
                dedup_key=f"gather_fail:{type(_critical_exc).__name__}",
                force=True,
            )
        except Exception:  # noqa: BLE001
            pass  # never let the Telegram call block the shutdown path
        raise
    finally:
        for task in running_tasks:
            if not task.done():
                task.cancel()
        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)
        if startup_alert_task and not startup_alert_task.done():
            startup_alert_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup_alert_task
        _live.stop()
        if _mt5_initialized:
            shutdown_mt5()
        await close_db()

    logger.info("🛑 ClawdBot shut down cleanly")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except BaseException as _fatal_exc:
        if not isinstance(_fatal_exc, (KeyboardInterrupt, SystemExit)):
            try:
                asyncio.run(
                    send_priority_telegram_alert(
                        f"🚨 *PROCESO CAÍDO*\n"
                        f"`{type(_fatal_exc).__name__}`: `{str(_fatal_exc)[:300]}`",
                        priority="critical",
                        dedup_key=f"fatal:{type(_fatal_exc).__name__}",
                        force=True,
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        raise