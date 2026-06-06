"""
ClawdBot – entry point.

Sets up a structured JSON logger and starts the asyncio event loop.
Runs an MT5 market-data poller (ticks + multi-timeframe candles).  OHLC
snapshots are persisted to TimescaleDB.

A dedicated XGBoost model emits BUY / HOLD from probability ≥ ``BUY_PROB_THRESHOLD``
(default 0.50 max-performance profile; ``BUY_PROB_THRESHOLD`` env); no external sentiment gates.

When the signal is BUY the :class:`~risk.risk_manager.RiskManager` sizes the
position using a **fixed fractional** rule (``balance × RISK_PER_TRADE ×
LEVERAGE``, capped per ``max_positions`` — see ``risk/risk_manager.py``), not
the Kelly criterion (that remains a possible future enhancement). The executor
then opens the trade. Up to ``max_positions`` trades may be open at once, one
per symbol.

**Long-only execution:** shorts are not opened. A model **SELL** means “bearish
guidance”; exits for open LONGs are handled by :meth:`~execution.paper_executor.PaperExecutor.check_ml_exit`
(smart reversal / TTL) and by mechanical stops in :meth:`~execution.paper_executor.PaperExecutor.check_and_close`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import utils.env_bootstrap  # noqa: E402 — patches dotenv.load_dotenv (UTF-16 .env)
from utils.env_bootstrap import load_env_file

try:
    load_env_file(_ROOT / ".env")
except Exception:
    pass
logging.getLogger("dotenv.main").setLevel(logging.ERROR)

from rich.box import ASCII
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from bot import state as dash_state
from bot.dashboard import generate_dashboard
from bot.dashboard_helpers import display_timezone
from bot.loops import (
    close_pending_reconciler_loop,
    dashboard_logger,
    health_monitor_loop,
    monthly_report_loop,
    position_sync_loop,
    risk_stats_reset_loop,
    weekly_report_loop,
)
from bot.task_supervisor import TaskSupervisor
from bot.market_consumer import market_consumer
from bot.mt5_preload import preload_historical_data_mt5
from bot.signal_emitter import signal_emitter
from bot.web_server import start_web_dashboard
from bot.dashboard_launcher import start_dashboard_servers
from bot.biweekly_retrainer import biweekly_retrainer
from data_ingestion.mt5_market_client import MT5MarketDataClient
from database.db_manager import close_db, db, init_db
from execution.mt5_executor import (
    MT5Executor,
    SYMBOL_MAP,
    fetch_mt5_account_balance,
    fetch_mt5_wallet_snapshot,
    initialize_mt5,
    shutdown_mt5,
)
from execution.paper_executor import POSITION_TTL_HOURS, PaperExecutor
from risk.risk_manager import BASE_SL, MAX_POSITIONS, RISK_PER_TRADE, RiskManager
from strategy.ml_predictor import (
    BUY_PROB_THRESHOLD,
    MLPredictor,
    load_booster_from_disk,
    model_json_path_for_symbol,
)
from utils.diagnostic_bundle import write_diagnostic_bundle
from utils.runtime_snapshot import runtime_metrics_loop, write_startup_snapshot
from utils.telegram_notifier import (
    install_asyncio_critical_telegram_alerts,
    install_telegram_log_alerts,
    send_priority_telegram_alert,
    send_telegram_alert,
    telegram_command_poller,
)
from vibe_mcp.server import start_mcp_server, SERVER_PORT as VIBE_MCP_PORT

try:
    from vibe.hybrid_client import VibeHybridClient as VibeClient
    from vibe.scheduled_tasks import (
        factor_analysis_loop,
        journal_analysis_loop,
        pattern_detection_loop,
        shadow_account_loop,
        weekly_backtest_loop,
    )
    from vibe.swarm_loops import crypto_desk_swarm_loop

    _VIBE_AVAILABLE = True
except ImportError:
    _VIBE_AVAILABLE = False

# ── ANSI colour helpers (no extra dependency) ─────────────────────────────────
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# Dashboard event buffer and start time live in :mod:`bot.state`.


def _print_operational_deploy(console: Console) -> None:
    """Rich banner: active aggressive-profile parameters (startup audit)."""
    sym = WATCHLIST[0] if WATCHLIST else "ETH/USDT"
    mt5_sym = SYMBOL_MAP.get(sym, "?")
    tbl = Table(title="DESPLIEGUE OPERATIVO AGRESIVO", box=ASCII, show_header=True, header_style="bold")
    tbl.add_column("Parametro")
    tbl.add_column("Valor", justify="right")
    tbl.add_row("ML BUY threshold (prob)", f"{BUY_PROB_THRESHOLD:.2f}")
    tbl.add_row("RISK_PER_TRADE (equity)", f"{RISK_PER_TRADE * 100:.1f}%")
    tbl.add_row("MAX_POSITIONS", str(MAX_POSITIONS))
    tbl.add_row("POSITION_TTL_HOURS", f"{POSITION_TTL_HOURS:.1f}")
    tbl.add_row("BASE_SL (initial)", f"{BASE_SL * 100:.1f}%")
    tbl.add_row("Watchlist -> MT5", f"{sym} -> {mt5_sym}")
    console.print(Panel(tbl, border_style="cyan"))


def _apply_safe_env_defaults() -> None:
    """Set conservative defaults for optional operational ENV knobs."""
    defaults: dict[str, str] = {
        "DIAGNOSTIC_BUNDLE_INTERVAL_S": "1800",
        "TELEGRAM_LOG_ALERTS": "1",
        "TELEGRAM_LOG_MIN_LEVEL": "WARNING",
    }
    for key, value in defaults.items():
        if os.environ.get(key, "").strip():
            continue
        os.environ[key] = value
        print(
            f"[ENV] {key} not set; using default {value}.",
            file=sys.stderr,
        )
    if not os.environ.get("BUY_PROB_THRESHOLD", "").strip():
        print(
            "[ENV] BUY_PROB_THRESHOLD not set; using model default 0.50 "
            "(strategy/ml_predictor.py).",
            file=sys.stderr,
        )


def _check_env() -> None:
    """Warn if ``.env`` is missing (process env still used)."""
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


_check_env()
_apply_safe_env_defaults()

def _watchlist_from_env() -> list[str]:
    raw = os.environ.get("WATCHLIST", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]


# Symbols must exist in MT5 Market Watch (see execution.mt5_executor.SYMBOL_MAP).
WATCHLIST: list[str] = _watchlist_from_env()

_MODEL_PATH = (
    Path(__file__).parent / "models" / f"{WATCHLIST[0].replace('/', '_')}_v1.json"
    if WATCHLIST
    else Path(__file__).parent / "models" / "ETH_USDT_v1.json"
)
_REPO_ROOT = _ROOT

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

    Timestamps use ``REPORT_TIMEZONE`` / ``DASHBOARD_TIMEZONE`` (default Bogota).
    """

    _LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG:    "",
        logging.INFO:     "",
        logging.WARNING:  _YELLOW,
        logging.ERROR:    _RED + _BOLD,
        logging.CRITICAL: _RED + _BOLD,
    }

    def __init__(self, display_tz: ZoneInfo | None = None) -> None:
        super().__init__()
        self._tz = display_tz if display_tz is not None else ZoneInfo("America/Bogota")

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, _RESET)
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .astimezone(self._tz)
            .strftime("%H:%M:%S")
        )
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        # INFO/DEBUG: sin ANSI — en PowerShell sin VT, \033[0m se ve como "←[0m".
        if record.levelno <= logging.INFO:
            return f"{ts} | {msg}"
        return f"{color}{ts} | {msg}{_RESET}"


class _DashboardEventHandler(logging.Handler):
    """Captures INFO+ log messages for the dashboard events panel.

    Appends a short timestamped line to :data:`bot.state.dashboard_events` so
    the mega-dashboard can display
    the last few operational events without interfering with other handlers.
    """

    def __init__(self, display_tz: ZoneInfo | None = None) -> None:
        super().__init__()
        self._tz = display_tz if display_tz is not None else ZoneInfo("America/Bogota")

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

            ts = (
                datetime.fromtimestamp(record.created, tz=timezone.utc)
                .astimezone(self._tz)
                .strftime("%H:%M:%S")
            )
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
    _tz_disp, _ = display_timezone()
    console_handler.setFormatter(_ConsoleFormatter(_tz_disp))
    root.addHandler(console_handler)

    # ── Dashboard event handler: capture INFO+ events for TUI panel ───────────
    dashboard_event_handler = _DashboardEventHandler(_tz_disp)
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

    # Write header only when file is new/empty — avoids duplicate headers on restart.
    if not audit_log_file.exists() or audit_log_file.stat().st_size == 0:
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


async def diagnostic_bundle_refresh_loop(repo_root: Path, interval_s: float) -> None:
    """Escribe ``DIAGNOSTIC_FOR_REVIEW.md`` en la raíz del repo de forma periódica."""
    log = logging.getLogger("clawdbot")
    await asyncio.sleep(45.0)
    while True:
        try:
            path = write_diagnostic_bundle(repo_root=repo_root)
            log.info("Diagnostic bundle actualizado: %s", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Diagnostic bundle falló: %s", exc)
        await asyncio.sleep(interval_s)


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

    if os.environ.get("DASH_QUIET_ACCESS_LOG", "").strip().lower() in ("1", "true", "yes"):
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    install_asyncio_critical_telegram_alerts()
    install_telegram_log_alerts()

    logger.info("🚀 ClawdBot starting up...")
    _print_operational_deploy(_rich_console)

    # ── Telegram startup notification ─────────────────────────────────────────
    startup_alert_task: asyncio.Task[bool] | None = asyncio.create_task(
        send_telegram_alert("🚀 *ClawdBot* ha iniciado correctamente.")
    )

    # ── Record bot start time for uptime display ──────────────────────────────
    dash_state.bot_start_time = datetime.now(tz=timezone.utc)

    await init_db()

    market_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    shared_state: dict[str, Any] = {
        "session_id": _LOG_SESSION_ID,
        "session_started_at": datetime.now(tz=timezone.utc).isoformat(),
        "prices": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        # Homogeneous closes for dashboard RSI (single TF; see DASHBOARD_RSI_TF, default 15m).
        "dashboard_rsi_closes": {symbol: deque(maxlen=500) for symbol in WATCHLIST},
        # [ELITE] OHLCV buffers for ADX / ATR computation
        "highs": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        "lows": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        "volumes": {symbol: deque(maxlen=1000) for symbol in WATCHLIST},
        # [ATR] Latest ATR_14 value per symbol (updated each signal cycle; None = not yet computed)
        "atrs": {symbol: None for symbol in WATCHLIST},
        # [ELITE] Latest Order Book Imbalance ratio per symbol
        "obi_ratios": {symbol: 1.0 for symbol in WATCHLIST},
        # [ELITE] Latest perpetual-futures funding rate per symbol
        "funding_rates": {symbol: 0.0 for symbol in WATCHLIST},
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
        # [ML] Last generate_signal() per symbol (BUY/SELL/HOLD); drives TUI/web vs raw prob
        "ml_signals": {symbol: "HOLD" for symbol in WATCHLIST},
        # [VIBE] Vibe-Trading analysis results (populated by scheduled tasks)
        "vibe_journal_analysis": None,
        "vibe_backtest": {},
        "vibe_patterns": {},
        "vibe_factors": {},
        "vibe_shadow_report": None,
        "vibe_client": None,
        # [DASHBOARD] Mega-dashboard telemetry
        "api_latency_ms": 0.0,   # REST/WS round-trip latency in milliseconds
        "max_drawdown": 0.0,     # most negative unrealised PnL seen this session
        "last_market_message_at": None,  # datetime | None
    }

    predictor = MLPredictor()

    # ── Execution mode: MT5-first (Binance removed) ─────────────────────────
    execution_mode: str = os.environ.get("EXECUTION_MODE", "mt5").strip().lower()
    initial_balance: float = 10_000.0
    _ib_raw = os.environ.get("INITIAL_BALANCE", "").strip()
    if _ib_raw:
        try:
            initial_balance = float(_ib_raw)
            logger.info(
                "[ENV] INITIAL_BALANCE=%.2f (paper / fallback si MT5 no aporta equity)",
                initial_balance,
            )
        except ValueError:
            logger.warning("[ENV] INITIAL_BALANCE inválido - usando 10000.0")
            initial_balance = 10_000.0
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
                "✅ <b>ClawdBot [MT5 LIVE]</b> conectado\n"
                f"Servidor: <code>{escape(mt5_server)}</code> | "
                f"Login: <code>{escape(str(mt5_login))}</code>\n"
                f"Balance: <b>{initial_balance:,.2f} USDT</b>",
                parse_mode="HTML",
            )
        )
        unresolved = paper_executor.validate_symbol_mapping(WATCHLIST)
        if unresolved:
            logger.critical(
                "❌ [MT5] Symbol mapping validation failed: %s",
                unresolved,
            )
            await send_priority_telegram_alert(
                "🚨 *SYMBOL MAP ERROR*\n"
                + "\n".join(f"• `{x}`" for x in unresolved),
                priority="critical",
                dedup_key="startup:symbol_map_invalid",
                force=True,
            )
            if _mt5_initialized:
                shutdown_mt5()
            await close_db()
            return

        # Adopt orphan MT5 positions after reboot
        try:
            recovered = await paper_executor.recover_positions_on_startup()
            if recovered:
                await send_telegram_alert(
                    f"♻️ *Recovery*: {recovered} posición(es) adoptada(s) al reiniciar."
                )
        except Exception as exc:
            logger.warning("[RECOVERY] Startup recovery failed: %s", exc)
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
        "market_feed=%s initial_balance=%.2f watchlist=%s",
        _LOG_SESSION_ID,
        execution_mode,
        _mt5_initialized,
        mt5_market_client is not None,
        initial_balance,
        WATCHLIST,
    )
    if execution_mode == "mt5" and not isinstance(paper_executor, MT5Executor):
        raise RuntimeError("EXECUTION_MODE=mt5 requires MT5Executor only.")
    if execution_mode != "mt5" and isinstance(paper_executor, MT5Executor):
        raise RuntimeError("Paper mode must not run MT5Executor.")

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
            real_count = await paper_executor.sync_positions_with_exchange(confirmations_required=1)
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
            logger.info(
                "[MT5 SYNC] Startup ghost confirmation override=%d (runtime=%d).",
                1,
                getattr(paper_executor, "_ghost_min_confirmations", 1),
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
    for _sym in WATCHLIST:
        _mp = model_json_path_for_symbol(_sym)
        if _mp.is_file():
            try:
                load_booster_from_disk(_mp)
                logger.info("✅ ML booster cached: %s", _mp.name)
            except Exception as _exc:  # noqa: BLE001
                logger.warning("Could not load model for %s (%s): %s", _sym, _mp, _exc)

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
        "[AUDIT] Decision pipeline: ML_BUY_PROB>=%.2f | symbols=%s (ML-only entries).",
        BUY_PROB_THRESHOLD,
        WATCHLIST,
    )
    logger.info(
        "[AUDIT] Session state reset: max_drawdown=0.0  trading_halted=%s",
        risk_manager.is_trading_halted(),
    )
    logger.info(
        "[AUDIT] UI coherence: COMPRAR/BUY only when "
        "ml_signals[symbol]==BUY and prob>=%.2f.",
        BUY_PROB_THRESHOLD,
    )
    logger.warning(
        "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=%.2f risk=%.1f%%.",
        BUY_PROB_THRESHOLD,
        RISK_PER_TRADE * 100.0,
    )

    try:
        snap_path = write_startup_snapshot(
            execution_mode=execution_mode,
            session_id=_LOG_SESSION_ID,
            watchlist=WATCHLIST,
            risk_manager=risk_manager,
            paper_executor=paper_executor,
        )
        logger.info("📄 Startup snapshot: %s", snap_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup runtime snapshot failed: %s", exc)

    try:
        _diag_path = write_diagnostic_bundle(repo_root=_REPO_ROOT)
        logger.info("📎 Primer DIAGNOSTIC_FOR_REVIEW.md → %s", _diag_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diagnostic bundle inicial falló: %s", exc)

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

    # ── Task Supervisor (FASE 1: async robusto) ─────────────────────────────
    supervisor = TaskSupervisor()

    # CRITICAL tasks: si mueren, el bot muere
    supervisor.spawn(
        "mt5_market_client",
        lambda: mt5_market_client.run(),
        critical=True,
        restart=False,
    )
    supervisor.spawn(
        "market_consumer",
        lambda: market_consumer(market_queue, shared_state, paper_executor),
        critical=True,
        restart=False,
    )
    supervisor.spawn(
        "signal_emitter",
        lambda: signal_emitter(shared_state, predictor, paper_executor, watchlist=WATCHLIST, interval=15),
        critical=True,
        restart=False,
    )
    supervisor.spawn(
        "dashboard_logger",
        lambda: dashboard_logger(
            paper_executor,
            risk_manager,
            shared_state,
            _live,
            watchlist=WATCHLIST,
            interval=1,
        ),
        critical=True,
        restart=False,
    )
    supervisor.spawn(
        "position_sync",
        lambda: position_sync_loop(
            paper_executor,
            interval=max(3, min(int(float(os.environ.get("MT5_POSITION_SYNC_S", "4"))), 300)),
        ),
        critical=True,
        restart=False,
    )
    supervisor.spawn(
        "health_monitor",
        lambda: health_monitor_loop(shared_state, market_queue, paper_executor, risk_manager, interval=30),
        critical=True,
        restart=False,
    )
    supervisor.spawn(
        "close_pending_reconciler",
        lambda: close_pending_reconciler_loop(shared_state, paper_executor, interval=20),
        critical=True,
        restart=False,
    )

    # NON-CRITICAL tasks: se reinician automáticamente si crashean
    supervisor.spawn(
        "biweekly_retrainer",
        lambda: biweekly_retrainer(watchlist=WATCHLIST),
        critical=False,
        restart=True,
        restart_delay_s=60.0,
    )
    supervisor.spawn(
        "telegram_command_poller",
        lambda: telegram_command_poller(shared_state, paper_executor, risk_manager, interval=5),
        critical=False,
        restart=True,
    )
    supervisor.spawn(
        "weekly_report",
        lambda: weekly_report_loop(),
        critical=False,
        restart=True,
    )
    supervisor.spawn(
        "monthly_report",
        lambda: monthly_report_loop(),
        critical=False,
        restart=True,
    )
    supervisor.spawn(
        "risk_stats_reset",
        lambda: risk_stats_reset_loop(risk_manager),
        critical=False,
        restart=True,
    )
    supervisor.spawn(
        "web_dashboard",
        lambda: start_web_dashboard(shared_state, paper_executor, risk_manager, WATCHLIST, port=8080),
        critical=False,
        restart=True,
        restart_delay_s=10.0,
    )
    supervisor.spawn(
        "dashboard_servers",
        start_dashboard_servers,
        critical=False,
        restart=False,
    )

    # ── Vibe-Trading hybrid client integration (optional) ───────────────────
    # Start the internal HTTP server first so the hybrid client can connect
    if _VIBE_AVAILABLE and os.environ.get("VIBE_TRADING_ENABLED", "1").strip() in ("1", "true", "yes"):
        mcp_server_task = asyncio.create_task(start_mcp_server(port=VIBE_MCP_PORT))
        await asyncio.sleep(2)  # Give the server time to bind

    vibe_client: VibeClient | None = None
    if _VIBE_AVAILABLE and os.environ.get("VIBE_TRADING_ENABLED", "1").strip() in ("1", "true", "yes"):
        vibe_client = VibeClient()
        try:
            vibe_started = await vibe_client.start()
        except Exception as exc:
            logger.warning("Vibe-Trading start raised exception (%s) - tools disabled.", exc)
            vibe_started = False
        if vibe_started:
            shared_state["vibe_client"] = vibe_client
            supervisor.spawn(
                "vibe_journal",
                lambda: journal_analysis_loop(vibe_client, shared_state),
                critical=False,
                restart=True,
            )
            supervisor.spawn(
                "vibe_backtest",
                lambda: weekly_backtest_loop(vibe_client, shared_state, WATCHLIST),
                critical=False,
                restart=True,
            )
            supervisor.spawn(
                "vibe_patterns",
                lambda: pattern_detection_loop(vibe_client, shared_state, WATCHLIST),
                critical=False,
                restart=True,
            )
            supervisor.spawn(
                "vibe_factors",
                lambda: factor_analysis_loop(vibe_client, shared_state, WATCHLIST),
                critical=False,
                restart=True,
            )
            supervisor.spawn(
                "vibe_shadow",
                lambda: shadow_account_loop(vibe_client, shared_state),
                critical=False,
                restart=True,
            )
            if os.environ.get("VIBE_SWARM_ENABLED", "0").strip() in ("1", "true", "yes"):
                supervisor.spawn(
                    "vibe_swarm",
                    lambda: crypto_desk_swarm_loop(vibe_client, shared_state, WATCHLIST),
                    critical=False,
                    restart=True,
                )
                logger.info("Vibe-Trading active - 5 scheduled tasks + SWARM (supervised).")
            else:
                logger.info("Vibe-Trading active - 5 scheduled tasks (swarm disabled, supervised).")
        else:
            err = vibe_client.last_error or "unknown"
            logger.warning("Vibe-Trading start failed (%s) - tools disabled. Bot runs normally.", err)
    else:
        logger.info("ℹ️ Vibe-Trading integration disabled.")

    # Default 120s recurring snapshot → logs/runtime_metrics.jsonl (set to 0 to disable).
    _rmi = float(os.environ.get("RUNTIME_METRICS_INTERVAL_S", "120").strip() or "0")
    if _rmi >= 10.0:
        supervisor.spawn(
            "runtime_metrics",
            lambda: runtime_metrics_loop(
                _LOG_SESSION_ID,
                execution_mode,
                risk_manager,
                paper_executor,
                WATCHLIST,
                _rmi,
            ),
            critical=False,
            restart=True,
        )
        logger.info(
            "📊 Runtime metrics JSONL every %.0fs → logs/runtime_metrics.jsonl",
            _rmi,
        )
    elif _rmi > 0:
        logger.warning(
            "RUNTIME_METRICS_INTERVAL_S=%.1f < 10s - metrics JSONL disabled (use >=10 or 0).",
            _rmi,
        )
    else:
        logger.info(
            "📊 Runtime metrics JSONL disabled (RUNTIME_METRICS_INTERVAL_S=0). "
            "Startup snapshot still written once → logs/bot_startup_snapshot.json",
        )
    _bundle_iv = float(os.environ.get("DIAGNOSTIC_BUNDLE_INTERVAL_S", "1800").strip() or "0")
    if _bundle_iv >= 60.0:
        supervisor.spawn(
            "diagnostic_bundle",
            lambda: diagnostic_bundle_refresh_loop(_REPO_ROOT, _bundle_iv),
            critical=False,
            restart=True,
        )
        logger.info(
            "📎 Cada %.0fs → %s (un solo archivo para pegar al asistente)",
            _bundle_iv,
            _REPO_ROOT / "DIAGNOSTIC_FOR_REVIEW.md",
        )
    elif _bundle_iv > 0:
        logger.warning(
            "DIAGNOSTIC_BUNDLE_INTERVAL_S=%.1f < 60s - bundle automático desactivado.",
            _bundle_iv,
        )

    # ── Start supervised loop ───────────────────────────────────────────────
    try:
        await supervisor.run_forever()
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
        if isinstance(paper_executor, MT5Executor):
            paper_executor.begin_shutdown()
        if vibe_client:
            await vibe_client.stop()
        await supervisor.shutdown()
        if startup_alert_task and not startup_alert_task.done():
            startup_alert_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup_alert_task
        
        from utils.telegram_notifier import _pending_tasks
        if _pending_tasks:
            for t in list(_pending_tasks):
                if not t.done():
                    t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*_pending_tasks, return_exceptions=True)
                
        _live.stop()
        await close_db()
        if _mt5_initialized:
            shutdown_mt5()

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