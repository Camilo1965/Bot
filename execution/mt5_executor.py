"""
execution.mt5_executor
~~~~~~~~~~~~~~~~~~~~~~

MetaTrader 5 execution layer for ClawdBot.

Provides:

* :func:`initialize_mt5` / :func:`shutdown_mt5` – connect to a local MT5
  terminal and log in to a trading account.
* :func:`calculate_lot_size` – convert a fixed risk-percentage of the account
  balance into a standard MT5 lot size for a given crypto CFD symbol.
* :class:`MT5Executor` – drop-in replacement for the Binance-backed section of
  :class:`~execution.paper_executor.PaperExecutor`.  It inherits all paper-book
  keeping logic (stop-loss tracking, trailing stop, journal CSV, database
  persistence) but replaces every ``ccxt`` call with the corresponding
  ``MetaTrader5`` API call.

Trailing-stop and spread handling
----------------------------------
MT5 crypto CFD instruments carry a dynamic bid/ask spread.  A naïve 1.5 %
trailing distance measured from the mid-price would occasionally be breached by
normal spread widening alone, triggering a premature exit.

:class:`MT5Executor` guards against this by inflating the effective trailing
distance by the current spread before evaluating the stop level::

    effective_trailing_distance = trailing_distance_pct + (spread / entry_price)

The spread is fetched once per price evaluation cycle via
``mt5.symbol_info_tick(symbol).ask - mt5.symbol_info_tick(symbol).bid``.
The bid price is also used when checking exit conditions (because the
broker will fill a SELL order at the bid), which keeps the logic consistent
with real MT5 execution.

Usage example::

    import MetaTrader5 as mt5
    from execution.mt5_executor import initialize_mt5, shutdown_mt5, MT5Executor

    ok = initialize_mt5(
        account=123456,
        password="my_password",
        server="PropFirmXYZ-Demo",
    )
    if not ok:
        raise RuntimeError("MT5 initialization failed")

    executor = MT5Executor(db=db, risk_manager=rm, symbol="BTCUSD")
    opened   = await executor.try_open_trade(entry_price=42000.0, win_probability=0.72)
    ...
    shutdown_mt5()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import TYPE_CHECKING, Any

try:
    import pandas as pd  # type: ignore[import-untyped]

    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

from execution.journal_symbols import journal_symbol
from risk import risk_manager as _risk_cap_mod
from execution.paper_executor import (
    _TAKER_FEE_RATE,
    ATR_SL_MULTIPLIER,
    ATR_TRAILING_MULTIPLIER,
    OpenPosition,
    PaperExecutor,
    _build_trade_report,
    compute_dynamic_tp_hint,
    record_trade,
)
from risk.risk_manager import (
    LEVERAGE,
    RISK_PER_TRADE,
    DynamicThresholds,
    get_execution_thresholds,
    get_sector,
)
from strategy.ml_predictor import get_symbol_config
from utils.telegram_notifier import send_telegram_alert

_SL_CAP_VALUE: float = float(getattr(_risk_cap_mod, "_SL_CAP", 0.05))

if TYPE_CHECKING:
    from database.db_manager import DatabaseManager
    from risk.risk_manager import RiskManager

try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]

    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover – only available on Windows with MT5 installed
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Default risk per trade used by calculate_lot_size ─────────────────────────
# Mirrors :data:`~risk.risk_manager.RISK_PER_TRADE`. Override by passing ``risk_pct``.
_DEFAULT_RISK_PCT: float = RISK_PER_TRADE

# Small buffer added to the trailing distance to absorb spread noise (fraction
# of price, not percentage points).  A dedicated per-tick spread measurement
# supersedes this when MT5 is available.
_SPREAD_BUFFER_FALLBACK: float = 0.001  # 0.10 % fallback when spread is unknown

# Minimum recognised lot size when the broker reports 0.0 (safety guard).
_FALLBACK_VOLUME_MIN: float = 0.01
_FALLBACK_VOLUME_STEP: float = 0.01

# ── MT5 timeframe constants ────────────────────────────────────────────────────
# Mirrors the values exported by the MetaTrader5 library so callers can import
# them without the library being installed (e.g. in tests or non-Windows envs).
TIMEFRAME_M1:  int = 1
TIMEFRAME_M5:  int = 5
TIMEFRAME_M15: int = 15
TIMEFRAME_M30: int = 30
TIMEFRAME_H1:  int = 16385
TIMEFRAME_H4:  int = 16388
TIMEFRAME_D1:  int = 16408

# ── Symbol map ────────────────────────────────────────────────────────────────
# Maps internal/Binance symbol names to the exact broker symbol name required
# by the Admirals MT5 server.  Add entries here whenever you trade a new asset.
SYMBOL_MAP: dict[str, str] = {
    # Binance/CCXT format (entries from WATCHLIST in main.py)
    "BTC/USDT":    "BTCUSD-T",
    "ETH/USDT":    "ETHUSD-T",
    "SOL/USDT":    "SOLUSD-T",
    "XRP/USDT":    "XRPUSD-T",
    "ADA/USDT":    "ADAUSD-T",
    "AVAX/USDT":   "AVAXUSD-T",
    "DOT/USDT":    "DOTUSD-T",
    "BCH/USDT":    "BCHUSD-T",
    "BNB/USDT":    "BNBUSD-T",
    "LINK/USDT":   "LINKUSD-T",
    "INJ/USDT":    "INJUSD-T",
    "FET/USDT":    "FETUSD-T",
    "RENDER/USDT": "RENDERUSD-T",
    "DOGE/USDT":   "DGEUSD-T",
    "PEPE/USDT":   "PEPEUSD-T",
    "PAXG/USDT":   "XAUUSD-T",
    # Raw MT5 base names (pass-through for code that already normalises)
    "BTCUSD":      "BTCUSD-T",
    "ETHUSD":      "ETHUSD-T",
    "SOLUSD":      "SOLUSD-T",
    "BNBUSD":      "BNBUSD-T",
    "LINKUSD":     "LINKUSD-T",
    "INJUSD":      "INJUSD-T",
    "FETUSD":      "FETUSD-T",
    "RENDERUSD":   "RENDERUSD-T",
    "DOGEUSD":     "DGEUSD-T",
    "PEPEUSD":     "PEPEUSD-T",
}


def _apply_mt5_symbol_env_overrides() -> None:
    """Merge ``MT5_SYMBOL_OVERRIDES`` JSON object into :data:`SYMBOL_MAP`.

    Example::

        MT5_SYMBOL_OVERRIDES={"SOL/USDT":"SOLUSDT","SOLUSD":"SOLUSDT"}

    Use exact names from *Market Watch* in MT5 if defaults (e.g. ``SOLUSD-T``)
    do not exist on your broker.
    """
    raw = os.environ.get("MT5_SYMBOL_OVERRIDES", "").strip()
    if not raw:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MT5_SYMBOL_OVERRIDES is not valid JSON — ignoring.")
        return
    if not isinstance(data, dict):
        logger.warning("MT5_SYMBOL_OVERRIDES must be a JSON object — ignoring.")
        return
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            vv = v.strip()
            SYMBOL_MAP[k] = vv
            logger.info("MT5 symbol override: %s → %s", k, vv)


_apply_mt5_symbol_env_overrides()

# ── MT5 return-code catalogue ─────────────────────────────────────────────────
# Human-readable descriptions for every known MT5 trade retcode.  Used by
# _log_mt5_retcode() to produce actionable log messages.
MT5_RETCODE_DESCRIPTIONS: dict[int, str] = {
    10004: "REQUOTE – price changed between request and execution; retry",
    10006: "REQUEST_REJECTED – request rejected by the broker",
    10007: "REQUEST_CANCEL – request was cancelled by the client",
    10008: "REQUEST_PLACED – order placed but not yet executed",
    10009: "TRADE_RETCODE_DONE – request completed successfully",
    10010: "DONE_PARTIAL – request partially executed; retry remainder",
    10011: "REQUEST_ERROR – processing error; retry",
    10012: "REQUEST_TIMEOUT – request timed out; retry",
    10013: "INVALID_REQUEST – malformed order request",
    10014: "INVALID_VOLUME – lot size is outside broker limits",
    10015: "INVALID_PRICE – price is out of range or stale",
    10016: "INVALID_STOPS – SL/TP price is invalid (check digits/distance)",
    10017: "TRADE_DISABLED – trading is disabled for this symbol",
    10018: "MARKET_CLOSED – market is closed for this symbol",
    10019: "NO_MONEY – insufficient margin to open the position",
    10020: "PRICE_CHANGED – price changed; retry with updated price",
    10021: "PRICE_OFF – no quotes available; retry shortly",
    10022: "INVALID_EXPIRATION – order expiration time is invalid",
    10023: "ORDER_CHANGED – order state changed during processing",
    10024: "TOO_MANY_REQUESTS – request flood limit hit; retry after pause",
    10025: "NO_CHANGES – modification request contains no actual changes",
    10026: "SERVER_DISABLES_AT – algo-trading disabled on the server side",
    10027: "CLIENT_DISABLES_AT – 'Algo Trading' button is OFF in MT5 terminal",
    10028: "LOCKED – order or position is locked by the server",
    10029: "FROZEN – order or position is frozen (e.g. during market auction)",
    10030: "INVALID_FILL – order filling type is not supported",
    10031: "CONNECTION – no connection to the trade server; retry",
    10032: "ONLY_REAL – operation only allowed on real accounts",
    10033: "LIMIT_ORDERS – maximum number of pending orders reached",
    10034: "LIMIT_VOLUME – volume limit per symbol/direction reached",
    10035: "INVALID_ORDER – unknown or invalid order type",
    10036: "POSITION_CLOSED – position is already closed",
    10038: "INVALID_CLOSE_VOLUME – close volume exceeds open volume",
    10039: "CLOSE_ORDER_EXIST – a close order for this position already exists",
    10040: "LIMIT_POSITIONS – maximum number of open positions reached",
    10041: "REJECT_CANCEL – pending order activation failed; order was cancelled",
    10042: "LONG_ONLY – only long (BUY) positions are allowed",
    10043: "SHORT_ONLY – only short (SELL) positions are allowed",
    10044: "CLOSE_ONLY – only close operations are allowed at this time",
    10045: "FIFO_CLOSE – positions must be closed in FIFO order",
}

# Return codes that represent transient conditions worth retrying.
_RETRYABLE_RETCODES: frozenset[int] = frozenset({
    10004,  # REQUOTE
    10010,  # DONE_PARTIAL
    10011,  # REQUEST_ERROR
    10012,  # REQUEST_TIMEOUT
    10020,  # PRICE_CHANGED
    10024,  # TOO_MANY_REQUESTS
    10031,  # CONNECTION
})

# Expected pauses (weekend / session) — must not trip the circuit breaker.
_MT5_SOFT_SESSION_RETCODES: frozenset[int] = frozenset({
    10018,  # MARKET_CLOSED
    10021,  # PRICE_OFF
})


def _log_mt5_retcode(retcode: int, context: str = "") -> None:
    """Log a human-readable description for an MT5 trade return code.

    Successful retcodes (10009) are logged at INFO; retryable ones at WARNING;
    all others at ERROR so they stand out in the bot's log output.
    """
    description = MT5_RETCODE_DESCRIPTIONS.get(retcode, f"Unknown retcode {retcode}")
    prefix = f"[MT5 RETCODE {retcode}]"
    ctx = f" ({context})" if context else ""
    if retcode == 10009:
        logger.info("%s %s%s", prefix, description, ctx)
    elif retcode in _MT5_SOFT_SESSION_RETCODES:
        logger.warning("%s %s%s", prefix, description, ctx)
    elif retcode in _RETRYABLE_RETCODES:
        logger.warning("%s %s%s", prefix, description, ctx)
    else:
        logger.error("%s %s%s", prefix, description, ctx)


# ── Connection helpers ─────────────────────────────────────────────────────────


def initialize_mt5(
    account: int,
    password: str,
    server: str,
    path: str | None = None,
) -> bool:
    """Initialize the MetaTrader5 library and log in to a trading account.

    Must be called once before any :class:`MT5Executor` methods that interact
    with the MT5 terminal.  On Windows the library communicates with a locally
    running MetaTrader 5 terminal process; on other operating systems it is not
    supported and this function returns ``False`` immediately.

    Parameters
    ----------
    account:
        MT5 account number (integer login).
    password:
        Account password.
    server:
        Broker server name exactly as shown in the MT5 terminal
        (e.g. ``"ICMarkets-Demo02"``).
    path:
        Optional path to the ``terminal64.exe`` executable.  When *None*
        the library attempts to locate the running terminal automatically.

    Returns
    -------
    bool
        ``True`` on success, ``False`` otherwise.  Failure details are
        available via ``mt5.last_error()``.
    """
    if not _MT5_AVAILABLE:
        logger.error(
            "MetaTrader5 library is not installed.  "
            "Install it with: pip install MetaTrader5"
        )
        return False

    init_kwargs: dict = {"login": account, "password": password, "server": server}
    if path is not None:
        init_kwargs["path"] = path

    if not mt5.initialize(**init_kwargs):
        error = mt5.last_error()
        logger.error("mt5.initialize() failed: %s", error)
        return False

    account_info = mt5.account_info()
    if account_info is None:
        logger.error(
            "MT5 initialized but account_info() returned None – "
            "check account number / password / server."
        )
        mt5.shutdown()
        return False

    logger.info(
        "[MT5] Connected to %s | Account: %d | Balance: %.2f %s",
        server,
        account_info.login,
        account_info.balance,
        account_info.currency,
    )
    return True


def shutdown_mt5() -> None:
    """Disconnect from the MetaTrader 5 terminal gracefully."""
    if _MT5_AVAILABLE:
        mt5.shutdown()
        logger.info("[MT5] Connection closed.")


def fetch_mt5_account_balance() -> float | None:
    """Return the current balance of the connected MT5 account.

    Must be called after :func:`initialize_mt5` has returned ``True``.
    Uses ``mt5.account_info().balance`` (the cash balance, not equity)
    so that the value is consistent with what Binance returns as the
    *totalWalletBalance* field in its account-info endpoint.

    Returns
    -------
    float | None
        Account balance in account currency, or ``None`` when MT5 is
        unavailable or ``mt5.account_info()`` returns ``None``.
    """
    if not _MT5_AVAILABLE:
        return None
    acct = mt5.account_info()
    if acct is None:
        logger.warning("[MT5] account_info() returned None – cannot fetch balance.")
        return None
    logger.debug(
        "[MT5] Account balance=%.2f  equity=%.2f  currency=%s",
        acct.balance,
        acct.equity,
        acct.currency,
    )
    return acct.balance


def fetch_mt5_wallet_snapshot() -> dict[str, float | str] | None:
    """Return live account figures from the connected MT5 terminal.

    Used by the Rich dashboard to show the same balance / equity / margin
    fields as the MT5 terminal status bar.  Must be called after a successful
    :func:`initialize_mt5` (or equivalent ``mt5.login``) in the same process.

    Returns
    -------
    dict[str, float | str] | None
        Keys: ``balance``, ``equity``, ``margin`` (used), ``margin_free``,
        and ``currency`` (account currency string).
        ``None`` when the MetaTrader5 module is unavailable or
        ``mt5.account_info()`` returns ``None``.
    """
    if not _MT5_AVAILABLE:
        return None
    acct = mt5.account_info()
    if acct is None:
        return None
    cur = getattr(acct, "currency", "") or "USD"
    return {
        "balance": float(acct.balance),
        "equity": float(acct.equity),
        "margin": float(acct.margin),
        "margin_free": float(acct.margin_free),
        "currency": cur,
    }


# ── Lot size calculation ───────────────────────────────────────────────────────


def calculate_lot_size(
    symbol: str,
    account_balance: float,
    sl_distance_price: float,
    risk_pct: float = _DEFAULT_RISK_PCT,
) -> float:
    """Calculate the MT5 lot size for a crypto CFD based on a fixed risk amount.

    The formula ensures that a stop-loss placed ``sl_distance_price`` away from
    the entry would cost exactly ``account_balance × risk_pct`` in account
    currency::

        risk_amount   = account_balance × risk_pct
        lots_raw      = risk_amount / (sl_distance_price × contract_size)
        lots_stepped  = floor(lots_raw / volume_step) × volume_step
        lots          = clamp(lots_stepped, volume_min, volume_max)

    For example, on a 10,000 USD account with a 1 % risk and a 500 USD SL
    distance on BTCUSD (contract_size = 1 BTC):

        risk_amount = 100 USD
        lots        = 100 / (500 × 1) = 0.20 lots

    Parameters
    ----------
    symbol:
        MT5 symbol name (e.g. ``"BTCUSD"``).
    account_balance:
        Current account balance or equity in account currency.
    sl_distance_price:
        Absolute price distance between entry and stop-loss
        (``entry_price - stop_loss_price`` for a LONG).
    risk_pct:
        Fraction of balance to risk per trade (default: ``0.01`` = 1 %).

    Returns
    -------
    float
        Normalised lot size ready to pass to ``mt5.order_send``.
        Returns ``_FALLBACK_VOLUME_MIN`` when MT5 is unavailable or the
        symbol cannot be looked up.
    """
    if not _MT5_AVAILABLE or sl_distance_price <= 0.0 or account_balance <= 0.0:
        logger.warning(
            "calculate_lot_size: invalid inputs or MT5 unavailable – "
            "returning fallback volume %.2f",
            _FALLBACK_VOLUME_MIN,
        )
        return _FALLBACK_VOLUME_MIN

    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning(
            "calculate_lot_size: mt5.symbol_info('%s') returned None – "
            "returning fallback volume %.2f",
            symbol,
            _FALLBACK_VOLUME_MIN,
        )
        return _FALLBACK_VOLUME_MIN

    contract_size: float = info.trade_contract_size  # e.g. 1.0 for BTCUSD
    volume_min: float = info.volume_min or _FALLBACK_VOLUME_MIN
    volume_max: float = info.volume_max or 100.0
    volume_step: float = info.volume_step or _FALLBACK_VOLUME_STEP

    risk_amount = account_balance * risk_pct
    lots_raw = risk_amount / (sl_distance_price * contract_size)

    # Use Decimal arithmetic to avoid floating-point precision errors when
    # stepping down to the nearest broker volume_step (e.g. 0.12000000001).
    lots_decimal = Decimal(str(lots_raw))
    volume_step_decimal = Decimal(str(volume_step))
    lots_quotient = (lots_decimal / volume_step_decimal).quantize(
        Decimal(1),
        rounding=ROUND_DOWN,
    )
    lots_stepped = float(lots_quotient * volume_step_decimal)

    # Clamp to broker limits.
    lots = max(volume_min, min(volume_max, lots_stepped))

    # Small accounts: enforce minimum tradable volume (default 0.01) if margin allows;
    # reject when implied loss at SL would exceed risk budget × overshoot factor.
    try:
        floor_target = max(
            0.01,
            float(os.environ.get("MT5_MIN_LOT_FLOOR", "0.01").strip() or "0.01"),
        )
    except ValueError:
        floor_target = 0.01
    try:
        risk_mult = float(
            os.environ.get("MT5_FLOOR_LOT_MAX_RISK_MULT", "1.5").strip() or "1.5"
        )
    except ValueError:
        risk_mult = 1.5
    floor_target = max(floor_target, volume_min)
    step_dec = Decimal(str(volume_step))
    floor_dec = Decimal(str(floor_target))
    ceil_parts = (floor_dec / step_dec).quantize(Decimal("1"), rounding=ROUND_UP)
    min_lot_floor = float(ceil_parts * step_dec)
    min_lot_floor = min(volume_max, max(volume_min, min_lot_floor))

    if lots + 1e-12 < min_lot_floor:
        implied_loss = min_lot_floor * sl_distance_price * contract_size
        cap = account_balance * risk_pct * max(risk_mult, 1.0)
        if implied_loss > cap + 1e-9:
            logger.warning(
                "calculate_lot_size: min lot %.4f would risk %.4f at SL vs cap %.4f — skip",
                min_lot_floor,
                implied_loss,
                cap,
            )
            return 0.0
        lots = min_lot_floor

    logger.debug(
        "calculate_lot_size: symbol=%s  balance=%.2f  risk_pct=%.4f  "
        "sl_distance=%.4f  contract_size=%.4f  "
        "lots_raw=%.4f  lots_stepped=%.4f (precise)  lots=%.4f",
        symbol,
        account_balance,
        risk_pct,
        sl_distance_price,
        contract_size,
        lots_raw,
        lots_stepped,
        lots,
    )
    return lots


# ── MT5 Executor ──────────────────────────────────────────────────────────────


class MT5Executor(PaperExecutor):
    """PaperExecutor with live MetaTrader 5 execution via ``mt5.order_send``.

    All paper-book keeping, trailing-stop logic, database persistence, and
    Telegram alerts are inherited unchanged from :class:`PaperExecutor`.
    The only parts replaced are:

    * The Binance ``create_market_buy_order`` call in :meth:`try_open_trade`.
    * The Binance ``create_market_sell_order`` call in :meth:`_close_position`.
    * The Binance ``fetch_positions`` call in :meth:`sync_positions_with_exchange`.

    When ``live=False`` (default) the executor operates in pure paper mode;
    ``mt5.order_send`` calls are skipped and only the simulated book-keeping
    runs.  Set ``live=True`` to submit real orders to the connected MT5
    terminal.

    Parameters
    ----------
    db:
        DatabaseManager instance for trade persistence.
    risk_manager:
        RiskManager instance providing position-sizing and circuit-breaker
        logic.
    symbol:
        Default MT5 symbol name (e.g. ``"BTCUSD"``).
    live:
        When ``True`` real MT5 orders are sent via ``mt5.order_send``.
    risk_pct:
        Fraction of account equity to risk per trade when computing lot size.
        Defaults to ``_DEFAULT_RISK_PCT`` (1 %).
    magic:
        MT5 magic number used to tag orders from this bot instance.  Using a
        unique magic number allows the bot to identify and manage its own
        orders without interfering with manual trades.
    deviation:
        Maximum acceptable slippage in points for market orders.
    """

    def __init__(
        self,
        db: "DatabaseManager",
        risk_manager: "RiskManager",
        symbol: str = "BTCUSD",
        live: bool = False,
        risk_pct: float = _DEFAULT_RISK_PCT,
        magic: int = 20240101,
        deviation: int = 20,
    ) -> None:
        # Pass exchange=None so the parent class skips its Binance code paths.
        super().__init__(db=db, risk_manager=risk_manager, exchange=None)
        self._positions_lock = asyncio.Lock()
        self._pending_symbols: set[str] = set()
        self._live = live
        self._magic = magic
        self._deviation = deviation

        # Issue 3 – warn if risk_pct clearly above typical prop cap (skip noise at 2.5% aggressive).
        if risk_pct > 0.025 + 1e-9:
            logger.warning(
                "[MT5] ⚠️ RISK ALERT: risk_pct=%.4f exceeds 2.5%%. "
                "For prop-firm trading keep risk_pct <= 1%%.",
                risk_pct,
            )
        self._risk_pct = risk_pct

        # Issue 6 – circuit breaker counters to detect MT5 terminal outages.
        self._mt5_failure_count: int = 0
        self._mt5_last_failure_time: float = 0.0
        self._mt5_max_consecutive_failures: int = 10
        self._mt5_cooldown_seconds: float = 300.0  # 5 minutes
        self._mt5_last_circuit_log_mono: float = 0.0
        # Throttle visibility logs for why TP was / wasn't pushed to MT5.
        self._mt5_tp_trace_at: dict[str, float] = {}
        # Same SL source → same clamp every tick would spam WARNING; throttle per MT5 symbol.
        self._sl_widen_last_warn_mono: dict[str, float] = {}
        # 10016 on SLTP is often quote vs clamp edge — throttle; avoid ERROR every tick.
        self._sltp_10016_last_mono: dict[str, float] = {}
        # Graceful-stop guard checked by sync/reconcile paths.
        self._shutting_down: bool = False
        # Ghost detection must be confirmed N times before closing local state.
        self._ghost_missing_counts: dict[str, int] = {}
        try:
            self._ghost_min_confirmations = max(
                1, int(os.environ.get("MT5_GHOST_MIN_CONFIRMATIONS", "3").strip() or "3")
            )
        except ValueError:
            self._ghost_min_confirmations = 3

    def begin_shutdown(self) -> None:
        """Signal graceful shutdown to stop broker-sync side effects."""
        self._shutting_down = True

    def _mt5_terminal_connected(self) -> bool:
        """Return True only when MT5 terminal is still alive/connected."""
        if not _MT5_AVAILABLE:
            return False
        try:
            tinfo = mt5.terminal_info()
            ainfo = mt5.account_info()
        except Exception:  # noqa: BLE001
            return False
        return tinfo is not None and ainfo is not None

    def _ghost_key(self, sym: str, ticket: int | None) -> str:
        if isinstance(ticket, int) and ticket > 0:
            return f"ticket:{ticket}"
        return f"symbol:{sym}"

    def _ghost_mark_missing(self, key: str, confirmations_required: int | None = None) -> bool:
        """Increment miss counter. True when ghost is confirmed."""
        count = self._ghost_missing_counts.get(key, 0) + 1
        self._ghost_missing_counts[key] = count
        needed = self._ghost_min_confirmations if confirmations_required is None else max(1, confirmations_required)
        return count >= needed

    def _ghost_reset(self, key: str) -> None:
        self._ghost_missing_counts.pop(key, None)

    def _tp_sync_trace(self, sym: str, reason: str, message: str) -> None:
        """Why TP/SLTP sync skipped or changed; throttled per (sym,reason).

        INFO → console + ``logs/last_session.log`` + ``bot_debug.log``.
        Interval ``MT5_TP_TRACE_INTERVAL_S`` (default 45; ``0`` = every call, very noisy).
        """
        try:
            interval = float(os.environ.get("MT5_TP_TRACE_INTERVAL_S", "45").strip() or "45")
        except ValueError:
            interval = 45.0
        key = f"{sym}|{reason}"
        now = time.monotonic()
        if interval > 0:
            last = self._mt5_tp_trace_at.get(key, 0.0)
            if (now - last) < interval:
                return
            self._mt5_tp_trace_at[key] = now
        logger.info("[MT5 TP sync] %s | %s | %s", sym, reason, message)

    def _log_sltp_10016_throttled(self, mt5_sym: str | None) -> None:
        """INVALID_STOPS on position modify — expected at broker edge; do not log as ERROR each tick."""
        try:
            interval = float(
                os.environ.get("MT5_SLTP_10016_LOG_INTERVAL_S", "90").strip() or "90"
            )
        except ValueError:
            interval = 90.0
        key = mt5_sym or "?"
        now_m = time.monotonic()
        if interval > 0:
            last = self._sltp_10016_last_mono.get(key, 0.0)
            if (now_m - last) < interval:
                logger.debug(
                    "[MT5 RETCODE 10016] INVALID_STOPS SL/TP %s (suppressed, %.0fs throttle)",
                    key,
                    interval,
                )
                return
            self._sltp_10016_last_mono[key] = now_m
        logger.warning(
            "[MT5] SL/TP modify rejected — 10016 INVALID_STOPS (%s). "
            "Quote vs min distance/freeze; next sync re-clamps. "
            "If often: raise MT5_STOPS_BUFFER_POINTS.",
            key,
        )

    # ------------------------------------------------------------------
    # Spread helper
    # ------------------------------------------------------------------

    def _get_spread_fraction(self, symbol: str, entry_price: float) -> float:
        """Return the current spread as a fraction of *entry_price*.

        Used to inflate the trailing-stop distance so that normal bid/ask
        spread widening does not trigger a premature exit.

        Falls back to :data:`_SPREAD_BUFFER_FALLBACK` when MT5 is unavailable
        or the tick cannot be fetched.
        """
        if not _MT5_AVAILABLE or not self._live:
            return _SPREAD_BUFFER_FALLBACK
        mt5_sym = self._resolve_symbol(symbol)
        if mt5_sym is None:
            return _SPREAD_BUFFER_FALLBACK
        try:
            tick = mt5.symbol_info_tick(mt5_sym)
            if tick is None or tick.ask <= 0 or tick.bid <= 0:
                return _SPREAD_BUFFER_FALLBACK
            spread_price = tick.ask - tick.bid
            return spread_price / entry_price if entry_price > 0 else _SPREAD_BUFFER_FALLBACK
        except (OSError, TimeoutError, AttributeError, TypeError) as exc:
            logger.debug(
                "_get_spread_fraction: failed to fetch tick for %s – "
                "using fallback spread buffer. Error: %s",
                symbol,
                exc,
                exc_info=True,
            )
            return _SPREAD_BUFFER_FALLBACK

    def _effective_trailing_distance(
        self,
        symbol: str,
        entry_price: float,
        trailing_distance_pct: float,
    ) -> float:
        """Return trailing distance inflated by the current spread.

        Adjusting for spread prevents the trailing stop from triggering
        prematurely when the broker widens the spread (common during low
        liquidity windows such as weekend re-opens).

        The effective distance is::

            effective = trailing_distance_pct + spread_fraction

        where ``spread_fraction = spread_price / entry_price``.

        Parameters
        ----------
        symbol:
            MT5 symbol name.
        entry_price:
            Position entry price (used to normalise the spread).
        trailing_distance_pct:
            Base trailing distance as a fraction (e.g. ``0.015`` = 1.5 %).

        Returns
        -------
        float
            Effective trailing distance fraction ≥ *trailing_distance_pct*.
        """
        spread_fraction = self._get_spread_fraction(symbol, entry_price)
        effective = trailing_distance_pct + spread_fraction
        if spread_fraction > _SPREAD_BUFFER_FALLBACK:
            logger.debug(
                "_effective_trailing_distance: symbol=%s  base=%.4f  "
                "spread_fraction=%.4f  effective=%.4f",
                symbol,
                trailing_distance_pct,
                spread_fraction,
                effective,
            )
        return effective

    # ------------------------------------------------------------------
    # Symbol resolution and price normalisation helpers
    # ------------------------------------------------------------------

    def _resolve_symbol(self, sym: str) -> str | None:
        """Translate an internal symbol name to the broker's MT5 symbol name.

        Looks up *sym* in :data:`SYMBOL_MAP`.  If the mapped symbol exists but
        is not currently visible in the Market Watch, this method calls
        ``mt5.symbol_select`` to add it automatically.

        Parameters
        ----------
        sym:
            Internal symbol (e.g. ``"BTC/USDT"`` or ``"BTCUSD"``).

        Returns
        -------
        str | None
            The broker symbol string (e.g. ``"BTCUSD-T"``), or ``None`` when
            the symbol is not in :data:`SYMBOL_MAP` or cannot be validated.
        """
        # Allow direct broker symbols (e.g. "BTCUSD-T") to pass through.
        if sym in SYMBOL_MAP.values():
            broker_sym = sym
        else:
            broker_sym = SYMBOL_MAP.get(sym)
        if broker_sym is None:
            logger.error(
                "[MT5] Unmapped symbol: '%s'. Add it to SYMBOL_MAP in mt5_executor.py.",
                sym,
            )
            return None

        if not _MT5_AVAILABLE:
            return broker_sym  # Paper mode – no terminal to query.

        info = mt5.symbol_info(broker_sym)
        if info is None:
            logger.error(
                "[MT5] mt5.symbol_info('%s') returned None – "
                "symbol does not exist on the broker server.",
                broker_sym,
            )
            return None

        if not info.visible:
            if not mt5.symbol_select(broker_sym, True):
                logger.error(
                    "[MT5] Could not add '%s' to Market Watch. "
                    "Add it manually in MetaTrader 5.",
                    broker_sym,
                )
                return None
            logger.info("[MT5] Symbol '%s' added to Market Watch.", broker_sym)

        return broker_sym

    def validate_symbol_mapping(self, watchlist: list[str]) -> list[str]:
        """Validate/auto-heal watchlist symbol mapping against broker availability."""
        if not _MT5_AVAILABLE:
            return []
        unresolved: list[str] = []
        for internal in watchlist:
            mapped = SYMBOL_MAP.get(internal)
            if mapped is None:
                unresolved.append(f"{internal} -> <missing-map>")
                continue
            if mt5.symbol_info(mapped) is not None:
                continue
            variants = [
                mapped.replace("-T", ""),
                mapped.replace("-T", "m"),
                f"{mapped}.r",
                mapped + "m",
                "ETHUSD-T",
                "DGEUSD-T",
            ]
            # Dedupe while preserving order
            variants = list(dict.fromkeys(variants))
            fixed: str | None = None
            for v in variants:
                if mt5.symbol_info(v) is not None:
                    fixed = v
                    break
            if fixed is None:
                unresolved.append(f"{internal} -> {mapped}")
                continue
            SYMBOL_MAP[internal] = fixed
            base = internal.replace("/USDT", "USD")
            SYMBOL_MAP[base] = fixed
            logger.warning(
                "[MT5] Auto-remapped %s from %s to broker symbol %s",
                internal,
                mapped,
                fixed,
            )
        return unresolved

    @staticmethod
    def _local_symbol_from_broker(broker_symbol: str) -> str:
        """Map broker MT5 symbol back to the local/internal symbol name."""
        reverse_map = {v: k for k, v in SYMBOL_MAP.items() if "/" in k}
        return reverse_map.get(broker_symbol, broker_symbol)

    @staticmethod
    def _normalize_price(price: float, digits: int) -> float:
        """Round *price* to the number of decimal places required by the broker.

        Parameters
        ----------
        price:
            Raw price value (e.g. stop-loss level).
        digits:
            Number of significant decimal places from ``symbol_info.digits``
            (e.g. ``2`` for BTCUSD-T on Admirals).

        Returns
        -------
        float
            Price rounded to *digits* decimal places.
        """
        return round(price, digits)

    # ------------------------------------------------------------------
    # MT5 order helpers
    # ------------------------------------------------------------------

    def _build_buy_request(
        self,
        symbol: str,
        lots: float,
        price: float,
        sl: float,
        comment: str = "ClawdBot BUY",
    ) -> dict:
        """Build a market BUY order request dict for ``mt5.order_send``."""
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    def _build_sell_request(
        self,
        symbol: str,
        lots: float,
        price: float,
        position_id: int,
        comment: str = "ClawdBot SELL",
    ) -> dict:
        """Build a market SELL (close) order request dict for ``mt5.order_send``."""
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "position": position_id,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    # ------------------------------------------------------------------
    # Pre-trade validation helpers (Issues 1, 2, 4)
    # ------------------------------------------------------------------

    def _clamp_stop_loss_buy(
        self,
        mt5_sym: str,
        sl: float,
        ask: float,
        tick: Any,
        digits: int,
    ) -> tuple[float | None, bool]:
        """If SL is closer than broker ``trade_stops_level`` (+ buffer), widen it.

        For a LONG, SL must sit at least ``stops_level`` points from both the
        order price (ask) and the trigger price (bid). Optional extra buffer
        points via ``MT5_STOPS_BUFFER_POINTS`` (default ``2``) avoids edge
        rejections from rounding/spread.

        Returns
        -------
        (sl_clamped, adjusted)
            ``sl_clamped`` is ``None`` if no valid level exists.
        """
        if not _MT5_AVAILABLE:
            return sl, False

        info = mt5.symbol_info(mt5_sym)
        if info is None or tick is None:
            return None, False

        point = float(info.point or 0.0)
        if point <= 0:
            return None, False

        stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
        freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)
        try:
            buffer_pts = float(os.environ.get("MT5_STOPS_BUFFER_POINTS", "2").strip() or "2")
        except ValueError:
            buffer_pts = 2.0

        # Stops + freeze (broker forbids SL/TP mods inside freeze distance — causes 10016).
        min_total_pts = float(stops_level + freeze_level) + buffer_pts
        min_total_price = min_total_pts * point

        bid = float(tick.bid)
        ask_f = float(ask)
        if bid <= 0 or ask_f <= 0:
            return None, False

        # Tightest SL price still allowed (far enough below bid and ask).
        sl_ceiling = min(ask_f - min_total_price, bid - min_total_price)
        if sl_ceiling <= 0:
            logger.error(
                "[MT5] Invalid SL ceiling for %s (bid=%.5f ask=%.5f min_pts=%.1f).",
                mt5_sym,
                bid,
                ask_f,
                min_total_pts,
            )
            return None, False

        original = float(sl)
        sl_adj = min(original, sl_ceiling)
        sl_adj = self._normalize_price(sl_adj, digits)

        for _ in range(64):
            if self._validate_stops(mt5_sym, ask_f, sl_adj, 0.0, is_buy=True, tick=tick):
                adjusted = abs(sl_adj - self._normalize_price(original, digits)) > point * 0.01
                if adjusted:
                    try:
                        widen_iv = float(
                            os.environ.get("MT5_SL_WIDEN_LOG_INTERVAL_S", "120").strip()
                            or "120"
                        )
                    except ValueError:
                        widen_iv = 120.0
                    now_m = time.monotonic()
                    last_m = self._sl_widen_last_warn_mono.get(mt5_sym, 0.0)
                    msg = (
                        "[MT5] SL widened to broker minimum for %s: %.5f → %.5f "
                        "(stops_level=%d + buffer=%.1f pts)"
                    ) % (
                        mt5_sym,
                        original,
                        sl_adj,
                        stops_level,
                        buffer_pts,
                    )
                    if widen_iv <= 0 or (now_m - last_m) >= widen_iv:
                        self._sl_widen_last_warn_mono[mt5_sym] = now_m
                        logger.warning(msg)
                    else:
                        logger.debug(msg)
                return sl_adj, adjusted
            sl_adj = self._normalize_price(sl_adj - point, digits)
            if sl_adj <= 0 or sl_adj >= bid:
                break

        logger.error(
            "[MT5] Could not clamp SL for %s – bid=%.5f ask=%.5f last_sl=%.5f",
            mt5_sym,
            bid,
            ask_f,
            sl_adj,
        )
        return None, False

    def _validate_stops(
        self,
        symbol: str,
        execution_price: float,
        sl: float,
        tp: float = 0.0,
        is_buy: bool = True,
        *,
        tick: Any | None = None,
    ) -> bool:
        """Check that SL/TP distances satisfy broker minimum distance.

        Uses ``trade_stops_level + trade_freeze_level`` (points × point) — freeze
        applies to SL/TP modifications and triggers 10016 if omitted.

        CRITICAL: For a BUY order, the SL is triggered by the BID price.
        For a SELL order, it is triggered by the ASK price. This method uses the
        appropriate reference price to ensure compliance.

        Pass *tick* when validating in the same cycle as a clamp or quote read;
        otherwise a fresh ``symbol_info_tick`` can differ by a few points and
        spuriously fail while price moves.
        """
        if not _MT5_AVAILABLE:
            return True

        info = mt5.symbol_info(symbol)
        tick_use = tick if tick is not None else mt5.symbol_info_tick(symbol)
        if not info or not tick_use:
            logger.error("[MT5] Cannot fetch symbol_info/tick for stops validation on %s.", symbol)
            return False

        # Reference price for SL triggering:
        # BUY positions exit at BID; SELL positions exit at ASK.
        ref_price = tick_use.bid if is_buy else tick_use.ask
        stops_pts = int(getattr(info, "trade_stops_level", 0) or 0)
        freeze_pts = int(getattr(info, "trade_freeze_level", 0) or 0)
        min_pts = stops_pts + freeze_pts
        min_distance_price = min_pts * info.point

        # 1. Check distance from execution price (the price in the order request)
        sl_from_exec = abs(execution_price - sl)
        if sl_from_exec < min_distance_price:
            logger.error(
                "[MT5 VALIDATION] SL too close to ORDER PRICE for %s: "
                "min %.2f pts (stops+freeze), current %.2f pts. Order: %.5f | SL: %.5f",
                symbol,
                float(min_pts),
                sl_from_exec / info.point if info.point > 0 else 0,
                execution_price,
                sl,
            )
            return False

        # 2. Check distance from trigger price (Bid/Ask) - This is usually what causes 10016
        sl_from_ref = abs(ref_price - sl)
        if sl_from_ref < min_distance_price:
            logger.error(
                "[MT5 VALIDATION] SL too close to TRIGGER PRICE (%s) for %s: "
                "min %.2f pts (stops+freeze), current %.2f pts. Trigger: %.5f | SL: %.5f. (SPREAD TOO WIDE?)",
                "BID" if is_buy else "ASK",
                symbol,
                float(min_pts),
                sl_from_ref / info.point if info.point > 0 else 0,
                ref_price,
                sl,
            )
            return False

        # 3. Ensure SL is on the correct side
        if is_buy and sl >= ref_price:
            logger.error("[MT5 VALIDATION] SL for BUY must be BELOW price. SL: %.5f | Bid: %.5f", sl, ref_price)
            return False
        if not is_buy and sl <= ref_price:
            logger.error("[MT5 VALIDATION] SL for SELL must be ABOVE price. SL: %.5f | Ask: %.5f", sl, ref_price)
            return False

        if tp > 0.0:
            tp_from_ref = abs(ref_price - tp)
            if tp_from_ref < min_distance_price:
                logger.error(
                    "[MT5 VALIDATION] TP too close to trigger for %s: min %.2f pts, current %.2f pts.",
                    symbol,
                    float(min_pts),
                    tp_from_ref / info.point if info.point > 0 else 0,
                )
                return False

        return True

    def _check_margin_available(self, symbol: str, lots: float, entry_price: float) -> bool:
        """Verify the account has sufficient free margin to open the trade.

        Prevents MT5 from returning NO_MONEY (10019) by comparing the required
        margin against the account's current free margin before sending the
        order.

        Parameters
        ----------
        symbol:
            Resolved MT5 broker symbol (e.g. ``"BTCUSD-T"``).
        lots:
            Position size in lots.
        entry_price:
            Expected entry price, used as a fallback when ``margin_initial``
            is not available from the broker.

        Returns
        -------
        bool
            ``True`` when free margin is sufficient, ``False`` otherwise.
        """
        if not _MT5_AVAILABLE:
            return True

        info = mt5.symbol_info(symbol)
        acct = mt5.account_info()

        if not info:
            logger.error("[MT5] Cannot fetch symbol_info for margin check on %s.", symbol)
            return False

        if not acct:
            logger.error("[MT5] Cannot fetch account_info for margin check.")
            return False

        margin_per_lot = (
            info.margin_initial
            if info.margin_initial > 0
            else (info.trade_contract_size * entry_price / 100)
        )
        margin_required = lots * margin_per_lot

        if acct.margin_free < margin_required:
            logger.error(
                "[MT5] INSUFFICIENT MARGIN for %s: "
                "%.2f required, %.2f available "
                "(equity=%.2f, balance=%.2f, used=%.2f).",
                symbol,
                margin_required,
                acct.margin_free,
                acct.equity,
                acct.balance,
                acct.margin,
            )
            return False

        margin_ratio = (
            (acct.margin_free - margin_required) / acct.equity
            if acct.equity > 0
            else 0.0
        )
        if margin_ratio < 0.2:
            logger.warning(
                "[MT5] Margin is tight after this trade: %.1f%% of equity remaining. "
                "Consider reducing position size.",
                margin_ratio * 100,
            )

        logger.debug(
            "[MT5] Margin OK – required=%.2f, free=%.2f, post_trade_ratio=%.1f%%",
            margin_required,
            acct.margin_free,
            margin_ratio * 100,
        )
        return True

    def _validate_tick_freshness(
        self,
        tick: Any,
        symbol: str,
        max_age_seconds: float = 5.0,
    ) -> bool:
        """Reject a tick that is stale or has invalid bid/ask prices.

        Guards against entering at a bad price when the MT5 terminal has not
        yet received a fresh quote from the broker (e.g. immediately after
        market open or a connection drop).

        Parameters
        ----------
        tick:
            ``TickInfo`` object returned by ``mt5.symbol_info_tick()``, or
            ``None`` if the call failed.
        symbol:
            MT5 symbol name, used only for log messages.
        max_age_seconds:
            Maximum acceptable age of the tick in seconds (default: 5 s).

        Returns
        -------
        bool
            ``True`` when the tick is valid and fresh, ``False`` otherwise.
        """
        if tick is None:
            logger.error("[MT5] Tick is None for %s – cannot validate freshness.", symbol)
            return False

        if tick.ask <= 0 or tick.bid <= 0:
            logger.error(
                "[MT5] Invalid tick prices for %s: bid=%.5f, ask=%.5f",
                symbol,
                tick.bid,
                tick.ask,
            )
            return False

        tick_age = time.time() - tick.time
        if tick_age > max_age_seconds:
            logger.error(
                "[MT5] Stale tick for %s: %.1f s old (max: %.1f s).",
                symbol,
                tick_age,
                max_age_seconds,
            )
            return False

        logger.debug(
            "[MT5] Tick fresh for %s: age=%.2f s, bid=%.5f, ask=%.5f, spread=%.5f",
            symbol,
            tick_age,
            tick.bid,
            tick.ask,
            tick.ask - tick.bid,
        )
        return True

    def _broker_blocks_new_long_on_symbol(self, mt5_sym: str) -> tuple[bool, str]:
        """True if MT5 already has a BUY on *mt5_sym* so we must not stack another long.

        Foreign (non-bot magic) positions are rejected by default so manual trades
        do not collide with bot entries on the same instrument.  Set environment
        ``MT5_ALLOW_FOREIGN_SYMBOL_OVERLAP=1`` to allow opens while a manual long
        exists (dangerous — double exposure).

        Also blocks when the broker already shows a bot-tagged BUY (sync lag vs
        local ``open_positions``).
        """
        if not _MT5_AVAILABLE:
            return False, ""
        raw_allow = os.environ.get("MT5_ALLOW_FOREIGN_SYMBOL_OVERLAP", "0").strip().lower()
        allow_foreign = raw_allow in ("1", "true", "yes")
        raw = mt5.positions_get(symbol=mt5_sym)
        if not raw:
            return False, ""
        for p in raw:
            if p.type != mt5.POSITION_TYPE_BUY:
                continue
            if p.magic != self._magic:
                if allow_foreign:
                    continue
                return True, "foreign_long_on_symbol"
            return True, "broker_already_has_bot_long"
        return False, ""

    async def _send_order_with_retry(
        self,
        request: dict,
        max_retries: int = 3,
    ) -> Any | None:
        """Send an MT5 order with automatic retry on transient failures.

        Retries up to *max_retries* times when the return code is in
        :data:`_RETRYABLE_RETCODES` (e.g. requote, price changed, connection
        error).  Each attempt waits an increasing back-off before retrying.

        A circuit breaker prevents hammering the broker when the MT5 terminal
        is down: after ``_mt5_max_consecutive_failures`` consecutive failures
        the executor pauses for ``_mt5_cooldown_seconds`` before accepting
        new orders.  The counter resets on the first successful order.

        Parameters
        ----------
        request:
            Order request dictionary to pass to ``mt5.order_send``.
        max_retries:
            Maximum number of attempts (default: 3).

        Returns
        -------
        Any | None
            The ``order_send`` result on success, or ``None`` on final failure.
        """
        # Issue 6 – circuit breaker: refuse new orders while terminal is down.
        if self._mt5_failure_count >= self._mt5_max_consecutive_failures:
            elapsed = time.time() - self._mt5_last_failure_time
            if elapsed < self._mt5_cooldown_seconds:
                remaining = self._mt5_cooldown_seconds - elapsed
                now_mono = time.monotonic()
                if now_mono - self._mt5_last_circuit_log_mono >= 30.0:
                    self._mt5_last_circuit_log_mono = now_mono
                    logger.error(
                        "[MT5 CIRCUIT BREAKER] Too many consecutive failures (%d). "
                        "System paused for %.0f s. Retry in %.0f s.",
                        self._mt5_failure_count,
                        self._mt5_cooldown_seconds,
                        remaining,
                    )
                return None
            else:
                # Cooldown expired – reset and allow a new attempt.
                logger.info(
                    "[MT5 CIRCUIT BREAKER] Cooldown expired. "
                    "Resetting failure counter (was %d).",
                    self._mt5_failure_count,
                )
                self._mt5_failure_count = 0

        if not _MT5_AVAILABLE:
            return None

        for attempt in range(1, max_retries + 1):
            result = mt5.order_send(request)

            if result is None:
                self._mt5_failure_count += 1
                self._mt5_last_failure_time = time.time()
                logger.error(
                    "mt5.order_send returned None (attempt %d/%d, failure_count=%d) for %s. "
                    "Last error: %s",
                    attempt,
                    max_retries,
                    self._mt5_failure_count,
                    request.get("symbol"),
                    mt5.last_error(),
                )
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * attempt)
                continue

            sltp_10016 = (
                result.retcode == 10016
                and request.get("action") == mt5.TRADE_ACTION_SLTP
            )
            # Do not ERROR-log expected SLTP edge rejects; _log_sltp_10016_throttled handles it.
            if not sltp_10016:
                _log_mt5_retcode(
                    result.retcode,
                    context=(
                        f"attempt {attempt}/{max_retries}  symbol={request.get('symbol')}  "
                        f"failure_count={self._mt5_failure_count}"
                    ),
                )

            if result.retcode in _MT5_SOFT_SESSION_RETCODES:
                return None

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Success – reset circuit breaker counter.
                if self._mt5_failure_count > 0:
                    logger.info(
                        "[MT5] Order succeeded – resetting failure counter (was %d).",
                        self._mt5_failure_count,
                    )
                self._mt5_failure_count = 0

                action = request.get("action")
                if action == mt5.TRADE_ACTION_SLTP:
                    logger.info(
                        "MT5 SL/TP MODIFIED  position=%d  sl=%.5f  tp=%.5f",
                        request.get("position", 0),
                        request.get("sl", 0.0),
                        request.get("tp", 0.0),
                    )
                else:
                    logger.info(
                        "MT5 ORDER PLACED  symbol=%s  type=%s  volume=%.4f  price=%.5f  "
                        "order=%d  deal=%d",
                        request.get("symbol"),
                        "BUY" if request.get("type") == mt5.ORDER_TYPE_BUY else "SELL",
                        request.get("volume", 0.0),
                        result.price,
                        result.order,
                        result.deal,
                    )
                return result

            # SLTP + 10016: retrying the *same* request dict never fixes INVALID_STOPS —
            # quote moved vs clamp; next _sync_exchange_stops tick rebuilds levels.
            if sltp_10016:
                self._log_sltp_10016_throttled(request.get("symbol"))
                return None

            if result.retcode in _RETRYABLE_RETCODES and attempt < max_retries:
                self._mt5_failure_count += 1
                self._mt5_last_failure_time = time.time()
                back_off = 0.5 * attempt
                logger.warning(
                    "[MT5] Transient error for %s (attempt %d/%d, failure_count=%d) – "
                    "retrying in %.1f s...",
                    request.get("symbol"),
                    attempt,
                    max_retries,
                    self._mt5_failure_count,
                    back_off,
                )
                await asyncio.sleep(back_off)
            else:
                self._mt5_failure_count += 1
                self._mt5_last_failure_time = time.time()
                logger.error(
                    "[MT5] Non-retryable error or final attempt exhausted "
                    "(failure_count=%d).",
                    self._mt5_failure_count,
                )
                return None

        return None

    # ------------------------------------------------------------------
    # try_open_trade override
    # ------------------------------------------------------------------

    async def try_open_trade(
        self,
        entry_price: float,
        win_probability: float,
        symbol: str | None = None,
        timestamp: datetime | None = None,
        sentiment_score: float = 0.0,
        current_atr: float | None = None,
    ) -> bool:
        """Open a new long position using MT5 OrderSend (when *live=True*).

        Performs all the same risk checks, position sizing, and book-keeping as
        :meth:`PaperExecutor.try_open_trade`.  When *live=True* a real market
        BUY order is sent to MT5 before the paper position is recorded.  The
        stop-loss level is passed directly in the order request so MT5 itself
        manages the hard SL on the broker side.

        On order rejection the balance deduction is rolled back and ``False``
        is returned, identical to the Binance error path.

        Spread adjustment
        ~~~~~~~~~~~~~~~~~
        The effective trailing distance stored on the position is inflated by
        the current spread fraction via :meth:`_effective_trailing_distance` so
        that normal spread widening does not prematurely trigger the trailing
        stop when it activates later.
        """
        sym = symbol or self.symbol
        cfg_sym = get_symbol_config(sym)
        risk_pct_sym = float(cfg_sym["risk"])
        fixed_sl_sym = min(float(cfg_sym["fixed_sl_pct"]), _SL_CAP_VALUE)

        # ── MT5 symbol resolution (fail-fast before any state mutation) ────
        mt5_sym: str | None = None
        if self._live:
            mt5_sym = self._resolve_symbol(sym)
            if mt5_sym is None:
                return False

        # ── All pre-checks under lock to prevent race conditions ───────────
        async with self._positions_lock:
            if self._risk.is_trading_halted():
                logger.warning(
                    "⚠️ [ALERT] Trading halted due to daily loss limit (symbol=%s).",
                    sym,
                )
                return False

            if self._risk.is_portfolio_dd_exceeded():
                logger.warning(
                    "🚨 [CIRCUIT BREAKER] All new positions blocked – "
                    "portfolio drawdown limit reached (symbol=%s).",
                    sym,
                )
                return False

            if sym in self.open_positions or sym in self._pending_symbols:
                logger.debug("Trade skipped – a position for %s is already open.", sym)
                return False

            if not self._risk.can_open_position():
                logger.debug(
                    "Trade skipped – max open positions (%d) reached.",
                    self._risk.max_positions,
                )
                return False

            occupied_syms = set(self.open_positions.keys()) | self._pending_symbols
            if self._risk.is_sector_exposed(sym, list(occupied_syms)):
                sector = get_sector(sym)
                logger.warning(
                    "🛡️ [RISK CONTROL] BUY signal for %s ignored – "
                    "maximum sector exposure reached: %s.",
                    sym,
                    sector,
                )
                return False

            position_size = self._risk.calculate_position_size(
                win_probability,
                risk_pct=risk_pct_sym,
            )
            if position_size <= 0.0:
                logger.warning(
                    "⚠️ [ALERT] Position size is 0 for %s — portfolio risk budget exhausted or drawdown limit.",
                    sym,
                )
                return False
            if not self._risk.has_sufficient_balance(position_size):
                logger.warning(
                    "⚠️ [ALERT] Insufficient balance (%.2f) for position size %.2f.",
                    self._risk.balance,
                    position_size,
                )
                return False

            ts = timestamp or datetime.now(tz=timezone.utc)
            
            # [PRO] Portfolio risk accounting
            risk_usd = position_size * fixed_sl_sym
            self._risk.register_open(risk_usd=risk_usd)
            self._pending_symbols.add(sym)

        # ── Per-symbol fixed SL (SYMBOL_CONFIG) + spread-aware trailing % ────
        th0 = get_execution_thresholds()
        effective_trailing_pct = self._effective_trailing_distance(
            sym, entry_price, th0.trailing_distance_pct
        )
        thresholds = DynamicThresholds(
            multiplier=th0.multiplier,
            sl_pct=fixed_sl_sym,
            activation_pct=th0.activation_pct,
            trailing_distance_pct=effective_trailing_pct,
        )
        stop_loss_price = float(entry_price) * (1.0 - fixed_sl_sym)
        atr_trailing_distance = (
            float(current_atr) * ATR_TRAILING_MULTIPLIER
            if current_atr is not None and float(current_atr) > 0.0
            else 0.0
        )
        sl_distance: float | None = max(float(entry_price) - stop_loss_price, 0.0)

        # ── Live MT5 order ─────────────────────────────────────────────────
        mt5_ticket: int | None = None
        if self._live:
            if not _MT5_AVAILABLE:
                logger.error(
                    "MT5 live mode requested but MetaTrader5 library is not "
                    "installed.  Rolling back and aborting."
                )
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False

            # mt5_sym is guaranteed non-None here (resolved before deductions).
            assert mt5_sym is not None  # noqa: S101

            block_open, _occ_reason = self._broker_blocks_new_long_on_symbol(mt5_sym)
            if block_open:
                logger.warning(
                    "[MT5] New LONG blocked (%s) on %s — sym=%s",
                    _occ_reason,
                    mt5_sym,
                    sym,
                )
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False

            sym_info = mt5.symbol_info(mt5_sym)
            digits: int = sym_info.digits if sym_info else 5

            # Issue 4 – reject stale tick prices before entering.
            tick = mt5.symbol_info_tick(mt5_sym)
            if not self._validate_tick_freshness(tick, mt5_sym, max_age_seconds=5.0):
                logger.error("[MT5] Tick freshness check failed – aborting trade for %s.", sym)
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False

            ask_price = self._normalize_price(tick.ask, digits)
            stop_loss_price = self._normalize_price(
                float(tick.bid) * (1.0 - fixed_sl_sym),
                digits,
            )

            # Issue 1 – widen SL if tighter than broker ``trade_stops_level`` (+ buffer).
            sl_clamped, _ = self._clamp_stop_loss_buy(
                mt5_sym, stop_loss_price, ask_price, tick, digits
            )
            if sl_clamped is None:
                logger.error(
                    "[MT5] Stop clamp failed (broker rules) – aborting trade for %s.",
                    sym,
                )
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False
            stop_loss_price = sl_clamped

            # Fetch account equity from MT5 for lot-size calculation.
            acct = mt5.account_info()
            account_equity: float = acct.equity if acct else self._risk.balance

            # RISK CALCULATION: distance between entry (Ask) and stop (Bid - distance)
            lots = calculate_lot_size(
                symbol=mt5_sym,
                account_balance=account_equity,
                sl_distance_price=max(ask_price - stop_loss_price, 1e-8),
                risk_pct=risk_pct_sym,
            )
            if lots <= 0.0:
                logger.error(
                    "[MT5] Lot size 0 (min-lot / risk cap) – aborting trade for %s.",
                    sym,
                )
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False

            # Issue 2 – verify sufficient margin before sending the order.
            if not self._check_margin_available(mt5_sym, lots, ask_price):
                logger.error("[MT5] Margin check failed – aborting trade for %s.", sym)
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False

            request = self._build_buy_request(
                symbol=mt5_sym,
                lots=lots,
                price=ask_price,
                sl=stop_loss_price,
            )
            result = await self._send_order_with_retry(request)
            if result is None:
                # Order rejected – roll back counter (balance wasn't deducted)
                async with self._positions_lock:
                    self._pending_symbols.discard(sym)
                    self._risk.register_close()
                return False
            mt5_ticket = await self._resolve_position_ticket_after_buy(mt5_sym, result)

        try:
            trade_id = await self._db.insert_open_trade(
                symbol=sym,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=ts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[MT5][DB] insert_open_trade failed for %s — rolling back: %s", sym, exc)
            async with self._positions_lock:
                self._pending_symbols.discard(sym)
                self._risk.register_close()
            return False

        act_px = float(entry_price) * (1.0 + thresholds.activation_pct)
        async with self._positions_lock:
            self.open_positions[sym] = OpenPosition(
                trade_id=trade_id,
                symbol=sym,
                entry_time=ts,
                entry_price=entry_price,
                position_size=position_size,
                sl_price=stop_loss_price,
                activation_price=act_px,
                trailing_distance_pct=thresholds.trailing_distance_pct,
                peak_price=float(entry_price),
                ml_confidence=win_probability,
                sl_pct=thresholds.sl_pct,
                activation_pct=thresholds.activation_pct,
                stop_loss_price=stop_loss_price,
                atr_trailing_distance=atr_trailing_distance,
                sentiment_score=sentiment_score,
                mt5_position_ticket=mt5_ticket,
                current_stop_loss=stop_loss_price,
            )
            pos_ref = self.open_positions[sym]
            self._pending_symbols.discard(sym)

        js = journal_symbol(sym)
        record_trade(
            timestamp=ts,
            symbol=js,
            action="BUY",
            execution_price=entry_price,
            quantity=position_size / entry_price,
            ml_confidence_at_entry=win_probability,
            sentiment_score_at_entry=sentiment_score,
            idempotency_key=f"BUY:{js}:ticket:{mt5_ticket or trade_id}",
        )
        self.save_state()

        if sl_distance is not None and sl_distance > 0:
            logger.info(
                "✅ [OPEN LONG] %s at %.2f. Fixed SL %%: %.2f%% → price %.2f (dist≈%.4f).",
                sym,
                entry_price,
                thresholds.sl_pct * 100.0,
                stop_loss_price,
                sl_distance,
            )
        else:
            logger.info(
                "🚀 [ENTRY] %s | Side: BUY | Confidence: %.1f%% | "
                "SL: %.2f%% | TP activation: %.2f%%.",
                sym,
                win_probability * 100,
                thresholds.sl_pct * 100,
                thresholds.activation_pct * 100,
            )

        asyncio.create_task(
            send_telegram_alert(
                f"🚀 *OPEN BUY* | #{sym}\n"
                f"Precio: {entry_price:.2f}\n"
                f"SL fijo (~{thresholds.sl_pct * 100:.2f}%): {self.open_positions[sym].stop_loss_price:.2f}"
            )
        )
        if self._live and isinstance(mt5_ticket, int):
            await self._verify_initial_sl_synced(sym, pos_ref, mt5_ticket)
        return True

    # ------------------------------------------------------------------
    # Broker-close bookkeeping (SL/TP filled on server before our sell)
    # ------------------------------------------------------------------

    async def _apply_mt5_closed_bookkeeping(
        self,
        symbol: str,
        pos: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        gross_pnl: float,
        exit_reason_code: str,
        *,
        telegram_after: bool = False,
    ) -> None:
        """Persist close, risk, journal — shared by normal exit and ghost sync."""
        fee_total = pos.position_size * _TAKER_FEE_RATE * 2
        pnl_net = gross_pnl - fee_total
        commission = 0.0
        swap = 0.0
        fee = fee_total
        mt5_ticket = getattr(pos, "mt5_position_ticket", None)
        book_net = gross_pnl - fee_total
        economics: dict[str, float] | None = None
        if self._live and isinstance(mt5_ticket, int):
            raw = self._resolve_mt5_close_economics(mt5_ticket)
            if raw is not None:
                mt5_net = float(raw["pnl_net"])
                diff = abs(mt5_net - book_net)
                # Reject deal-history aggregate that disagrees wildly with the book
                # (bad filter / cross-deal sum still produced huge phantom PnL in CEO).
                cap = max(150.0, 4.0 * max(abs(book_net), 25.0))
                if diff <= cap:
                    economics = raw
                else:
                    logger.warning(
                        "[MT5] Deal-history PnL dropped for ticket=%s "
                        "(book_net≈%.2f mt5_net=%.2f diff=%.2f > cap=%.2f).",
                        mt5_ticket,
                        book_net,
                        mt5_net,
                        diff,
                        cap,
                    )
            if economics is not None:
                pnl_net = economics["pnl_net"]
                commission = economics["commission"]
                swap = economics["swap"]
                fee = economics["fee"]

        if pos.trade_id is not None:
            try:
                await self._db.close_trade(
                    trade_id=pos.trade_id,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    pnl=gross_pnl,
                    pnl_net=pnl_net,
                    commission=commission,
                    swap=swap,
                    fee=fee,
                    exit_reason=exit_reason_code,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[MT5][DB] close_trade failed trade_id=%s — %s",
                    pos.trade_id,
                    exc,
                )

        async with self._positions_lock:
            self._risk.credit(gross_pnl)
            
            # [PRO] Portfolio risk accounting
            risk_usd = pos.position_size * pos.sl_pct
            self._risk.register_close(risk_usd=risk_usd)
            
            if gross_pnl < 0.0:
                self._risk.record_daily_loss(-gross_pnl)
            self.total_pnl += gross_pnl
            sym_ref = pos.symbol
            if symbol in self.open_positions:
                del self.open_positions[symbol]

        margin_used = pos.position_size / LEVERAGE
        pnl_pct = (pnl_net / margin_used * 100) if margin_used > 0 else 0.0
        js = journal_symbol(sym_ref)
        record_trade(
            timestamp=exit_time,
            symbol=js,
            action="SELL",
            execution_price=exit_price,
            quantity=pos.position_size / exit_price,
            ml_confidence_at_entry=pos.ml_confidence,
            sentiment_score_at_entry=pos.sentiment_score,
            exit_reason=exit_reason_code,
            pnl_usdt=pnl_net,
            pnl_percent=pnl_pct,
            idempotency_key=f"SELL:{js}:ticket:{mt5_ticket or pos.trade_id}:{exit_reason_code}",
            journal_entry_price=pos.entry_price,
            journal_exit_price=exit_price,
        )
        self.save_state()

        if telegram_after:
            asyncio.create_task(
                send_telegram_alert(
                    _build_trade_report(
                        sym=sym_ref,
                        pos=pos,
                        exit_price=exit_price,
                        exit_time=exit_time,
                        gross_pnl=gross_pnl,
                        exit_reason_code=exit_reason_code,
                        current_balance=self._risk.balance,
                    )
                )
            )

    async def _finalize_if_broker_already_closed(
        self,
        symbol: str,
        pos: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason_code: str,
    ) -> bool:
        """If MT5 has no position but deals show our ticket closed, sync books."""
        if not _MT5_AVAILABLE:
            return False
        mt5_ticket = getattr(pos, "mt5_position_ticket", None)
        if not isinstance(mt5_ticket, int):
            return False
        economics = self._resolve_mt5_close_economics(mt5_ticket)
        if economics is None:
            return False
        mt5_sym = self._resolve_symbol(symbol)
        if mt5_sym is None:
            return False
        tick = mt5.symbol_info_tick(mt5_sym)
        exit_px = float(tick.bid) if tick and tick.bid else exit_price
        price_change_pct = (exit_px - pos.entry_price) / pos.entry_price
        gross_pnl = price_change_pct * pos.position_size
        code = exit_reason_code if exit_reason_code else "BROKER_CLOSED"
        await self._apply_mt5_closed_bookkeeping(
            symbol,
            pos,
            exit_px,
            exit_time,
            gross_pnl,
            code,
            telegram_after=False,
        )
        logger.info(
            "[MT5] Posición ya cerrada en broker — libro sincronizado sym=%s ticket=%s",
            symbol,
            mt5_ticket,
        )
        return True

    async def _reconcile_ghost_position(self, sym: str) -> None:
        """Ghost = local open but MT5 has no position; close DB + notify."""
        if self._shutting_down:
            return
        if not self._mt5_terminal_connected():
            logger.warning("[MT5 SYNC] Terminal disconnected — skip ghost reconcile for %s.", sym)
            return
        async with self._positions_lock:
            pos = self.open_positions.get(sym)
        if pos is None:
            return
        mt5_sym = self._resolve_symbol(sym)
        exit_px = float(pos.entry_price)
        if mt5_sym and _MT5_AVAILABLE:
            tick = mt5.symbol_info_tick(mt5_sym)
            if tick and tick.bid:
                exit_px = float(tick.bid)
        ts = datetime.now(tz=timezone.utc)
        gross = (exit_px - pos.entry_price) / pos.entry_price * pos.position_size
        ticket = getattr(pos, "mt5_position_ticket", None)
        logger.info(
            "[MT5 SYNC] 👻 Reconciliando ghost sym=%s ticket=%s gross≈%.4f",
            sym,
            ticket,
            gross,
        )
        await self._apply_mt5_closed_bookkeeping(
            sym,
            pos,
            exit_px,
            ts,
            gross,
            "GHOST_SYNC",
            telegram_after=True,
        )

    async def _try_adopt_mt5_position(self, sym: str) -> bool:
        """Pull broker-side LONG position into ``open_positions`` + DB + journal.

        Called when MT5 shows our *magic* position but local state has no entry.
        """
        if not _MT5_AVAILABLE:
            return False
        mt5_sym = self._resolve_symbol(sym)
        if mt5_sym is None:
            return False
        raw = mt5.positions_get(symbol=mt5_sym)
        if not raw:
            return False
        p = None
        for cand in raw:
            if cand.magic != self._magic:
                continue
            if cand.type != mt5.POSITION_TYPE_BUY:
                logger.warning(
                    "[MT5 SYNC] Cannot adopt %s — broker position is not BUY.", sym
                )
                return False
            p = cand
            break
        if p is None:
            return False

        info = mt5.symbol_info(mt5_sym)
        digits: int = info.digits if info else 5
        contract_size = float(info.trade_contract_size) if info else 1.0

        entry = float(p.price_open)
        volume = float(p.volume)
        position_size = volume * contract_size * entry

        sl_raw = float(p.sl or 0.0)
        thresholds = get_execution_thresholds()
        cfg_a = get_symbol_config(sym)
        fixed_ad = min(float(cfg_a["fixed_sl_pct"]), _SL_CAP_VALUE)
        if sl_raw > 0.0:
            stop_loss_price = self._normalize_price(sl_raw, digits)
        else:
            stop_loss_price = self._normalize_price(
                entry * (1.0 - fixed_ad), digits
            )

        try:
            ts = datetime.fromtimestamp(int(p.time), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            ts = datetime.now(tz=timezone.utc)

        try:
            trade_id = await self._db.insert_open_trade(
                symbol=sym,
                entry_price=entry,
                position_size=position_size,
                entry_time=ts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "[MT5 SYNC] adopt insert_open_trade failed sym=%s — %s", sym, exc
            )
            return False

        peak = max(entry, float(p.price_current))
        act_px_ad = entry * (1.0 + thresholds.activation_pct)
        op = OpenPosition(
            trade_id=trade_id,
            symbol=sym,
            entry_time=ts,
            entry_price=entry,
            position_size=position_size,
            sl_price=stop_loss_price,
            activation_price=act_px_ad,
            trailing_distance_pct=thresholds.trailing_distance_pct,
            peak_price=peak,
            sl_pct=fixed_ad,
            activation_pct=thresholds.activation_pct,
            stop_loss_price=stop_loss_price,
            atr_trailing_distance=0.0,
            ml_confidence=0.0,
            sentiment_score=0.0,
            mt5_position_ticket=int(p.ticket),
            current_stop_loss=stop_loss_price,
        )
        if sl_raw > 0.0:
            op.last_broker_sl_synced = stop_loss_price
        tp_raw = float(p.tp or 0.0)
        if tp_raw > 0.0:
            op.last_broker_tp_synced = self._normalize_price(tp_raw, digits)

        async with self._positions_lock:
            if sym in self.open_positions:
                logger.debug("_try_adopt_mt5_position: %s already tracked — skipping.", sym)
                return False
            self.open_positions[sym] = op
            risk_usd = position_size * fixed_ad
            self._risk.register_open(risk_usd=risk_usd)

        js = journal_symbol(sym)
        record_trade(
            timestamp=ts,
            symbol=js,
            action="BUY",
            execution_price=entry,
            quantity=position_size / entry if entry > 0 else 0.0,
            ml_confidence_at_entry=0.0,
            sentiment_score_at_entry=0.0,
            idempotency_key=f"BUY:{js}:ticket:{int(p.ticket)}",
        )
        self.save_state()

        logger.info(
            "[MT5 SYNC] ✅ Adopted orphan MT5 position sym=%s ticket=%s entry=%.5f sl=%.5f",
            sym,
            p.ticket,
            entry,
            stop_loss_price,
        )
        return True

    # ------------------------------------------------------------------
    # _close_position override
    # ------------------------------------------------------------------

    async def _close_position(
        self,
        symbol: str,
        exit_price: float,
        arg3: Any = None,
        arg4: Any = None,
        arg5: str = "",
    ) -> Any:
        """Dispatch broker close + bookkeeping.

        * **PaperExecutor API** — ``(symbol, exit_price, reason: str)`` from
          :meth:`~execution.paper_executor.PaperExecutor.check_and_close` and
          :meth:`~execution.paper_executor.PaperExecutor.check_ml_exit`.
        * **Internal API** — ``(symbol, exit_price, exit_time, pnl, exit_reason_code)``.
        """
        if isinstance(arg3, str) and arg4 is None:
            exit_time = datetime.now(tz=timezone.utc)
            pos = self.open_positions.get(symbol)
            if pos is None:
                logger.warning(
                    "_close_position (paper API): no open position for %s — skipping.",
                    symbol,
                )
                return None
            gross_est = (
                (exit_price - float(pos.entry_price))
                / float(pos.entry_price)
                * float(pos.position_size)
            )
            ok = await self._close_position_live(
                symbol, exit_price, exit_time, gross_est, arg3
            )
            return gross_est if ok else None

        if isinstance(arg3, datetime):
            return await self._close_position_live(
                symbol,
                exit_price,
                arg3,
                float(arg4 or 0.0),
                arg5,
            )

        logger.error(
            "_close_position: invalid signature for %s (arg3=%r arg4=%r).",
            symbol,
            arg3,
            arg4,
        )
        return False

    async def _close_position_live(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        pnl: float,
        exit_reason_code: str = "",
    ) -> bool:
        """Close a live MT5 position and update the paper book.

        When *live=True* a market SELL order is sent via ``mt5.order_send``
        using the SELL-at-BID price.  The broker fills it immediately at the
        best available bid, which is the correct reference price for a LONG
        exit.

        If the MT5 order is rejected the position is still removed from local
        state and the paper PnL is updated — this mirrors the Binance fallback
        behaviour and prevents ghost positions from accumulating.
        """
        pos = self.open_positions.get(symbol)
        if pos is None:
            logger.warning(
                "_close_position called for %s but no open position found – skipping.",
                symbol,
            )
            return False

        # ── Live MT5 order ─────────────────────────────────────────────────
        close_error: str | None = None
        if self._live:
            if not _MT5_AVAILABLE:
                logger.error(
                    "MT5 live mode requested but MetaTrader5 library is not installed. "
                    "Keeping position with close_pending for retry."
                )
                close_error = "mt5_library_unavailable"
            else:
                mt5_sym = self._resolve_symbol(symbol)
                if mt5_sym is None:
                    logger.error(
                        "[MT5] Cannot send close order for '%s' – symbol not in SYMBOL_MAP. "
                        "keeping local position for retry.",
                        symbol,
                    )
                    close_error = "mt5_symbol_unresolved"
                else:
                    # Find all MT5 positions for this symbol tagged with our magic.
                    mt5_positions = mt5.positions_get(symbol=mt5_sym)
                    if mt5_positions:
                        closed_count = 0
                        for mt5_pos in mt5_positions:
                            if mt5_pos.magic != self._magic:
                                continue

                            # Use the current bid price for a LONG close (SELL at bid).
                            tick = mt5.symbol_info_tick(mt5_sym)
                            bid_price = tick.bid if tick else exit_price

                            request = self._build_sell_request(
                                symbol=mt5_sym,
                                lots=mt5_pos.volume,
                                price=bid_price,
                                position_id=mt5_pos.ticket,
                            )
                            result = await self._send_order_with_retry(request)
                            
                            if result is None:
                                logger.warning("[MT5] Close order failed. Attempting to clear stops and retry...")
                                clear_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": mt5_sym,
                                    "position": mt5_pos.ticket,
                                    "sl": 0.0,
                                    "tp": 0.0,
                                }
                                await self._send_order_with_retry(clear_request)
                                result = await self._send_order_with_retry(request)

                            if result is not None:
                                closed_count += 1
                                logger.info(
                                    "[MT5] Position closed: ticket=%d, symbol=%s.",
                                    mt5_pos.ticket,
                                    symbol,
                                )
                            else:
                                logger.error(
                                    "MT5 close order rejected for %s (ticket=%d) – "
                                    "keeping local position for retry.",
                                    symbol,
                                    mt5_pos.ticket,
                                )
                                close_error = "mt5_close_rejected"

                        if closed_count == 0:
                            logger.warning(
                                "[MT5] No positions closed for %s with magic=%d.",
                                symbol,
                                self._magic,
                            )
                            close_error = close_error or "mt5_no_positions_closed"
                        elif closed_count > 1:
                            logger.warning(
                                "[MT5] Closed %d positions for %s (expected 1) – "
                                "possible overlap after restart.",
                                closed_count,
                                symbol,
                            )
                    else:
                        # Broker may have closed first (SL/TP server-side).
                        synced = await self._finalize_if_broker_already_closed(
                            symbol, pos, exit_price, exit_time, exit_reason_code
                        )
                        if synced:
                            return True
                        logger.warning(
                            "[MT5] No open position found for %s with magic=%d – "
                            "keeping local position for retry.",
                            symbol,
                            self._magic,
                        )
                        close_error = "mt5_position_not_found"

        if close_error is not None:
            pos.close_pending = True
            pos.last_close_error = close_error
            self.save_state()
            return False

        await self._apply_mt5_closed_bookkeeping(
            symbol,
            pos,
            exit_price,
            exit_time,
            pnl,
            exit_reason_code,
            telegram_after=False,
        )
        return True

    def _resolve_mt5_close_economics(self, position_ticket: int) -> dict[str, float] | None:
        """Fetch broker-realized close economics from MT5 deals history."""
        if not _MT5_AVAILABLE:
            return None
        date_to = datetime.now(tz=timezone.utc)
        date_from = date_to - timedelta(days=14)
        try:
            deals = mt5.history_deals_get(date_from, date_to, position=position_ticket)
        except Exception:  # noqa: BLE001
            return None
        if not deals:
            return None
        close_deals = []
        for d in deals:
            pid = getattr(d, "position_id", None)
            if pid is not None and int(pid) != int(position_ticket):
                continue
            entry = getattr(d, "entry", None)
            if entry == mt5.DEAL_ENTRY_OUT:
                close_deals.append(d)
        # Never sum the whole deal list as fallback: without OUT filtering MT5 can
        # return IN+OUT rows or broader history — summing everything inflated PnL
        # (e.g. journal +358 USDT ghosts). Fall back to book formula via None.
        if not close_deals:
            return None
        pnl_total = 0.0
        commission_total = 0.0
        swap_total = 0.0
        fee_total = 0.0
        for d in close_deals:
            pnl_total += float(getattr(d, "profit", 0.0) or 0.0)
            commission_total += float(getattr(d, "commission", 0.0) or 0.0)
            swap_total += float(getattr(d, "swap", 0.0) or 0.0)
            fee_total += float(getattr(d, "fee", 0.0) or 0.0)
        pnl_net = pnl_total + commission_total + swap_total + fee_total
        return {
            "pnl_net": pnl_net,
            "commission": commission_total,
            "swap": swap_total,
            "fee": fee_total,
        }

    # ------------------------------------------------------------------
    # sync_positions_with_exchange override
    # ------------------------------------------------------------------

    async def _mt5_positions_get_retry(self) -> Any | None:
        """Call ``mt5.positions_get()`` from a worker thread with retries.

        ``None`` means MT5 could not return a snapshot (transient terminal/API).
        """
        if not _MT5_AVAILABLE:
            return None
        delay = 0.04
        for attempt in range(4):
            raw = await asyncio.to_thread(mt5.positions_get)
            if raw is not None:
                return raw
            await asyncio.sleep(delay * (attempt + 1))
        return None

    async def _mt5_positions_fallback_symbols(self) -> list[Any]:
        """When global ``positions_get()`` fails, merge per known local symbols.

        Prevents total sync starvation when the aggregate call returns ``None``
        (seen under terminal stress): we still see broker tickets for symbols we track.
        """
        merged: list[Any] = []
        seen_tickets: set[int] = set()
        async with self._positions_lock:
            syms = list(self.open_positions.keys())
        for sym in syms:
            mt5_sym = self._resolve_symbol(sym)
            if not mt5_sym:
                continue
            raw = await asyncio.to_thread(lambda s=mt5_sym: mt5.positions_get(symbol=s))
            if not raw:
                continue
            for p in raw:
                tid = int(getattr(p, "ticket", 0) or 0)
                if tid and tid not in seen_tickets:
                    seen_tickets.add(tid)
                    merged.append(p)
        return merged

    async def _reconcile_stale_tickets(self, confirmations_required: int | None = None) -> None:
        """Remove locals whose MT5 *ticket* no longer exists.

        Symbol-set reconciliation misses the case: old ticket closed on broker,
        new position opened same symbol (still ``live_symbols`` contains sym).
        """
        if not _MT5_AVAILABLE:
            return
        if self._shutting_down or not self._mt5_terminal_connected():
            return
        async with self._positions_lock:
            snapshot = [(s, p) for s, p in self.open_positions.items()]
        for sym, pos in snapshot:
            tid = getattr(pos, "mt5_position_ticket", None)
            if not isinstance(tid, int) or tid <= 0:
                continue
            gk = self._ghost_key(sym, tid)
            pins = await asyncio.to_thread(lambda t=tid: mt5.positions_get(ticket=t))
            if pins is None:
                logger.warning(
                    "[MT5 SYNC] positions_get(ticket=%s) returned None — deferring ghost for %s.",
                    tid,
                    sym,
                )
                continue
            if len(pins) == 0:
                needed = (
                    self._ghost_min_confirmations
                    if confirmations_required is None
                    else max(1, confirmations_required)
                )
                if not self._ghost_mark_missing(gk, confirmations_required=needed):
                    logger.warning(
                        "[MT5 SYNC] Ticket %s missing for %s (%d/%d) — waiting confirmation.",
                        tid,
                        sym,
                        self._ghost_missing_counts.get(gk, 0),
                        needed,
                    )
                    continue
                logger.info(
                    "[MT5 SYNC] Ticket %s closed on broker — ghost reconcile %s",
                    tid,
                    sym,
                )
                await self._reconcile_ghost_position(sym)
            else:
                self._ghost_reset(gk)

    async def sync_positions_with_exchange(self, confirmations_required: int | None = None) -> int:
        """Re-sync local state against live MT5 positions.

        Detects *ghost* positions (in local memory but not on MT5) and removes
        them.  Positions open on MT5 with our *magic* but missing locally are
        **adopted** (LONG only) into ``open_positions`` + DB.

        Also reconciles by **position ticket** so a stale local row does not
        survive when a new trade on the same symbol replaced the broker ticket.

        In pure paper mode (``live=False``) returns the local position count
        unchanged.
        """
        if not self._live:
            return len(self.open_positions)
        if self._shutting_down:
            return len(self.open_positions)

        if not _MT5_AVAILABLE:
            logger.warning("[MT5 SYNC] MetaTrader5 not available – skipping sync.")
            return len(self.open_positions)
        if not self._mt5_terminal_connected():
            logger.warning("[MT5 SYNC] MT5 disconnected — sync skipped (no ghost actions).")
            return len(self.open_positions)

        needed_confirmations = (
            self._ghost_min_confirmations
            if confirmations_required is None
            else max(1, confirmations_required)
        )
        mt5_positions_raw = await self._mt5_positions_get_retry()
        if mt5_positions_raw is None:
            logger.warning(
                "[MT5 SYNC] mt5.positions_get() still None after retries — "
                "using per-symbol fallback + ticket checks.",
            )
            mt5_positions_raw = await self._mt5_positions_fallback_symbols()
            if not mt5_positions_raw:
                await self._reconcile_stale_tickets(confirmations_required=needed_confirmations)
                async with self._positions_lock:
                    actual = len(self.open_positions)
                    if self._risk.open_count != actual:
                        self._risk.sync_open_count(actual)
                return actual

        # Convert broker symbols (e.g. BTCUSD-T) to local symbols (BTC/USDT)
        # so comparisons against self.open_positions keys are consistent.
        live_symbols: set[str] = {
            self._local_symbol_from_broker(p.symbol)
            for p in mt5_positions_raw
            if p.magic == self._magic
        }

        # Ghost detection — broker closed without bot bookkeeping (SL/TP / manual)
        async with self._positions_lock:
            snapshot = list(self.open_positions.items())
            ghost_symbols: list[str] = []
            for sym, pos in snapshot:
                if sym in live_symbols:
                    self._ghost_reset(self._ghost_key(sym, getattr(pos, "mt5_position_ticket", None)))
                    continue
                gk = self._ghost_key(sym, getattr(pos, "mt5_position_ticket", None))
                if self._ghost_mark_missing(gk, confirmations_required=needed_confirmations):
                    ghost_symbols.append(sym)
                else:
                    logger.warning(
                        "[MT5 SYNC] Ghost candidate %s (%d/%d) — waiting confirmation.",
                        sym,
                        self._ghost_missing_counts.get(gk, 0),
                        needed_confirmations,
                    )
        for sym in ghost_symbols:
            await self._reconcile_ghost_position(sym)

        # Same symbol, new broker ticket — remove stale local ticket row
        await self._reconcile_stale_tickets(confirmations_required=needed_confirmations)

        # Broker-only positions → import into local book (LONG + our magic)
        async with self._positions_lock:
            occupied = set(self.open_positions.keys()) | self._pending_symbols
            untracked = [sym for sym in live_symbols if sym not in occupied]
        for sym in untracked:
            adopted = await self._try_adopt_mt5_position(sym)
            if not adopted:
                logger.warning(
                    "[MT5 SYNC] ⚠️ Position for %s is open on MT5 but not tracked locally "
                    "(adoption failed — check logs).",
                    sym,
                )

        # Enforce counter invariant after sync
        async with self._positions_lock:
            actual = len(self.open_positions)
            if self._risk.open_count != actual:
                logger.warning(
                    "[MT5 SYNC] Counter drift detected: risk.open_count=%d actual=%d — correcting.",
                    self._risk.open_count,
                    actual,
                )
                self._risk.sync_open_count(actual)
            self._risk.recalc_total_risk(self.open_positions)

        return len(live_symbols)

    async def _verify_initial_sl_synced(
        self,
        sym: str,
        pos: OpenPosition,
        ticket: int,
    ) -> None:
        """Verify broker SL right after BUY; force modify when missing/mismatched."""
        if not self._live or not _MT5_AVAILABLE:
            return
        mt5_sym = self._resolve_symbol(sym)
        if mt5_sym is None:
            return
        info = mt5.symbol_info(mt5_sym)
        digits = int(info.digits) if info and info.digits else 5
        expected = self._normalize_price(float(pos.stop_loss_price), digits)
        for attempt in range(1, 4):
            pins = mt5.positions_get(ticket=ticket)
            if not pins:
                await asyncio.sleep(0.12 * attempt)
                continue
            actual = float(getattr(pins[0], "sl", 0.0) or 0.0)
            if actual > 0.0 and abs(actual - expected) <= (10 ** (-max(digits - 1, 1))):
                pos.last_broker_sl_synced = expected
                pos.last_mt5_modify_mono = time.monotonic()
                return
            ok = await self.modify_position(ticket=ticket, new_sl=expected, new_tp=0.0, symbol=sym)
            if ok:
                pos.last_broker_sl_synced = expected
                pos.last_mt5_modify_mono = time.monotonic()
                logger.info(
                    "[MT5] Initial SL synced by force modify sym=%s ticket=%d sl=%.5f",
                    sym,
                    ticket,
                    expected,
                )
                return
            await asyncio.sleep(0.12 * attempt)
        logger.error(
            "[MT5] Initial SL verification failed sym=%s ticket=%d expected_sl=%.5f",
            sym,
            ticket,
            expected,
        )

    # ------------------------------------------------------------------
    # Position modification and explicit close
    # ------------------------------------------------------------------

    async def modify_position(
        self,
        ticket: int,
        new_sl: float,
        new_tp: float = 0.0,
        symbol: str | None = None,
    ) -> bool:
        """Modify the Stop Loss (and optionally Take Profit) of an open MT5 position.

        Used by the trailing-stop logic to push the SL level upward as price
        moves in favour of the trade.  Prices are normalised to the broker's
        required number of decimal digits before being submitted.

        Parameters
        ----------
        ticket:
            MT5 position ticket number.
        new_sl:
            New stop-loss price (absolute, not a distance).
        new_tp:
            New take-profit price.  Pass ``0.0`` to leave TP unchanged /
            unset (broker default).
        symbol:
            Internal symbol name used only for price-digit lookup.  When
            *None* the executor's default ``self.symbol`` is used.

        Returns
        -------
        bool
            ``True`` if the modification was accepted by MT5, ``False``
            otherwise (including paper-mode where no MT5 call is made).
        """
        if not self._live:
            logger.debug("modify_position: not in live mode – skipping MT5 call.")
            return True

        if not _MT5_AVAILABLE:
            logger.error("modify_position: MetaTrader5 library is not installed.")
            return False

        sym = symbol or self.symbol
        mt5_sym = self._resolve_symbol(sym)
        if mt5_sym is None:
            return False

        info = mt5.symbol_info(mt5_sym)
        digits: int = info.digits if info else 5
        norm_sl = self._normalize_price(new_sl, digits) if new_sl else 0.0
        norm_tp = self._normalize_price(new_tp, digits) if new_tp else 0.0

        request: dict = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": mt5_sym,
            "position": ticket,
            "sl": norm_sl,
            "tp": norm_tp,
        }
        result = await self._send_order_with_retry(request)
        return result is not None

    async def _resolve_position_ticket_after_buy(self, mt5_sym: str, result: Any) -> int | None:
        """Return the hedging-mode position ticket after a successful market BUY."""
        if not _MT5_AVAILABLE:
            return None
        pos_id = int(getattr(result, "position", 0) or 0)
        if pos_id > 0:
            chk = mt5.positions_get(ticket=pos_id)
            if chk:
                return pos_id
        await asyncio.sleep(0.05)
        raw = mt5.positions_get(symbol=mt5_sym)
        if raw is None:
            return None
        ours = [
            p for p in raw
            if p.magic == self._magic and p.type == mt5.POSITION_TYPE_BUY
        ]
        if not ours:
            logger.warning(
                "[MT5] Could not resolve position ticket after BUY on %s (magic=%d).",
                mt5_sym,
                self._magic,
            )
            return None
        return int(max(ours, key=lambda p: p.ticket).ticket)

    def _find_magic_long_ticket(self, sym: str) -> int | None:
        """Locate a BUY position ticket for *sym* tagged with :attr:`_magic`."""
        if not _MT5_AVAILABLE:
            return None
        mt5_sym = self._resolve_symbol(sym)
        if mt5_sym is None:
            return None
        raw = mt5.positions_get(symbol=mt5_sym)
        if raw is None:
            return None
        ours = [
            p for p in raw
            if p.magic == self._magic and p.type == mt5.POSITION_TYPE_BUY
        ]
        if not ours:
            return None
        return int(max(ours, key=lambda p: p.ticket).ticket)

    def _dynamic_tp_for_long(self, pos: OpenPosition) -> float | None:
        """Broker TP target; shared formula in :func:`~execution.paper_executor.compute_dynamic_tp_hint`."""
        return compute_dynamic_tp_hint(pos)

    async def _sync_exchange_stops(
        self,
        sym: str,
        pos: OpenPosition,
        current_price: float,
        current_atr: float | None,
    ) -> None:
        """Push ratcheted SL and optional dynamic TP to MT5 (``TRADE_ACTION_SLTP``)."""
        if not self._live or not _MT5_AVAILABLE:
            return

        ticket = pos.mt5_position_ticket or self._find_magic_long_ticket(sym)
        if ticket is None:
            return
        pos.mt5_position_ticket = ticket

        mt5_sym = self._resolve_symbol(sym)
        if mt5_sym is None:
            return

        now = time.monotonic()
        if now - pos.last_mt5_modify_mono < 0.12:
            return

        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None or tick.ask <= 0.0 or tick.bid <= 0.0:
            return
        ask = float(tick.ask)
        bid = float(tick.bid)

        pins = mt5.positions_get(ticket=ticket)
        if not pins:
            return
        bpos = pins[0]
        cur_sl_br = float(bpos.sl) if bpos.sl else 0.0
        cur_tp_br = float(bpos.tp) if bpos.tp else 0.0

        info = mt5.symbol_info(mt5_sym)
        if info is None:
            return
        point = float(info.point) if info.point else 0.01
        digits = int(info.digits) if info.digits else 5
        min_step = max(point * 4.0, 10.0 ** (-max(digits - 1, 1)))

        send_sl = self._normalize_price(pos.current_stop_loss, digits)
        sl_clamped, sl_adjusted = self._clamp_stop_loss_buy(mt5_sym, send_sl, ask, tick, digits)
        if sl_clamped is None:
            return
        send_sl = sl_clamped
        if sl_adjusted:
            pos.current_stop_loss = send_sl

        cand_tp = self._dynamic_tp_for_long(pos)
        send_tp = cur_tp_br
        if cand_tp is not None:
            cn = self._normalize_price(cand_tp, digits)
            need = ask + min_step
            if cn > need and self._validate_stops(
                mt5_sym, ask, send_sl, cn, is_buy=True, tick=tick
            ):
                send_tp = max(cur_tp_br, cn)
            else:
                if cn <= need:
                    self._tp_sync_trace(
                        sym,
                        "TP_TOO_CLOSE_TO_ASK",
                        f"cand_tp={cn:.5f} need > {need:.5f} (ask={ask:.5f} min_step={min_step:.5f})",
                    )
                else:
                    self._tp_sync_trace(
                        sym,
                        "TP_FAIL_BROKER_RULES",
                        f"cand_tp={cn:.5f} sl={send_sl:.5f} check logs for [MT5 VALIDATION]",
                    )
        else:
            self._tp_sync_trace(
                sym,
                "NO_TP_HINT",
                f"peak={pos.peak_price:.5f} entry={pos.entry_price:.5f} (need peak > entry for dynamic TP)",
            )

        if not self._validate_stops(mt5_sym, ask, send_sl, 0.0, is_buy=True, tick=tick):
            self._tp_sync_trace(
                sym,
                "SL_SYNC_VALIDATE_FAIL",
                f"send_sl={send_sl:.5f} ask={ask:.5f} bid={bid:.5f}",
            )
            return

        sl_delta = abs(send_sl - cur_sl_br)
        tp_delta = abs(send_tp - cur_tp_br) if send_tp > 0.0 or cur_tp_br > 0.0 else 0.0
        try:
            skip_mult = float(
                os.environ.get("MT5_SLTP_SKIP_THRESHOLD_MULT", "0.35").strip()
                or "0.35"
            )
        except ValueError:
            skip_mult = 0.35
        thr = min_step * max(0.05, min(skip_mult, 1.0))
        # ETH/SOL: trailing moves often < default thr vs broker SL → no MT5 modify.
        # Lower MT5_SLTP_SKIP_THRESHOLD_MULT (e.g. 0.12) if alts must track tighter.
        if sl_delta < thr and tp_delta < thr:
            self._tp_sync_trace(
                sym,
                "SKIP_NO_CHANGE",
                f"sl_delta={sl_delta:.6f} tp_delta={tp_delta:.6f} thr={thr:.6f} broker_sl={cur_sl_br:.5f} broker_tp={cur_tp_br:.5f} target_tp={send_tp:.5f}",
            )
            return

        ok = await self.modify_position(
            ticket,
            new_sl=send_sl,
            new_tp=send_tp if send_tp > 0.0 else 0.0,
            symbol=sym,
        )
        if not ok:
            self._tp_sync_trace(
                sym,
                "MODIFY_REJECTED",
                f"wanted sl={send_sl:.5f} tp={send_tp:.5f} (see MT5 RETCODE lines above)",
            )
        if ok:
            pos.last_broker_sl_synced = send_sl
            pos.last_broker_tp_synced = send_tp
            pos.last_mt5_modify_mono = now
            logger.debug(
                "[MT5 SYNC] SL/TP pushed  sym=%s  ticket=%d  sl=%.5f  tp=%.5f",
                sym,
                ticket,
                send_sl,
                send_tp,
            )

    async def close_position_by_ticket(self, ticket: int) -> bool:
        """Close an open MT5 position by its ticket number.

        Identifies whether the position is a BUY or SELL and sends the
        appropriate opposite order.  The close price is fetched live from the
        MT5 terminal (bid for BUY positions, ask for SELL positions) to ensure
        accurate fill prices.

        Parameters
        ----------
        ticket:
            MT5 position ticket to close.

        Returns
        -------
        bool
            ``True`` on successful close, ``False`` on failure or paper mode.
        """
        if not self._live:
            logger.debug("close_position_by_ticket: not in live mode – skipping.")
            return True

        if not _MT5_AVAILABLE:
            logger.error("close_position_by_ticket: MetaTrader5 library is not installed.")
            return False

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(
                "close_position_by_ticket: no MT5 position found for ticket=%d.", ticket
            )
            return False

        pos = positions[0]
        mt5_sym = pos.symbol
        tick = mt5.symbol_info_tick(mt5_sym)

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid if tick else pos.price_current
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask if tick else pos.price_current

        request: dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": "ClawdBot CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = await self._send_order_with_retry(request)
        return result is not None

    # ------------------------------------------------------------------
    # Market data extraction
    # ------------------------------------------------------------------

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: int = TIMEFRAME_M15,
        count: int = 100,
        start_pos: int = 0,
    ) -> "pd.DataFrame | None":
        """Fetch OHLCV candles from the MT5 terminal as a pandas DataFrame.

        The synchronous ``mt5.copy_rates_from_pos`` call is offloaded to a
        thread-pool executor via :func:`asyncio.get_event_loop().run_in_executor`
        so it does not block the event loop while waiting for the MT5 terminal.

        Parameters
        ----------
        symbol:
            Internal symbol name (e.g. ``"BTC/USDT"``).  Translated via
            :meth:`_resolve_symbol` before the MT5 call.
        timeframe:
            MT5 timeframe constant (e.g. :data:`TIMEFRAME_M15`,
            :data:`TIMEFRAME_H1`).  Defaults to 15-minute candles.
        count:
            Number of bars to fetch (most recent first from *start_pos*).
        start_pos:
            Starting bar index (0 = the most recent/current bar).

        Returns
        -------
        pd.DataFrame | None
            DataFrame with columns ``time``, ``open``, ``high``, ``low``,
            ``close``, ``tick_volume``, ``spread``, ``real_volume``.
            Returns ``None`` when MT5 is unavailable or the symbol cannot be
            resolved.
        """
        if not _MT5_AVAILABLE:
            logger.warning("fetch_candles: MetaTrader5 library is not installed.")
            return None

        if not _PANDAS_AVAILABLE:
            logger.warning("fetch_candles: pandas is not installed.")
            return None

        mt5_sym = self._resolve_symbol(symbol)
        if mt5_sym is None:
            return None

        loop = asyncio.get_event_loop()
        try:
            rates = await loop.run_in_executor(
                None,
                mt5.copy_rates_from_pos,
                mt5_sym,
                timeframe,
                start_pos,
                count,
            )
        except (OSError, TimeoutError, AttributeError, TypeError) as exc:
            logger.error(
                "fetch_candles: mt5.copy_rates_from_pos failed for %s: %s",
                mt5_sym,
                exc,
            )
            return None

        if rates is None or len(rates) == 0:
            logger.warning(
                "fetch_candles: mt5.copy_rates_from_pos returned no data for %s "
                "(timeframe=%d, count=%d).",
                mt5_sym,
                timeframe,
                count,
            )
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    async def fetch_tick(self, symbol: str) -> dict | None:
        """Return the latest bid and ask prices for *symbol*.

        The synchronous ``mt5.symbol_info_tick`` call is offloaded to a
        thread-pool executor via :func:`asyncio.get_event_loop().run_in_executor`
        so it does not block the event loop.

        Parameters
        ----------
        symbol:
            Internal symbol name (e.g. ``"BTC/USDT"``).

        Returns
        -------
        dict | None
            Dictionary with keys ``"symbol"``, ``"bid"``, ``"ask"``,
            ``"last"``, and ``"time"`` (UTC datetime), or ``None`` when MT5
            is unavailable or the tick cannot be fetched.
        """
        if not _MT5_AVAILABLE:
            logger.warning("fetch_tick: MetaTrader5 library is not installed.")
            return None

        mt5_sym = self._resolve_symbol(symbol)
        if mt5_sym is None:
            return None

        loop = asyncio.get_event_loop()
        try:
            tick = await loop.run_in_executor(None, mt5.symbol_info_tick, mt5_sym)
        except (OSError, TimeoutError, AttributeError, TypeError) as exc:
            logger.error("fetch_tick: mt5.symbol_info_tick failed for %s: %s", mt5_sym, exc)
            return None

        if tick is None:
            logger.warning("fetch_tick: mt5.symbol_info_tick('%s') returned None.", mt5_sym)
            return None

        return {
            "symbol": mt5_sym,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        }

    def get_open_positions(
        self,
        symbol: str | None = None,
        *,
        include_foreign: bool = False,
    ) -> list[dict]:
        """Return open MT5 positions opened by this bot instance.

        By default filters by :attr:`_magic` (bot positions only). Set
        ``include_foreign=True`` to include manual/other-EA broker positions.

        Parameters
        ----------
        symbol:
            Internal symbol name (e.g. ``"BTC/USDT"``).  When provided only
            positions for that symbol are returned; when *None* all open
            positions tagged with :attr:`_magic` are returned.

        Returns
        -------
        list[dict]
            Each entry contains:
            ``ticket``, ``symbol`` (broker name), ``type`` (``"BUY"``/``"SELL"``),
            ``volume``, ``price_open``, ``price_current``, ``sl``, ``tp``,
            ``profit``, ``magic``.
            Returns an empty list when MT5 is unavailable.
        """
        if not _MT5_AVAILABLE:
            logger.warning("get_open_positions: MetaTrader5 library is not installed.")
            return []

        if symbol is not None:
            mt5_sym = self._resolve_symbol(symbol)
            if mt5_sym is None:
                return []
            raw = mt5.positions_get(symbol=mt5_sym)
        else:
            raw = mt5.positions_get()

        if raw is None:
            return []

        result: list[dict] = []
        for pos in raw:
            if not include_foreign and pos.magic != self._magic:
                continue  # Ignore positions not opened by this bot instance.
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "magic": pos.magic,
            })
        return result
