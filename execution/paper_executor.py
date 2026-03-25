"""
execution.paper_executor
~~~~~~~~~~~~~~~~~~~~~~~~

Simulates order execution (paper trading) and, when a live Binance Futures
exchange client is supplied, submits real market orders via the ccxt API.

Opens and closes paper positions based on trailing stop thresholds that are
computed **dynamically** at position-open time from the AI sentiment score
(see :func:`~risk.risk_manager.get_dynamic_thresholds`).  Each completed
trade is persisted to the ``trades_history`` table via the database manager.

A BUY position is protected as follows:

* **Initial Stop Loss** (SL): price falls ≥ ``sl_pct`` below entry price.
  This hard floor is active from the moment the trade opens.  The value is
  scaled by the sentiment multiplier: tight in low-sentiment (scalping) and
  wide in high-sentiment (swing-trading) markets.
* **Trailing Stop** (TS): once the position gains ≥ ``activation_pct`` the
  active stop loss updates dynamically to
  ``peak_price * (1 - trailing_distance_pct)``.
  The stop only moves up — it never retreats — allowing the trade to capture
  exponential crypto runs while locking in profit.

The trade exits **only** when the current price drops below the active stop
loss (initial SL before activation, trailing SL afterwards).

Binance Futures live execution
------------------------------
Pass a ccxt ``binanceusdm`` (or equivalent Binance Futures) async client as
the *exchange* constructor argument to enable live order placement.  Each
trade open will call ``exchange.create_market_buy_order`` with
``params={'leverage': LEVERAGE}`` so the exchange applies the configured
leverage.  Each close will call ``exchange.create_market_sell_order`` with
``params={'leverage': LEVERAGE, 'reduceOnly': True}`` — the ``reduceOnly``
flag bypasses Binance's minimum-notional check (error -4164) that applies to
standard close orders, allowing positions with a notional value below $100 USDT
to be fully closed.

Safety break
------------
Before opening a new position :meth:`try_open_trade` checks
:meth:`~risk.risk_manager.RiskManager.is_trading_halted`.  If the daily loss
limit (3 % of the day's opening balance) has been reached, the call is
rejected until the 24-hour halt expires.

Multi-asset support: up to ``risk_manager.max_positions`` independent
positions may be open simultaneously, one per symbol.  Attempting to open a
second position on the same symbol is silently rejected.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ccxt.base.errors import (
    AuthenticationError as CcxtAuthenticationError,
    InsufficientFunds as CcxtInsufficientFunds,
    NetworkError as CcxtNetworkError,
    NotSupported as CcxtNotSupported,
)

from risk.risk_manager import (
    BASE_ACTIVATION_PCT,
    BASE_SL,
    BASE_TRAILING_DISTANCE,
    DynamicThresholds,
    LEVERAGE,
    RiskManager,
    get_dynamic_thresholds,
    get_sector,
)
from utils.telegram_notifier import send_telegram_alert

if TYPE_CHECKING:
    from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Hint appended to warning messages directing users to the debug log file.
_DEBUG_LOG_HINT = "(Revisa bot_debug.log para detalles técnicos)"


# Default TTL for open positions (hours).  Set to None to disable.
_DEFAULT_TTL_HOURS: float = 12.0

# Smart-exit ML confidence multiplier for Layer 1 (Exhaustion Exit).
# The closing threshold is: min_confidence * _ML_EXIT_CONFIDENCE_FACTOR
_ML_EXIT_CONFIDENCE_FACTOR: float = 0.8

# ── ATR-based dynamic stop-loss / trailing-stop multipliers ───────────────────
# Stop Loss distance  = current_atr * ATR_SL_MULTIPLIER
ATR_SL_MULTIPLIER: float = 1.5
# Trailing Stop distance = current_atr * ATR_TRAILING_MULTIPLIER
ATR_TRAILING_MULTIPLIER: float = 1.0

# ── Post-trade reporting ───────────────────────────────────────────────────────
# Simulated taker fee per side (Binance Futures standard: 0.04 % of notional).
# Applied twice (entry + exit) to compute the round-trip fee cost.
_TAKER_FEE_RATE: float = 0.0004

# Human-readable exit-reason labels used in the Telegram trade report.
_EXIT_REASON_LABELS: dict[str, str] = {
    "SL_BASE":               "SL_BASE: El precio tocó el stop loss inicial.",
    "TRAILING_STOP":         "TRAILING_STOP: El stop subió con el precio y se ejecutó al retroceder (Ganancia protegida).",
    "SMART_EXIT_ML":         "SMART_EXIT_ML: El modelo XGBoost detectó agotamiento de tendencia.",
    "SMART_EXIT_SENTIMENT":  "SMART_EXIT_SENTIMENT: Gemini detectó un cambio brusco a sentimiento negativo.",
    "TTL_TIMEOUT":           "TTL_TIMEOUT: Tiempo máximo de exposición alcanzado.",
}


def _format_duration(total_seconds: float) -> str:
    """Return a human-readable duration string (e.g. ``'2h 15m 30s'``).

    Parameters
    ----------
    total_seconds:
        Elapsed time in seconds (non-negative).
    """
    secs = int(abs(total_seconds))  # abs() guards against clock skew / rounding
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _build_trade_report(
    sym: str,
    pos: "OpenPosition",
    exit_price: float,
    exit_time: "datetime",
    gross_pnl: float,
    exit_reason_code: str,
    current_balance: float,
) -> str:
    """Build the institutional post-trade Telegram report.

    Computes net PnL after simulated round-trip taker fees
    (``_TAKER_FEE_RATE`` applied twice: once on entry, once on exit).
    The percentage return is expressed on the *initial margin* deployed
    (``position_size / LEVERAGE``) so it reflects the actual capital at risk.

    Parameters
    ----------
    sym:
        Normalised trading pair symbol (e.g. ``"BTC/USDT"``).
    pos:
        The :class:`OpenPosition` that was just closed.
    exit_price:
        Price at which the position was closed.
    exit_time:
        UTC timestamp of the close event.
    gross_pnl:
        Unrealised PnL *before* fee deduction (in USDT).
    exit_reason_code:
        One of the keys in :data:`_EXIT_REASON_LABELS`.
    current_balance:
        Risk-manager balance *after* crediting back the closed position.

    Returns
    -------
    str
        Markdown-formatted Telegram message.
    """
    # Net PnL: deduct round-trip taker fees (assumes both legs are market/taker orders;
    # entry fee = fee_rate × notional, exit fee = fee_rate × notional).
    total_fees = pos.position_size * _TAKER_FEE_RATE * 2
    net_pnl = gross_pnl - total_fees

    # Percentage return on deployed margin (initial margin = notional / leverage)
    margin_used = pos.position_size / LEVERAGE
    if margin_used <= 0:
        logger.warning(
            "_build_trade_report: non-positive margin_used=%.4f for %s – pnl_pct set to 0",
            margin_used,
            sym,
        )
        pnl_pct = 0.0
    else:
        pnl_pct = net_pnl / margin_used * 100

    # Dynamic status based on profitability
    if net_pnl > 0:
        status_emoji = "🟢"
    else:
        status_emoji = "🔴"

    # Duration
    duration_str = _format_duration((exit_time - pos.entry_time).total_seconds())

    # Exit reason label
    exit_label = _EXIT_REASON_LABELS.get(exit_reason_code, exit_reason_code)

    # Sign prefixes for display
    pnl_sign = "+" if net_pnl >= 0 else ""
    pct_sign = "+" if pnl_pct >= 0 else ""

    return (
        f"🏁 *TRADE COMPLETED* | #{sym} {status_emoji}\n"
        f"────────────────────────\n"
        f"📈 *PnL:* {pnl_sign}{net_pnl:.4f} USDT ({pct_sign}{pnl_pct:.2f}%)\n"
        f"🚪 *Causa:* {exit_label}\n"
        f"⏱️ *Duración:* {duration_str}\n"
        f"📊 *Entrada:* {pos.entry_price:.2f} | *Salida:* {exit_price:.2f}\n"
        f"💰 *Wallet Final:* {current_balance:.2f} USDT\n"
        f"────────────────────────"
    )


@dataclass
class OpenPosition:
    """Lightweight container for a single open paper trade."""

    symbol: str
    entry_price: float
    position_size: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    trade_id: int | None = None
    # Dynamic risk thresholds computed at position open from sentiment score
    sl_pct: float = BASE_SL
    activation_pct: float = BASE_ACTIVATION_PCT
    trailing_distance_pct: float = BASE_TRAILING_DISTANCE
    # ATR-based stop loss (absolute price level; 0.0 → derive from sl_pct)
    stop_loss_price: float = 0.0
    # ATR-based trailing distance (absolute price distance; 0.0 → use pct-based)
    atr_trailing_distance: float = 0.0
    # [PRO] Trailing Stop state
    peak_price: float = field(init=False)
    trailing_stop_active: bool = False
    # [SMART EXIT] Maximum holding time in hours; None disables TTL (Layer 4).
    max_ttl_hours: float | None = _DEFAULT_TTL_HOURS

    def __post_init__(self) -> None:
        # Initialise peak_price to entry_price so it is always defined
        self.peak_price = self.entry_price
        # If stop_loss_price was not set explicitly, derive it from sl_pct
        if self.stop_loss_price == 0.0:
            self.stop_loss_price = self.entry_price * (1.0 - self.sl_pct)

    @property
    def max_price_seen(self) -> float:
        """Running maximum price observed since position open (high-watermark).

        Alias for :attr:`peak_price`; updated automatically in
        :meth:`~PaperExecutor.check_and_close` whenever the current price
        exceeds the previously recorded peak.  Used by the audit telemetry
        layer to surface the PICO_MÁX column in ``audit.log``.
        """
        return self.peak_price


class PaperExecutor:
    """Simulate (and optionally live-execute) order execution for multiple symbols.

    Parameters
    ----------
    db:
        :class:`~database.db_manager.DatabaseManager` instance used to
        persist completed trades.
    risk_manager:
        :class:`~risk.risk_manager.RiskManager` instance that tracks the
        simulated balance, computes position sizes, and enforces the
        maximum number of concurrent open positions.
    symbol:
        Default trading pair (e.g. ``"BTC/USDT"``).  Used when callers do
        not explicitly pass a *symbol* argument.
    exchange:
        Optional ccxt async Binance Futures client
        (e.g. ``ccxt.pro.binanceusdm``).  When provided, real market orders
        are placed on Binance Futures in addition to the paper simulation.
        When ``None`` (default) the executor runs in pure paper-trading mode.
    """

    def __init__(
        self,
        db: DatabaseManager,
        risk_manager: RiskManager,
        symbol: str = "BTC/USDT",
        exchange: Any | None = None,
    ) -> None:
        self._db = db
        self._risk = risk_manager
        self.symbol = symbol
        self._exchange = exchange
        self.open_positions: dict[str, OpenPosition] = {}
        self.total_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restore_position(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
    ) -> bool:
        """Restore a single open position from exchange state at startup.

        This method is used during bot initialisation to re-sync local state with
        positions that were already open on Binance when the bot was restarted.
        Unlike :meth:`try_open_trade`, this method:

        * Does **not** deduct *position_size* from the risk manager balance
          (the balance fetched from Binance already reflects open positions).
        * Does **not** place a live order (the position exists on the exchange).
        * Calls :meth:`~risk.risk_manager.RiskManager.register_open` so that the
          concurrent-position counter is correctly incremented.

        Parameters
        ----------
        symbol:
            Trading pair of the existing position (e.g. ``"BTC/USDT"``).
        entry_price:
            Average entry price of the existing position.
        position_size:
            Absolute notional value in USDT of the existing position.

        Returns
        -------
        bool
            *True* if the position was successfully restored, *False* if a
            position for *symbol* was already being tracked (no-op).
        """
        if symbol in self.open_positions:
            logger.debug(
                "restore_position: %s is already tracked – skipping.",
                symbol,
            )
            return False

        self.open_positions[symbol] = OpenPosition(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
        )
        self._risk.register_open()
        logger.info(
            "POSITION RESTORED  symbol=%s  entry_price=%.2f  size=%.2f",
            symbol,
            entry_price,
            position_size,
        )
        return True

    async def try_open_trade(
        self,
        entry_price: float,
        win_probability: float,
        symbol: str | None = None,
        timestamp: datetime | None = None,
        sentiment_score: float = 0.0,
        current_atr: float | None = None,
    ) -> bool:
        """Size and open a new trade using the leverage-based position formula.

        Returns *False* (and does nothing) if:

        * Trading is currently halted due to the daily loss limit.
        * A position for *symbol* is already open.
        * The maximum number of concurrent positions has been reached.
        * The simulated balance is insufficient.

        When an *exchange* client was provided at construction time a real
        market buy order is placed on Binance Futures
        (``create_market_buy_order`` with ``params={'leverage': LEVERAGE}``).

        Parameters
        ----------
        entry_price:
            Current market price at which the trade is entered.
        win_probability:
            ML-predicted probability of a profitable outcome (0–1).
        symbol:
            Trading pair to open.  Defaults to ``self.symbol``.
        timestamp:
            Trade entry time (UTC).  Defaults to *now*.
        sentiment_score:
            AI sentiment score in [-1.0, +1.0] used to compute dynamic
            risk thresholds.  Defaults to ``0.0`` (neutral / base thresholds).
        current_atr:
            Current ATR value for *symbol* (e.g. ATR_14 on the 15m chart).
            When provided, the initial stop loss and trailing stop distance
            are derived from this value instead of fixed percentages.
        """
        sym = symbol or self.symbol

        if self._risk.is_trading_halted():
            logger.warning(
                "⚠️ [ALERTA] Trading detenido por límite de pérdida diaria (symbol=%s) %s",
                sym,
                _DEBUG_LOG_HINT,
            )
            return False

        if sym in self.open_positions:
            logger.debug("Trade skipped – a position for %s is already open.", sym)
            return False

        if not self._risk.can_open_position():
            logger.debug(
                "Trade skipped – max open positions (%d) reached.",
                self._risk.max_positions,
            )
            return False

        # ── Sector / correlation-group exposure check ──────────────────────
        if self._risk.is_sector_exposed(sym, list(self.open_positions.keys())):
            sector = get_sector(sym)
            logger.warning(
                "🛡️ [RISK CONTROL] Señal de BUY para %s ignorada. "
                "Exposición máxima alcanzada para el sector: %s.",
                sym,
                sector,
            )
            return False

        position_size = self._risk.calculate_position_size(win_probability)
        if not self._risk.has_sufficient_balance(position_size):
            logger.warning(
                "⚠️ [ALERTA] Balance insuficiente (%.2f) para tamaño de posición %.2f %s",
                self._risk.balance,
                position_size,
                _DEBUG_LOG_HINT,
            )
            return False

        ts = timestamp or datetime.now(tz=timezone.utc)
        self._risk.deduct(position_size)
        self._risk.register_open()

        # ── Compute dynamic risk thresholds from sentiment ─────────────────
        thresholds: DynamicThresholds = get_dynamic_thresholds(sentiment_score)
        logger.debug(
            "DYNAMIC THRESHOLDS  symbol=%s  sentiment=%.4f  multiplier=%.2f  "
            "sl=%.4f  activation=%.4f  trailing_dist=%.4f",
            sym,
            sentiment_score,
            thresholds.multiplier,
            thresholds.sl_pct,
            thresholds.activation_pct,
            thresholds.trailing_distance_pct,
        )

        # ── Live Binance Futures order ─────────────────────────────────────
        if self._exchange is not None:
            try:
                # Set leverage at the symbol level (required by Binance Futures).
                # On Testnet this call may fail with NotSupported because there
                # is no sapi URL for the sandbox; in that case log a warning and
                # proceed with the order using whatever leverage is already set.
                try:
                    await self._exchange.set_leverage(LEVERAGE, sym)
                except CcxtNotSupported:
                    logger.warning(
                        "set_leverage not supported for %s (testnet?) – "
                        "proceeding with existing leverage setting.",
                        sym,
                    )
                # position_size is the leveraged notional value; convert to base
                # currency quantity by dividing by the entry price.
                amount = position_size / entry_price
                try:
                    await self._exchange.create_market_buy_order(
                        sym,
                        amount,
                        params={"leverage": LEVERAGE},
                    )
                except CcxtNotSupported:
                    # Testnet sapi endpoints are not available; the fapi order
                    # may still have been routed correctly.  Log a warning and
                    # continue rather than treating this as a hard failure.
                    logger.warning(
                        "create_market_buy_order raised NotSupported for %s "
                        "(testnet sapi metadata unavailable). "
                        "If 'fetchCurrencies' is False the fapi order likely "
                        "succeeded – verify via the Binance Testnet UI or "
                        "fetch_open_orders().",
                        sym,
                    )
                else:
                    logger.info(
                        "FUTURES BUY ORDER PLACED  symbol=%s  amount=%.6f  leverage=%d",
                        sym,
                        amount,
                        LEVERAGE,
                    )
            except (CcxtAuthenticationError, CcxtInsufficientFunds, CcxtNetworkError):
                logger.exception(
                    "Failed to place Binance Futures buy order for %s – "
                    "position recorded in paper simulation only.",
                    sym,
                )

        trade_id = await self._db.insert_open_trade(
            symbol=sym,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=ts,
        )

        # ── ATR-based dynamic SL / trailing distance ───────────────────────
        sl_distance: float | None = None
        stop_loss_price: float = 0.0  # 0.0 → __post_init__ derives from sl_pct
        atr_trailing_distance: float = 0.0  # 0.0 → use pct-based trailing
        if current_atr is not None and current_atr > 0.0:
            sl_distance = current_atr * ATR_SL_MULTIPLIER
            stop_loss_price = entry_price - sl_distance  # LONG position
            atr_trailing_distance = current_atr * ATR_TRAILING_MULTIPLIER

        self.open_positions[sym] = OpenPosition(
            symbol=sym,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=ts,
            trade_id=trade_id,
            sl_pct=thresholds.sl_pct,
            activation_pct=thresholds.activation_pct,
            trailing_distance_pct=thresholds.trailing_distance_pct,
            stop_loss_price=stop_loss_price,
            atr_trailing_distance=atr_trailing_distance,
        )
        # Operational console line – visible on the terminal at INFO level.
        if sl_distance is not None:
            logger.info(
                "✅ [OPEN LONG] %s a %.2f. SL Dinámico (ATR): %.2f (Distancia: %.2f)",
                sym,
                entry_price,
                self.open_positions[sym].stop_loss_price,
                sl_distance,
            )
        else:
            logger.info(
                "🚀 [ENTRADA] %s | Lado: BUY | Confianza: %.1f%% | SL: %.2f%% | TP: %.2f%%",
                sym,
                win_probability * 100,
                thresholds.sl_pct * 100,
                thresholds.activation_pct * 100,
            )
        asyncio.create_task(
            send_telegram_alert(
                f"🚀 *OPEN BUY* | #{sym}\n"
                f"Precio: {entry_price:.2f}\n"
                f"SL Dinámico (ATR): {self.open_positions[sym].stop_loss_price:.2f}"
            )
        )
        # Full technical details go to the debug log file only.
        logger.debug(
            "TRADE OPENED  symbol=%s  entry_price=%.2f  size=%.2f  id=%s  balance=%.2f  "
            "sl_price=%.2f  atr=%.4f",
            sym,
            entry_price,
            position_size,
            trade_id,
            self._risk.balance,
            self.open_positions[sym].stop_loss_price,
            current_atr if current_atr is not None else 0.0,
        )
        return True

    async def check_and_close(
        self,
        current_price: float,
        symbol: str | None = None,
        timestamp: datetime | None = None,
    ) -> float | None:
        """Check whether *current_price* triggers the stop loss for *symbol*.

        The active stop loss starts as the hard initial SL (``sl_pct`` below
        entry price – set dynamically at position open from the AI sentiment
        score).  Once the position profit reaches ``activation_pct`` the
        active SL updates dynamically to
        ``peak_price * (1 - trailing_distance_pct)``, never retreating below
        the previously set level.

        If the current price drops at or below the active stop loss the
        position is closed and the realised PnL is returned.

        Returns ``None`` when no position for *symbol* is open or the active
        stop loss has not been reached yet.

        Parameters
        ----------
        current_price:
            Latest market price to evaluate against open positions.
        symbol:
            Trading pair to check.  Defaults to ``self.symbol``.
        timestamp:
            Evaluation time (UTC).  Defaults to *now*.
        """
        sym = symbol or self.symbol
        if sym not in self.open_positions:
            return None

        pos = self.open_positions[sym]
        price_change_pct = (current_price - pos.entry_price) / pos.entry_price

        # ------------------------------------------------------------------
        # Trailing Stop logic
        # ------------------------------------------------------------------
        # 1. Update the running peak price whenever the price makes a new high.
        if current_price > pos.peak_price:
            pos.peak_price = current_price

        # 2. Activate the trailing stop once the position crosses activation_pct.
        if not pos.trailing_stop_active and price_change_pct >= pos.activation_pct:
            pos.trailing_stop_active = True
            logger.debug(
                "Trailing Stop ACTIVATED  symbol=%s  entry=%.2f  current=%.2f  profit=%.2f%%",
                sym,
                pos.entry_price,
                current_price,
                price_change_pct * 100,
            )

        # 3. Compute the active stop loss level.
        initial_sl_price = pos.stop_loss_price  # set at open (ATR or pct-based)
        if pos.trailing_stop_active:
            # Trailing SL: prefer ATR-based fixed distance; fall back to pct.
            if pos.atr_trailing_distance > 0.0:
                trailing_sl = pos.peak_price - pos.atr_trailing_distance
            else:
                trailing_sl = pos.peak_price * (1.0 - pos.trailing_distance_pct)
            # The SL can only move up, so take the max with the initial SL.
            active_sl = max(trailing_sl, initial_sl_price)
        else:
            active_sl = initial_sl_price

        # 4. Close the position if the current price hits the active SL.
        if current_price <= active_sl:
            pnl = price_change_pct * pos.position_size
            ts = timestamp or datetime.now(tz=timezone.utc)
            reason = "TSL" if pos.trailing_stop_active else "SL"
            reason_code = "TRAILING_STOP" if pos.trailing_stop_active else "SL_BASE"
            await self._close_position(symbol=sym, exit_price=current_price, exit_time=ts, pnl=pnl)
            # Operational console line – visible on the terminal at INFO level.
            logger.info(
                "💰 [CIERRE] %s | PnL: %.4f USDT | Motivo: %s",
                sym,
                pnl,
                reason,
            )
            asyncio.create_task(
                send_telegram_alert(
                    _build_trade_report(
                        sym=sym,
                        pos=pos,
                        exit_price=current_price,
                        exit_time=ts,
                        gross_pnl=pnl,
                        exit_reason_code=reason_code,
                        current_balance=self._risk.balance,
                    )
                )
            )
            # Full technical details go to the debug log file only.
            logger.debug(
                "TRADE CLOSED [%s]  symbol=%s  entry=%.2f  peak=%.2f  exit=%.2f  pnl=%.4f",
                reason,
                pos.symbol,
                pos.entry_price,
                pos.peak_price,
                current_price,
                pnl,
            )
            return pnl

        return None

    async def check_ml_exit(
        self,
        current_price: float,
        ml_signal: str,
        ml_probability: float | None = None,
        symbol: str | None = None,
        timestamp: datetime | None = None,
        min_confidence: float = 0.55,
    ) -> float | None:
        """Apply smart-exit logic driven by ML signals and TTL (Layers 1 and 4).

        This method is called from the signal-emitter loop (typically every 15 s)
        after the ML predictor has already generated a fresh signal.  It
        complements the mechanical stop-loss in :meth:`check_and_close` (Layer 3)
        with two higher-level exit conditions:

        Layer 1 – ML Exhaustion / Reversal Exit
            For a LONG position: if *ml_signal* is ``"SELL"`` **and** the
            predicted probability of a downward move
            (``1 - ml_probability``) is at least ``min_confidence *
            _ML_EXIT_CONFIDENCE_FACTOR`` (default 80 % of *min_confidence*),
            the position is closed immediately at *current_price*, ahead of any
            SL level.  This handles the scenario where the model detects trend
            exhaustion before the price reaches the trailing stop.

        Layer 4 – Time-To-Live (TTL) Exit
            If the position has been open for longer than its
            :attr:`~OpenPosition.max_ttl_hours` limit (default 12 h), and neither
            Layer 1 nor Layer 3 has yet closed it, the position is force-closed
            to free up margin in a sideways market.

        Parameters
        ----------
        current_price:
            Latest market price.
        ml_signal:
            Output of :meth:`~strategy.ml_predictor.MLPredictor.generate_signal`
            for *symbol* – ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        ml_probability:
            Output of :meth:`~strategy.ml_predictor.MLPredictor.predict_proba`
            for *symbol* – probability in ``[0, 1]`` that price will move up.
            When ``None`` Layer 1 is skipped entirely so that an untrained or
            unavailable model never triggers an early exit.
        symbol:
            Trading pair to evaluate.  Defaults to ``self.symbol``.
        timestamp:
            Evaluation time (UTC).  Defaults to *now*.
        min_confidence:
            Base ML confidence threshold used to compute the Layer-1 gate
            (``min_confidence * _ML_EXIT_CONFIDENCE_FACTOR``).  Pass the same
            value as ``_BUY_PROB_THRESHOLD`` used in the predictor (default
            0.55) to keep both thresholds in sync.

        Returns
        -------
        Realised PnL if the position was closed, ``None`` otherwise.
        """
        sym = symbol or self.symbol
        if sym not in self.open_positions:
            return None

        pos = self.open_positions[sym]
        price_change_pct = (current_price - pos.entry_price) / pos.entry_price
        pnl = price_change_pct * pos.position_size
        ts = timestamp or datetime.now(tz=timezone.utc)

        # ------------------------------------------------------------------
        # Layer 1: ML Exhaustion / Reversal Exit
        # ------------------------------------------------------------------
        # For a LONG position: close if ML signals a high-confidence SELL.
        # The downward-move confidence = 1 - ml_probability.
        # Threshold = min_confidence * _ML_EXIT_CONFIDENCE_FACTOR (default 0.44).
        # Skip Layer 1 entirely when ml_probability is unavailable (model not
        # yet trained) to avoid acting on a signal without probability support.
        if ml_signal == "SELL" and ml_probability is not None:
            sell_confidence = 1.0 - ml_probability
            confidence_gate = min_confidence * _ML_EXIT_CONFIDENCE_FACTOR
            if sell_confidence >= confidence_gate:
                await self._close_position(
                    symbol=sym,
                    exit_price=current_price,
                    exit_time=ts,
                    pnl=pnl,
                )
                logger.info(
                    "🤖💰 [SMART EXIT] %s cerrado por Reversión de Tendencia "
                    "(ML Exhaustion). PnL estimado: %.4f",
                    sym,
                    pnl,
                )
                asyncio.create_task(
                    send_telegram_alert(
                        _build_trade_report(
                            sym=sym,
                            pos=pos,
                            exit_price=current_price,
                            exit_time=ts,
                            gross_pnl=pnl,
                            exit_reason_code="SMART_EXIT_ML",
                            current_balance=self._risk.balance,
                        )
                    )
                )
                logger.debug(
                    "TRADE CLOSED [ML_EXHAUSTION]  symbol=%s  entry=%.2f  "
                    "exit=%.2f  sell_confidence=%.4f  gate=%.4f  pnl=%.4f",
                    sym,
                    pos.entry_price,
                    current_price,
                    sell_confidence,
                    confidence_gate,
                    pnl,
                )
                return pnl

        # ------------------------------------------------------------------
        # Layer 4: Time-To-Live (TTL) Exit
        # ------------------------------------------------------------------
        if pos.max_ttl_hours is not None:
            age_hours = (ts - pos.entry_time).total_seconds() / 3600.0
            if age_hours >= pos.max_ttl_hours:
                await self._close_position(
                    symbol=sym,
                    exit_price=current_price,
                    exit_time=ts,
                    pnl=pnl,
                )
                logger.info(
                    "⏳ [TTL EXIT] %s cerrado por tiempo máximo de exposición "
                    "alcanzado. PnL: %.4f",
                    sym,
                    pnl,
                )
                asyncio.create_task(
                    send_telegram_alert(
                        _build_trade_report(
                            sym=sym,
                            pos=pos,
                            exit_price=current_price,
                            exit_time=ts,
                            gross_pnl=pnl,
                            exit_reason_code="TTL_TIMEOUT",
                            current_balance=self._risk.balance,
                        )
                    )
                )
                logger.debug(
                    "TRADE CLOSED [TTL]  symbol=%s  entry=%.2f  exit=%.2f  "
                    "age_hours=%.2f  max_ttl=%.2f  pnl=%.4f",
                    sym,
                    pos.entry_price,
                    current_price,
                    age_hours,
                    pos.max_ttl_hours,
                    pnl,
                )
                return pnl

        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        pnl: float,
    ) -> None:
        """Persist the closed trade, update balance, and remove the open position."""
        pos = self.open_positions.get(symbol)
        if pos is None:
            logger.warning(
                "_close_position called for %s but no open position found – skipping.",
                symbol,
            )
            return

        # ── Live Binance Futures order ─────────────────────────────────────
        if self._exchange is not None:
            try:
                # position_size is the leveraged notional value; convert to base
                # currency quantity by dividing by the exit price.
                amount = pos.position_size / exit_price
                try:
                    await self._exchange.create_market_sell_order(
                        symbol,
                        amount,
                        params={"leverage": LEVERAGE, "reduceOnly": True},
                    )
                except CcxtNotSupported:
                    # Testnet sapi endpoints are not available; the fapi order
                    # may still have been routed correctly.  Log a warning and
                    # continue rather than treating this as a hard failure.
                    logger.warning(
                        "create_market_sell_order raised NotSupported for %s "
                        "(testnet sapi metadata unavailable). "
                        "If 'fetchCurrencies' is False the fapi order likely "
                        "succeeded – verify via the Binance Testnet UI or "
                        "fetch_open_orders().",
                        symbol,
                    )
                else:
                    logger.info(
                        "FUTURES SELL ORDER PLACED  symbol=%s  amount=%.6f  leverage=%d",
                        symbol,
                        amount,
                        LEVERAGE,
                    )
            except (CcxtAuthenticationError, CcxtInsufficientFunds, CcxtNetworkError):
                logger.exception(
                    "Failed to place Binance Futures sell order for %s – "
                    "position closed in paper simulation only.",
                    symbol,
                )

        if pos.trade_id is not None:
            await self._db.close_trade(
                trade_id=pos.trade_id,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
            )

        # Credit back the original stake plus any PnL (positive or negative)
        self._risk.credit(pos.position_size + pnl)
        self._risk.register_close()
        # Record realised losses so the safety break can trigger if needed
        if pnl < 0.0:
            self._risk.record_daily_loss(-pnl)
        self.total_pnl += pnl
        del self.open_positions[symbol]

