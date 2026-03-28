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

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from execution.paper_executor import (
    ATR_SL_MULTIPLIER,
    ATR_TRAILING_MULTIPLIER,
    OpenPosition,
    PaperExecutor,
    _TAKER_FEE_RATE,
    record_trade,
)
from risk.risk_manager import (
    LEVERAGE,
    DynamicThresholds,
    get_dynamic_thresholds,
    get_sector,
)
from utils.telegram_notifier import send_telegram_alert

if TYPE_CHECKING:
    from database.db_manager import DatabaseManager
    from risk.risk_manager import RiskManager

try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]

    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover – only available on Windows with MT5 installed
    mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

import asyncio

logger = logging.getLogger(__name__)

# ── Default risk per trade used by calculate_lot_size ─────────────────────────
# Mirrors RISK_PER_TRADE from risk_manager but expressed as a fraction for
# lot-size math.  Override by passing ``risk_pct`` explicitly.
_DEFAULT_RISK_PCT: float = 0.01  # 1 % of account equity per trade (Prop Firm safe)

# Small buffer added to the trailing distance to absorb spread noise (fraction
# of price, not percentage points).  A dedicated per-tick spread measurement
# supersedes this when MT5 is available.
_SPREAD_BUFFER_FALLBACK: float = 0.001  # 0.10 % fallback when spread is unknown

# Minimum recognised lot size when the broker reports 0.0 (safety guard).
_FALLBACK_VOLUME_MIN: float = 0.01
_FALLBACK_VOLUME_STEP: float = 0.01


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

    # Round DOWN to the nearest volume_step so we never exceed the risk budget.
    lots_stepped = math.floor(lots_raw / volume_step) * volume_step
    # Clamp to broker limits.
    lots = max(volume_min, min(volume_max, lots_stepped))

    logger.debug(
        "calculate_lot_size: symbol=%s  balance=%.2f  risk_pct=%.4f  "
        "sl_distance=%.4f  contract_size=%.4f  "
        "lots_raw=%.4f  lots_stepped=%.4f  lots=%.4f",
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
        super().__init__(db=db, risk_manager=risk_manager, symbol=symbol, exchange=None)
        self._live = live
        self._risk_pct = risk_pct
        self._magic = magic
        self._deviation = deviation

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
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or tick.ask <= 0 or tick.bid <= 0:
                return _SPREAD_BUFFER_FALLBACK
            spread_price = tick.ask - tick.bid
            return spread_price / entry_price if entry_price > 0 else _SPREAD_BUFFER_FALLBACK
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "_get_spread_fraction: failed to fetch tick for %s (%s) – "
                "using fallback spread buffer.",
                symbol,
                exc,
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

    def _send_order(self, request: dict) -> Any | None:
        """Send an order via ``mt5.order_send`` and log the result."""
        result = mt5.order_send(request)
        if result is None:
            logger.error(
                "mt5.order_send returned None for %s. Last error: %s",
                request.get("symbol"),
                mt5.last_error(),
            )
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "MT5 order rejected for %s: retcode=%d  comment=%s",
                request.get("symbol"),
                result.retcode,
                result.comment,
            )
            return None

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

        # ── Duplicate and circuit-breaker checks (inherited logic) ─────────
        if self._risk.is_trading_halted():
            logger.warning(
                "⚠️ [ALERTA] Trading detenido por límite de pérdida diaria (symbol=%s)",
                sym,
            )
            return False

        if self._risk.is_portfolio_dd_exceeded():
            logger.warning(
                "🚨 [CIRCUIT BREAKER] Todas las nuevas posiciones bloqueadas – "
                "límite de drawdown de cartera alcanzado (symbol=%s)",
                sym,
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
                "⚠️ [ALERTA] Balance insuficiente (%.2f) para tamaño de posición %.2f",
                self._risk.balance,
                position_size,
            )
            return False

        ts = timestamp or datetime.now(tz=timezone.utc)
        self._risk.deduct(position_size)
        self._risk.register_open()

        # ── Dynamic risk thresholds from sentiment ─────────────────────────
        thresholds: DynamicThresholds = get_dynamic_thresholds(sentiment_score)

        # ── ATR-based dynamic SL / trailing distance ───────────────────────
        sl_distance: float | None = None
        stop_loss_price: float = 0.0
        atr_trailing_distance: float = 0.0
        if current_atr is not None and current_atr > 0.0:
            sl_distance = current_atr * ATR_SL_MULTIPLIER
            stop_loss_price = entry_price - sl_distance
            atr_trailing_distance = current_atr * ATR_TRAILING_MULTIPLIER
        else:
            # Fallback: derive SL from the percentage threshold
            stop_loss_price = entry_price * (1.0 - thresholds.sl_pct)

        # ── Spread-adjusted trailing distance ─────────────────────────────
        effective_trailing_pct = self._effective_trailing_distance(
            sym, entry_price, thresholds.trailing_distance_pct
        )
        # Patch thresholds to carry the spread-adjusted value so OpenPosition
        # stores the correct distance.
        thresholds = DynamicThresholds(
            multiplier=thresholds.multiplier,
            sl_pct=thresholds.sl_pct,
            activation_pct=thresholds.activation_pct,
            trailing_distance_pct=effective_trailing_pct,
        )

        # ── Live MT5 order ─────────────────────────────────────────────────
        if self._live:
            if not _MT5_AVAILABLE:
                logger.error(
                    "MT5 live mode requested but MetaTrader5 library is not "
                    "installed.  Rolling back and aborting."
                )
                self._risk.credit(position_size)
                self._risk.register_close()
                return False

            # Fetch account equity from MT5 for lot-size calculation.
            acct = mt5.account_info()
            account_equity: float = acct.equity if acct else self._risk.balance

            lots = calculate_lot_size(
                symbol=sym,
                account_balance=account_equity,
                sl_distance_price=max(entry_price - stop_loss_price, 1e-8),
                risk_pct=self._risk_pct,
            )

            # Fetch latest ask price for the BUY order.
            tick = mt5.symbol_info_tick(sym)
            ask_price = tick.ask if tick else entry_price

            request = self._build_buy_request(
                symbol=sym,
                lots=lots,
                price=ask_price,
                sl=stop_loss_price,
            )
            result = self._send_order(request)
            if result is None:
                # Order rejected – roll back.
                self._risk.credit(position_size)
                self._risk.register_close()
                return False

        trade_id = await self._db.insert_open_trade(
            symbol=sym,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=ts,
        )

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
            ml_confidence=win_probability,
            sentiment_score=sentiment_score,
        )

        record_trade(
            timestamp=ts,
            symbol=sym,
            action="BUY",
            execution_price=entry_price,
            quantity=position_size / entry_price,
            ml_confidence_at_entry=win_probability,
            sentiment_score_at_entry=sentiment_score,
        )
        self._save_state()

        if sl_distance is not None:
            logger.info(
                "✅ [OPEN LONG] %s a %.2f. SL Dinámico (ATR): %.2f (Distancia: %.2f)",
                sym,
                entry_price,
                stop_loss_price,
                sl_distance,
            )
        else:
            logger.info(
                "🚀 [ENTRADA] %s | Lado: BUY | Confianza: %.1f%% | "
                "SL: %.2f%% | TP activ.: %.2f%%",
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
        return True

    # ------------------------------------------------------------------
    # _close_position override
    # ------------------------------------------------------------------

    async def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        pnl: float,
        exit_reason_code: str = "",
    ) -> None:
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
            return

        # ── Live MT5 order ─────────────────────────────────────────────────
        if self._live:
            if not _MT5_AVAILABLE:
                logger.error(
                    "MT5 live mode requested but MetaTrader5 library is not installed. "
                    "Closing in paper simulation only."
                )
            else:
                # Find the MT5 position ticket for this symbol tagged with our magic.
                mt5_positions = mt5.positions_get(symbol=symbol)
                if mt5_positions:
                    for mt5_pos in mt5_positions:
                        if mt5_pos.magic != self._magic:
                            continue

                        # Use the current bid price for a LONG close (SELL at bid).
                        tick = mt5.symbol_info_tick(symbol)
                        bid_price = tick.bid if tick else exit_price

                        request = self._build_sell_request(
                            symbol=symbol,
                            lots=mt5_pos.volume,
                            price=bid_price,
                            position_id=mt5_pos.ticket,
                        )
                        result = self._send_order(request)
                        if result is None:
                            logger.error(
                                "MT5 close order rejected for %s (ticket=%d) – "
                                "removing from local state anyway to avoid ghost positions.",
                                symbol,
                                mt5_pos.ticket,
                            )
                        break
                else:
                    logger.warning(
                        "[MT5] No open position found for %s with magic=%d – "
                        "closing in paper simulation only.",
                        symbol,
                        self._magic,
                    )

        # ── Paper book-keeping (identical to PaperExecutor) ────────────────
        if pos.trade_id is not None:
            await self._db.close_trade(
                trade_id=pos.trade_id,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
            )

        self._risk.credit(pos.position_size + pnl)
        self._risk.register_close()
        if pnl < 0.0:
            self._risk.record_daily_loss(-pnl)
        self.total_pnl += pnl
        del self.open_positions[symbol]

        total_fees = pos.position_size * _TAKER_FEE_RATE * 2
        net_pnl = pnl - total_fees
        margin_used = pos.position_size / LEVERAGE
        pnl_pct = (net_pnl / margin_used * 100) if margin_used > 0 else 0.0
        record_trade(
            timestamp=exit_time,
            symbol=symbol,
            action="SELL",
            execution_price=exit_price,
            quantity=pos.position_size / exit_price,
            ml_confidence_at_entry=pos.ml_confidence,
            sentiment_score_at_entry=pos.sentiment_score,
            exit_reason=exit_reason_code,
            pnl_usdt=net_pnl,
            pnl_percent=pnl_pct,
        )
        self._save_state()

    # ------------------------------------------------------------------
    # sync_positions_with_exchange override
    # ------------------------------------------------------------------

    async def sync_positions_with_exchange(self) -> int:
        """Re-sync local state against live MT5 positions.

        Detects *ghost* positions (in local memory but not on MT5) and removes
        them.  In pure paper mode (``live=False``) returns the local position
        count unchanged.
        """
        if not self._live:
            return len(self.open_positions)

        if not _MT5_AVAILABLE:
            logger.warning("[MT5 SYNC] MetaTrader5 not available – skipping sync.")
            return len(self.open_positions)

        mt5_positions = mt5.positions_get()
        if mt5_positions is None:
            logger.warning("[MT5 SYNC] mt5.positions_get() returned None.")
            return len(self.open_positions)

        live_symbols: set[str] = {
            p.symbol for p in mt5_positions if p.magic == self._magic
        }

        # Ghost detection
        ghost_symbols = [sym for sym in self.open_positions if sym not in live_symbols]
        for sym in ghost_symbols:
            logger.info(
                "[MT5 SYNC] 👻 Ghost position: %s in bot memory but not on MT5. Removing.",
                sym,
            )
            self.open_positions.pop(sym, None)
            self._risk.register_close()

        # Discrepancy detection
        untracked = [sym for sym in live_symbols if sym not in self.open_positions]
        for sym in untracked:
            logger.warning(
                "[MT5 SYNC] ⚠️ Position for %s is open on MT5 but not tracked locally. "
                "Consider restarting the bot or manually reconciling.",
                sym,
            )

        return len(live_symbols)
