"""
execution.paper_executor
~~~~~~~~~~~~~~~~~~~~~~~~

Simulates order execution and logs completed trades to TimescaleDB.

Opens and closes paper positions based on the trailing stop thresholds
defined in :mod:`risk.risk_manager`.  Each completed trade is persisted to
the ``trades_history`` table via the database manager.

A BUY position is protected as follows:

* **Initial Stop Loss** (SL): price falls ≥ ``INITIAL_SL`` (0.75 %) below
  entry price.  This hard floor is active from the moment the trade opens.
* **Trailing Stop** (TS): once the position gains ≥ ``ACTIVATION_PCT``
  (1.5 %) the active stop loss updates dynamically to
  ``peak_price * (1 - TRAILING_DISTANCE)`` (0.5 % below the running peak).
  The stop only moves up — it never retreats — allowing the trade to capture
  exponential crypto runs while locking in profit.

The trade exits **only** when the current price drops below the active stop
loss (initial SL before activation, trailing SL afterwards).

Multi-asset support: up to ``risk_manager.max_positions`` independent
positions may be open simultaneously, one per symbol.  Attempting to open a
second position on the same symbol is silently rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from risk.risk_manager import INITIAL_SL, ACTIVATION_PCT, TRAILING_DISTANCE, RiskManager

if TYPE_CHECKING:
    from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Lightweight container for a single open paper trade."""

    symbol: str
    entry_price: float
    position_size: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    trade_id: int | None = None
    # [PRO] Trailing Stop state
    peak_price: float = field(init=False)
    trailing_stop_active: bool = False

    def __post_init__(self) -> None:
        # Initialise peak_price to entry_price so it is always defined
        self.peak_price = self.entry_price


class PaperExecutor:
    """Simulate order execution for multiple symbols simultaneously.

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
    """

    def __init__(
        self,
        db: DatabaseManager,
        risk_manager: RiskManager,
        symbol: str = "BTC/USDT",
    ) -> None:
        self._db = db
        self._risk = risk_manager
        self.symbol = symbol
        self.open_positions: dict[str, OpenPosition] = {}
        self.total_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def try_open_trade(
        self,
        entry_price: float,
        win_probability: float,
        symbol: str | None = None,
        timestamp: datetime | None = None,
    ) -> bool:
        """Size and open a new paper trade using the Kelly Criterion.

        Returns *False* (and does nothing) if:

        * A position for *symbol* is already open.
        * The maximum number of concurrent positions has been reached.
        * The simulated balance is insufficient.

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
        """
        sym = symbol or self.symbol

        if sym in self.open_positions:
            logger.debug("Trade skipped – a position for %s is already open.", sym)
            return False

        if not self._risk.can_open_position():
            logger.info(
                "Trade skipped – max open positions (%d) reached.",
                self._risk.max_positions,
            )
            return False

        position_size = self._risk.calculate_position_size(win_probability)
        if not self._risk.has_sufficient_balance(position_size):
            logger.warning(
                "Trade skipped – insufficient balance (%.2f) for position size %.2f.",
                self._risk.balance,
                position_size,
            )
            return False

        ts = timestamp or datetime.now(tz=timezone.utc)
        self._risk.deduct(position_size)
        self._risk.register_open()

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
        )
        logger.info(
            "TRADE OPENED  symbol=%s  entry_price=%.2f  size=%.2f  id=%s  balance=%.2f",
            sym,
            entry_price,
            position_size,
            trade_id,
            self._risk.balance,
        )
        return True

    async def check_and_close(
        self,
        current_price: float,
        symbol: str | None = None,
        timestamp: datetime | None = None,
    ) -> float | None:
        """Check whether *current_price* triggers the stop loss for *symbol*.

        The active stop loss starts as the hard initial SL (``INITIAL_SL``
        below entry price).  Once the position profit reaches
        ``ACTIVATION_PCT`` the active SL updates dynamically to
        ``peak_price * (1 - TRAILING_DISTANCE)``, never retreating below the
        previously set level.

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

        # 2. Activate the trailing stop once the position crosses ACTIVATION_PCT.
        if not pos.trailing_stop_active and price_change_pct >= ACTIVATION_PCT:
            pos.trailing_stop_active = True
            logger.info(
                "Trailing Stop ACTIVATED  symbol=%s  entry=%.2f  current=%.2f  profit=%.2f%%",
                sym,
                pos.entry_price,
                current_price,
                price_change_pct * 100,
            )

        # 3. Compute the active stop loss level.
        initial_sl_price = pos.entry_price * (1.0 - INITIAL_SL)
        if pos.trailing_stop_active:
            # Trailing SL: TRAILING_DISTANCE below the running peak.
            # The SL can only move up, so take the max with the initial SL.
            trailing_sl = pos.peak_price * (1.0 - TRAILING_DISTANCE)
            active_sl = max(trailing_sl, initial_sl_price)
        else:
            active_sl = initial_sl_price

        # 4. Close the position if the current price hits the active SL.
        if current_price <= active_sl:
            pnl = price_change_pct * pos.position_size
            ts = timestamp or datetime.now(tz=timezone.utc)
            reason = "TSL" if pos.trailing_stop_active else "SL"
            await self._close_position(symbol=sym, exit_price=current_price, exit_time=ts, pnl=pnl)
            logger.info(
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
        self.total_pnl += pnl
        del self.open_positions[symbol]

