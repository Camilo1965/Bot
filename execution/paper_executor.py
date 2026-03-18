"""
execution.paper_executor
~~~~~~~~~~~~~~~~~~~~~~~~

Simulates order execution and logs completed trades to TimescaleDB.

Opens and closes paper positions based on the static Stop-Loss / Take-Profit
thresholds defined in :mod:`risk.risk_manager`.  Each completed trade is
persisted to the ``trades_history`` table via the database manager.

A position is closed automatically when the current market price hits either:

* **Take Profit** (TP): price rises ≥ 3 % above entry price.
* **Stop Loss** (SL): price falls ≥ 1.5 % below entry price.
* **Trailing Stop** (TS): once a position reaches 1.5 % profit a trailing
  stop is activated that follows the price at a 1 % distance from the
  running peak.  If the price then drops 1 % from its peak the position is
  closed to lock in profit.

Multi-asset support: up to ``risk_manager.max_positions`` independent
positions may be open simultaneously, one per symbol.  Attempting to open a
second position on the same symbol is silently rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from risk.risk_manager import STOP_LOSS_PCT, TAKE_PROFIT_PCT, RiskManager

# ── [PRO] Trailing Stop parameters ───────────────────────────────────────────
# Profit level at which the trailing stop is activated (1.5 %)
_TRAILING_STOP_ACTIVATION_PCT: float = 0.015
# Distance that the trailing stop maintains below the running peak price (1 %)
_TRAILING_STOP_DISTANCE_PCT: float = 0.01

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
        """Check whether *current_price* triggers the SL or TP threshold for *symbol*.

        If either threshold is reached the position is closed, the trade is
        persisted to the database, and the realised PnL is returned.

        Returns ``None`` when no position for *symbol* is open or neither
        threshold has been reached yet.

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
        # [PRO] Trailing Stop logic
        # ------------------------------------------------------------------
        # 1. Update the running peak price whenever the price makes a new high.
        if current_price > pos.peak_price:
            pos.peak_price = current_price

        # 2. Activate the trailing stop once the position crosses the 1.5 %
        #    profit threshold for the first time.
        if not pos.trailing_stop_active and price_change_pct >= _TRAILING_STOP_ACTIVATION_PCT:
            pos.trailing_stop_active = True
            logger.info(
                "[PRO] Trailing Stop ACTIVATED  symbol=%s  entry=%.2f  current=%.2f  profit=%.2f%%",
                sym,
                pos.entry_price,
                current_price,
                price_change_pct * 100,
            )

        # 3. When the trailing stop is active, close the position if the price
        #    drops more than TRAILING_STOP_DISTANCE_PCT below the peak.
        if pos.trailing_stop_active:
            drop_from_peak = (pos.peak_price - current_price) / pos.peak_price
            if drop_from_peak >= _TRAILING_STOP_DISTANCE_PCT:
                pnl = price_change_pct * pos.position_size
                ts = timestamp or datetime.now(tz=timezone.utc)
                await self._close_position(symbol=sym, exit_price=current_price, exit_time=ts, pnl=pnl)
                logger.info(
                    "[PRO] TRADE CLOSED [TS]  symbol=%s  entry=%.2f  peak=%.2f  exit=%.2f  pnl=%.4f",
                    sym,
                    pos.entry_price,
                    pos.peak_price,
                    current_price,
                    pnl,
                )
                return pnl

        hit_take_profit = price_change_pct >= TAKE_PROFIT_PCT
        hit_stop_loss = price_change_pct <= -STOP_LOSS_PCT

        if hit_take_profit or hit_stop_loss:
            reason = "TP" if hit_take_profit else "SL"
            pnl = price_change_pct * pos.position_size
            ts = timestamp or datetime.now(tz=timezone.utc)

            await self._close_position(symbol=sym, exit_price=current_price, exit_time=ts, pnl=pnl)

            logger.info(
                "TRADE CLOSED [%s]  symbol=%s  entry=%.2f  exit=%.2f  pnl=%.4f",
                reason,
                pos.symbol,
                pos.entry_price,
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

