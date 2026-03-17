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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from risk.risk_manager import STOP_LOSS_PCT, TAKE_PROFIT_PCT, RiskManager

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


class PaperExecutor:
    """Simulate order execution for a single symbol.

    Parameters
    ----------
    db:
        :class:`~database.db_manager.DatabaseManager` instance used to
        persist completed trades.
    risk_manager:
        :class:`~risk.risk_manager.RiskManager` instance that tracks the
        simulated balance and computes position sizes.
    symbol:
        Trading pair (e.g. ``"BTC/USDT"``).
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
        self.open_position: OpenPosition | None = None
        self.total_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def try_open_trade(
        self,
        entry_price: float,
        win_probability: float,
        timestamp: datetime | None = None,
    ) -> bool:
        """Size and open a new paper trade using the Kelly Criterion.

        Returns *False* (and does nothing) if a position is already open or
        the simulated balance is insufficient.

        Parameters
        ----------
        entry_price:
            Current market price at which the trade is entered.
        win_probability:
            ML-predicted probability of a profitable outcome (0–1).
        timestamp:
            Trade entry time (UTC).  Defaults to *now*.
        """
        if self.open_position is not None:
            logger.debug("Trade skipped – a position is already open.")
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

        trade_id = await self._db.insert_open_trade(
            symbol=self.symbol,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=ts,
        )
        self.open_position = OpenPosition(
            symbol=self.symbol,
            entry_price=entry_price,
            position_size=position_size,
            entry_time=ts,
            trade_id=trade_id,
        )
        logger.info(
            "TRADE OPENED  symbol=%s  entry_price=%.2f  size=%.2f  id=%s  balance=%.2f",
            self.symbol,
            entry_price,
            position_size,
            trade_id,
            self._risk.balance,
        )
        return True

    async def check_and_close(
        self,
        current_price: float,
        timestamp: datetime | None = None,
    ) -> float | None:
        """Check whether *current_price* triggers the SL or TP threshold.

        If either threshold is reached the position is closed, the trade is
        persisted to the database, and the realised PnL is returned.

        Returns ``None`` when no position is open or neither threshold has
        been reached yet.
        """
        if self.open_position is None:
            return None

        pos = self.open_position
        price_change_pct = (current_price - pos.entry_price) / pos.entry_price

        hit_take_profit = price_change_pct >= TAKE_PROFIT_PCT
        hit_stop_loss = price_change_pct <= -STOP_LOSS_PCT

        if hit_take_profit or hit_stop_loss:
            reason = "TP" if hit_take_profit else "SL"
            pnl = price_change_pct * pos.position_size
            ts = timestamp or datetime.now(tz=timezone.utc)

            await self._close_position(exit_price=current_price, exit_time=ts, pnl=pnl)

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
        exit_price: float,
        exit_time: datetime,
        pnl: float,
    ) -> None:
        """Persist the closed trade, update balance, and clear open position."""
        pos = self.open_position
        assert pos is not None  # noqa: S101 – guaranteed by caller

        if pos.trade_id is not None:
            await self._db.close_trade(
                trade_id=pos.trade_id,
                exit_price=exit_price,
                exit_time=exit_time,
                pnl=pnl,
            )

        # Credit back the original stake plus any PnL (positive or negative)
        self._risk.credit(pos.position_size + pnl)
        self.total_pnl += pnl
        self.open_position = None
