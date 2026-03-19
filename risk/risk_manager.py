"""
risk.risk_manager
~~~~~~~~~~~~~~~~~

Position sizing via Fractional (Half) Kelly Criterion.

Trades use a trailing stop instead of a hard Take Profit to capture
exponential crypto runs.  The Kelly reward-to-risk ratio is based on the
activation threshold and the initial stop loss (2.0)::

    f* = (p * b - (1 - p)) / b

where *b* = ACTIVATION_PCT / INITIAL_SL = 2 and *p* is the ML-predicted
win probability.

A half-Kelly multiplier (0.5) is applied to reduce variance::

    position_fraction = 0.5 * max(f*, 0)

When the Kelly fraction is zero or negative the trade has a non-positive
expected value and no position is opened.

Trailing stop parameters
------------------------
* ``INITIAL_SL``:        Hard stop loss during the entry phase (0.75 %).
* ``ACTIVATION_PCT``:    Minimum profit required to activate the trailing
                         stop (1.5 %).  Once this threshold is reached the
                         active stop loss updates dynamically.
* ``TRAILING_DISTANCE``: Gap maintained between the running peak price and
                         the trailing stop level (0.5 %).

Multi-asset risk controls:

* ``max_positions`` limits the number of simultaneously open positions.
* Each trade's position size is capped at ``balance / max_positions`` so
  that no single trade can consume more than an equal share of the
  portfolio.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Trade parameters ──────────────────────────────────────────────────────────
INITIAL_SL: float = 0.0075          # 0.75 % hard stop loss for initial protection
ACTIVATION_PCT: float = 0.015       # 1.5 % profit required to activate trailing stop
TRAILING_DISTANCE: float = 0.005    # 0.5 % trailing distance from the highest peak
_REWARD_RISK_RATIO: float = ACTIVATION_PCT / INITIAL_SL   # 2.0
_HALF_KELLY: float = 0.5            # Half-Kelly multiplier

MAX_POSITIONS: int = 3          # Maximum simultaneous open positions


class RiskManager:
    """Calculate position sizes using the Fractional (Half) Kelly Criterion.

    Parameters
    ----------
    initial_balance:
        Starting simulated balance in quote currency (e.g. USDT).
        Defaults to 10 000.
    max_positions:
        Maximum number of positions that may be open at the same time.
        Defaults to 3.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        max_positions: int = MAX_POSITIONS,
    ) -> None:
        self.balance: float = initial_balance
        self.max_positions: int = max_positions
        self._open_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(self, win_probability: float) -> float:
        """Return the position size in quote currency for the next trade.

        Uses the Half-Kelly Criterion.  Returns ``0.0`` when the expected
        value is non-positive (Kelly fraction ≤ 0).

        The position size is also capped at ``balance / max_positions`` so
        that no single trade consumes more than a fair share of the
        portfolio.

        Parameters
        ----------
        win_probability:
            ML-predicted probability of the trade being profitable (0–1).
        """
        b = _REWARD_RISK_RATIO
        kelly = (win_probability * b - (1.0 - win_probability)) / b
        fractional_kelly = _HALF_KELLY * max(kelly, 0.0)
        # Cap allocation to an equal share of the current balance
        max_allocation = self.balance / self.max_positions
        position_size = min(self.balance * fractional_kelly, max_allocation)
        logger.debug(
            "Kelly=%.4f  fractional=%.4f  position_size=%.2f  balance=%.2f  max_allocation=%.2f",
            kelly,
            fractional_kelly,
            position_size,
            self.balance,
            max_allocation,
        )
        return position_size

    def can_open_position(self) -> bool:
        """Return *True* if another position may be opened (below max_positions)."""
        return self._open_count < self.max_positions

    def register_open(self) -> None:
        """Increment the open-position counter (call when a trade is opened)."""
        self._open_count += 1

    def register_close(self) -> None:
        """Decrement the open-position counter (call when a trade is closed)."""
        if self._open_count == 0:
            logger.warning(
                "register_close called when open_count is already 0 – "
                "possible mismatched open/close calls."
            )
            return
        self._open_count -= 1

    def has_sufficient_balance(self, position_size: float) -> bool:
        """Return *True* if the current balance can cover *position_size*."""
        return self.balance >= position_size > 0.0

    def deduct(self, amount: float) -> None:
        """Subtract *amount* from the simulated balance (trade entry)."""
        self.balance -= amount

    def credit(self, amount: float) -> None:
        """Add *amount* to the simulated balance (trade close + PnL)."""
        self.balance += amount

