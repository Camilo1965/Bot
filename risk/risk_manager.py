"""
risk.risk_manager
~~~~~~~~~~~~~~~~~

Position sizing via Fractional (Half) Kelly Criterion.

Every trade uses a static Stop Loss of 1.5 % and a Take Profit of 3 %,
giving a reward-to-risk ratio of 2.  The Kelly fraction is computed as::

    f* = (p * b - (1 - p)) / b

where *b* = TP / SL = 2 and *p* is the ML-predicted win probability.

A half-Kelly multiplier (0.5) is applied to reduce variance::

    position_fraction = 0.5 * max(f*, 0)

When the Kelly fraction is zero or negative the trade has a non-positive
expected value and no position is opened.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Trade parameters ──────────────────────────────────────────────────────────
STOP_LOSS_PCT: float = 0.015    # 1.5 %
TAKE_PROFIT_PCT: float = 0.03   # 3.0 %
_REWARD_RISK_RATIO: float = TAKE_PROFIT_PCT / STOP_LOSS_PCT   # 2.0
_HALF_KELLY: float = 0.5        # Half-Kelly multiplier


class RiskManager:
    """Calculate position sizes using the Fractional (Half) Kelly Criterion.

    Parameters
    ----------
    initial_balance:
        Starting simulated balance in quote currency (e.g. USDT).
        Defaults to 10 000.
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.balance: float = initial_balance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(self, win_probability: float) -> float:
        """Return the position size in quote currency for the next trade.

        Uses the Half-Kelly Criterion.  Returns ``0.0`` when the expected
        value is non-positive (Kelly fraction ≤ 0).

        Parameters
        ----------
        win_probability:
            ML-predicted probability of the trade being profitable (0–1).
        """
        b = _REWARD_RISK_RATIO
        kelly = (win_probability * b - (1.0 - win_probability)) / b
        fractional_kelly = _HALF_KELLY * max(kelly, 0.0)
        position_size = self.balance * fractional_kelly
        logger.debug(
            "Kelly=%.4f  fractional=%.4f  position_size=%.2f  balance=%.2f",
            kelly,
            fractional_kelly,
            position_size,
            self.balance,
        )
        return position_size

    def has_sufficient_balance(self, position_size: float) -> bool:
        """Return *True* if the current balance can cover *position_size*."""
        return self.balance >= position_size > 0.0

    def deduct(self, amount: float) -> None:
        """Subtract *amount* from the simulated balance (trade entry)."""
        self.balance -= amount

    def credit(self, amount: float) -> None:
        """Add *amount* to the simulated balance (trade close + PnL)."""
        self.balance += amount
