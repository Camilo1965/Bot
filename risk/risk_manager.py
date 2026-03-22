"""
risk.risk_manager
~~~~~~~~~~~~~~~~~

Position sizing for Binance Futures with leverage and daily-loss safety break.

Position size formula::

    position_size = balance * RISK_PER_TRADE * LEVERAGE

where:
* ``RISK_PER_TRADE`` is the fraction of the balance risked per trade (15 %).
* ``LEVERAGE``       is the futures leverage multiplier (5× – Elite Sniper).

Trailing stop parameters
------------------------
* ``INITIAL_SL``:        Hard stop loss during the entry phase (1.5 %).
* ``ACTIVATION_PCT``:    Minimum profit required to activate the trailing
                         stop (3 %).  Once this threshold is reached the
                         active stop loss updates dynamically.
* ``TRAILING_DISTANCE``: Gap maintained between the running peak price and
                         the trailing stop level (2 %).

Dynamic risk management
-----------------------
:func:`get_dynamic_thresholds` adjusts the trailing-stop parameters
at position open time based on the absolute value of the AI sentiment
score provided by :mod:`strategy.sentiment_llm`:

* **Low sentiment** (abs < 0.30): multiplier 0.8 → *scalping* mode with
  tighter SL and faster profit-taking.
* **High sentiment** (abs > 0.60): multiplier 1.8 → *swing trading* mode
  with wider thresholds to capture large directional moves.
* Values in between are linearly interpolated.

The multiplier is hard-clamped to [0.5, 2.5] and the resulting stop loss
is additionally capped at 5 % to prevent dangerously wide stops.

Daily-loss safety break
-----------------------
If the total realised loss for the current UTC day exceeds
``MAX_DAILY_LOSS_PCT`` (3 %) of the account balance at the start of the
day, all new trades are blocked for 24 hours.  Call
:meth:`reset_daily_stats` once a day (e.g. at midnight UTC) to lift the
block and refresh the reference balance.

Multi-asset risk controls:

* ``max_positions`` limits the number of simultaneously open positions.
* Each trade's position size is capped at ``balance / max_positions`` so
  that no single trade can consume more than an equal share of the
  portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ── Trade parameters ──────────────────────────────────────────────────────────
INITIAL_SL: float = 0.015           # 1.5 % hard stop loss for initial protection
ACTIVATION_PCT: float = 0.03       # 3 % profit required to activate trailing stop
TRAILING_DISTANCE: float = 0.02    # 2 % trailing distance from the highest peak

# ── Futures / leverage parameters ─────────────────────────────────────────────
LEVERAGE: int = 5                   # 5× futures leverage (Elite Sniper)
RISK_PER_TRADE: float = 0.15        # 15 % of balance risked per trade

# ── Daily-loss safety break ───────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT: float = 0.03    # 3 % maximum daily loss before halting
_HALT_DURATION: timedelta = timedelta(hours=24)

MAX_POSITIONS: int = 3          # Maximum simultaneous open positions

# ── Dynamic risk management – base thresholds (neutral market) ────────────────
BASE_SL: float = 0.015                      # 1.5 % base stop loss
BASE_ACTIVATION_PCT: float = 0.03           # 3 % profit to activate trailing stop
BASE_TRAILING_DISTANCE: float = 0.02        # 2 % trailing distance

# ── Dynamic risk management – sentiment multiplier bounds ─────────────────────
_SENTIMENT_LOW_THRESHOLD: float = 0.30      # below this → scalping regime
_SENTIMENT_HIGH_THRESHOLD: float = 0.60    # above this → swing-trading regime
_MULTIPLIER_LOW: float = 0.8               # shrink thresholds in low-sentiment markets
_MULTIPLIER_HIGH: float = 1.8              # expand thresholds in high-sentiment markets
_MULTIPLIER_MIN: float = 0.5               # absolute floor (safety net)
_MULTIPLIER_MAX: float = 2.5               # absolute cap (safety net)
_SL_CAP: float = 0.05                      # maximum allowed stop-loss fraction (5 %)


@dataclass
class DynamicThresholds:
    """Risk-management thresholds adjusted by an AI sentiment multiplier."""

    sl_pct: float
    activation_pct: float
    trailing_distance_pct: float
    multiplier: float


def get_dynamic_thresholds(sentiment_score: float) -> DynamicThresholds:
    """Return sentiment-adjusted risk thresholds.

    The *sentiment_score* is expected in **[-1.0, +1.0]** (as returned by
    :func:`~strategy.sentiment_llm.get_gemini_sentiment`).  Only the
    absolute magnitude is used so that both strong bearish and strong
    bullish readings widen the thresholds.

    Regime mapping
    ~~~~~~~~~~~~~~
    * ``abs(sentiment) < 0.30`` → *scalping* – tighter SL / activation / trailing
      distance so the bot takes profits quickly.
    * ``abs(sentiment) > 0.60`` → *swing trading* – wider thresholds so the bot
      can capture large directional moves.
    * In between → multiplier is linearly interpolated.

    Safety net
    ~~~~~~~~~~
    * The multiplier is clamped to [``_MULTIPLIER_MIN``, ``_MULTIPLIER_MAX``]
      so it is never zero or negative.
    * The resulting SL is additionally capped at ``_SL_CAP`` (5 %) to prevent
      dangerously wide stop losses.

    Parameters
    ----------
    sentiment_score:
        LLM-generated sentiment score in [-1.0, +1.0].

    Returns
    -------
    DynamicThresholds
        Adjusted thresholds ready for use in position management.
    """
    abs_sentiment = abs(sentiment_score)

    if abs_sentiment < _SENTIMENT_LOW_THRESHOLD:
        multiplier = _MULTIPLIER_LOW
    elif abs_sentiment > _SENTIMENT_HIGH_THRESHOLD:
        multiplier = _MULTIPLIER_HIGH
    else:
        # Linear interpolation between _MULTIPLIER_LOW and _MULTIPLIER_HIGH
        span = _SENTIMENT_HIGH_THRESHOLD - _SENTIMENT_LOW_THRESHOLD
        t = (abs_sentiment - _SENTIMENT_LOW_THRESHOLD) / span
        multiplier = _MULTIPLIER_LOW + t * (_MULTIPLIER_HIGH - _MULTIPLIER_LOW)

    # Apply safety bounds
    multiplier = max(_MULTIPLIER_MIN, min(_MULTIPLIER_MAX, multiplier))

    sl_pct = min(BASE_SL * multiplier, _SL_CAP)
    activation_pct = BASE_ACTIVATION_PCT * multiplier
    trailing_distance_pct = BASE_TRAILING_DISTANCE * multiplier

    logger.debug(
        "dynamic_thresholds  abs_sentiment=%.4f  multiplier=%.4f  "
        "sl=%.4f  activation=%.4f  trailing_dist=%.4f",
        abs_sentiment,
        multiplier,
        sl_pct,
        activation_pct,
        trailing_distance_pct,
    )
    return DynamicThresholds(
        sl_pct=sl_pct,
        activation_pct=activation_pct,
        trailing_distance_pct=trailing_distance_pct,
        multiplier=multiplier,
    )


class RiskManager:
    """Calculate position sizes for Binance Futures with a daily-loss safety break.

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
        # Daily-loss tracking
        self._daily_start_balance: float = initial_balance
        self._daily_loss: float = 0.0
        self._trading_halted_until: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(self, win_probability: float) -> float:  # noqa: ARG002
        """Return the position size in quote currency for the next trade.

        Uses the formula ``balance * RISK_PER_TRADE * LEVERAGE``.

        The position size is also capped at ``balance / max_positions`` so
        that no single trade consumes more than a fair share of the
        portfolio.

        Parameters
        ----------
        win_probability:
            ML-predicted probability of the trade being profitable (0–1).
            Accepted for API compatibility; not used in the leverage-based
            formula.
        """
        position_size = self.balance * RISK_PER_TRADE * LEVERAGE
        # Cap allocation to an equal share of the current balance
        max_allocation = self.balance / self.max_positions
        position_size = min(position_size, max_allocation)
        logger.debug(
            "position_size=%.2f  balance=%.2f  leverage=%d  risk_per_trade=%.4f  max_allocation=%.2f",
            position_size,
            self.balance,
            LEVERAGE,
            RISK_PER_TRADE,
            max_allocation,
        )
        return position_size

    def can_open_position(self) -> bool:
        """Return *True* if another position may be opened (below max_positions)."""
        return self._open_count < self.max_positions

    # ------------------------------------------------------------------
    # Daily-loss safety break
    # ------------------------------------------------------------------

    def is_trading_halted(self) -> bool:
        """Return *True* if trading has been halted due to the daily loss limit.

        The halt is automatically lifted once the 24-hour window has elapsed.
        """
        if self._trading_halted_until is None:
            return False
        now = datetime.now(tz=timezone.utc)
        if now >= self._trading_halted_until:
            # Window has expired – clear the halt
            logger.info("Trading halt expired – resuming normal operation.")
            self._trading_halted_until = None
            return False
        return True

    def record_daily_loss(self, loss: float) -> None:
        """Accumulate a realised loss and trigger the safety break if needed.

        Parameters
        ----------
        loss:
            Positive value representing the loss amount in quote currency.
            If the value is negative (i.e. a profit), it is ignored.
        """
        if loss <= 0.0:
            return
        self._daily_loss += loss
        threshold = self._daily_start_balance * MAX_DAILY_LOSS_PCT
        if self._daily_loss >= threshold and self._trading_halted_until is None:
            self._trading_halted_until = datetime.now(tz=timezone.utc) + _HALT_DURATION
            logger.warning(
                "Daily loss limit breached (%.2f / %.2f = %.2f%%). "
                "Trading HALTED until %s.",
                self._daily_loss,
                self._daily_start_balance,
                (self._daily_loss / self._daily_start_balance) * 100,
                self._trading_halted_until.isoformat(),
            )

    def reset_daily_stats(self) -> None:
        """Reset daily-loss counters (call once per UTC day, e.g. at midnight).

        Clears the accumulated daily loss and refreshes the reference balance
        used for the 3 % threshold calculation.  Also lifts any active trading
        halt so the new day can start fresh.
        """
        self._daily_start_balance = self.balance
        self._daily_loss = 0.0
        self._trading_halted_until = None
        logger.info(
            "Daily stats reset.  New reference balance: %.2f",
            self._daily_start_balance,
        )

    @property
    def open_count(self) -> int:
        """Current number of open positions."""
        return self._open_count

    def sync_open_count(self, count: int) -> None:
        """Set the open-position counter to *count* (used at startup to re-sync state).

        Call this once during bot initialisation after querying the exchange for
        existing open positions so that :meth:`can_open_position` reflects the
        real number of live positions rather than assuming zero.

        Parameters
        ----------
        count:
            Number of currently open positions as reported by the exchange.
            Negative values are clamped to 0.
        """
        if count < 0:
            logger.warning(
                "sync_open_count called with negative value %d – clamping to 0.",
                count,
            )
            count = 0
        self._open_count = count
        logger.info("Open-position counter synchronised to %d.", count)

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

