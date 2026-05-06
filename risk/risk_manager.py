"""
risk.risk_manager
~~~~~~~~~~~~~~~~~

Position sizing for Binance Futures with leverage and daily-loss safety break.

Position size formula::

    position_size = balance * RISK_PER_TRADE * LEVERAGE

where:
* ``RISK_PER_TRADE`` is the fraction of the balance risked per trade.
* ``LEVERAGE``       is the futures leverage multiplier (5×).

Trailing stop parameters
------------------------
* ``INITIAL_SL``:        Hard stop loss during the entry phase (2.5 %).
* ``ACTIVATION_PCT``:    Minimum profit required to activate the trailing
                         stop (2.0 %).  Once this threshold is reached the
                         active stop loss updates dynamically.
* ``TRAILING_DISTANCE``: Gap maintained between the running peak price and
                         the trailing stop level (2.0 %).

Daily-loss safety break
-----------------------
If the total realised loss for the current UTC day exceeds
``MAX_DAILY_LOSS_PCT`` (3 %) of the account balance at the start of the
day, all new trades are blocked for 24 hours.  Call
:meth:`reset_daily_stats` once a day (e.g. at midnight UTC) to lift the
block and refresh the reference balance.

Portfolio drawdown circuit-breaker
-----------------------------------
A second, session-level safety net protects the starting capital from
"black swan" events.  If the running balance ever falls more than
``MAX_PORTFOLIO_DD_PCT`` (15 %) below the balance recorded at
instantiation time, :meth:`is_portfolio_dd_exceeded` returns *True* and
all new positions are permanently blocked for that session.  Unlike the
daily-loss halt this guard is **not** reset at midnight – it requires a
deliberate bot restart with fresh capital to resume trading.

Multi-asset risk controls:

* ``max_positions`` limits the number of simultaneously open positions.
* Each trade's position size is capped at ``balance / max_positions`` so
  that no single trade can consume more than an equal share of the
  portfolio.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


# ── Trade parameters ──────────────────────────────────────────────────────────
INITIAL_SL: float = 0.025          # 2.5 % hard stop loss for initial protection
ACTIVATION_PCT: float = 0.02       # 2.0 % profit to activate trailing stop (module-level legacy)
TRAILING_DISTANCE: float = 0.02    # 2.0 % trailing gap from peak once active

# ── Futures / leverage parameters ─────────────────────────────────────────────
LEVERAGE: int = 5                   # 5× futures leverage
RISK_PER_TRADE: float = _float_env("RISK_PER_TRADE", 0.05)  # default 5 % (max-performance profile)

# ── Daily-loss safety break ───────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT: float = 0.03    # 3 % maximum daily loss before halting
_HALT_DURATION: timedelta = timedelta(hours=24)

MAX_POSITIONS: int = _int_env("MAX_POSITIONS", 2)  # Maximum simultaneous open positions

# ── Portfolio drawdown circuit-breaker ───────────────────────────────────────
MAX_PORTFOLIO_DD_PCT: float = 0.15  # 15 % max peak-to-trough loss from initial balance

# ── Sector / correlation-group mapping ────────────────────────────────────────
# Used by :func:`get_sector` and :meth:`RiskManager.is_sector_exposed` to
# prevent opening more than one position within the same correlated group.
SECTOR_MAP: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
    "LINKUSDT": "DEFI",
    "INJUSDT": "DEFI",
    "FETUSDT": "AI",
    "RENDERUSDT": "AI",
    "DOGEUSDT": "MEME",
    "PEPEUSDT": "MEME",
    "PAXGUSDT": "METALS",
    "XAUUSD": "METALS",
}

_SECTOR_UNCLASSIFIED: str = "UNCLASSIFIED"


def get_sector(symbol: str) -> str:
    """Return the correlation sector for *symbol*.

    Strips the ``/`` separator so that both ``"BTCUSDT"`` and ``"BTC/USDT"``
    resolve correctly against :data:`SECTOR_MAP`.

    Returns ``"UNCLASSIFIED"`` when the symbol is not listed in the map.
    """
    normalised = symbol.replace("/", "").upper()
    return SECTOR_MAP.get(normalised, _SECTOR_UNCLASSIFIED)


# ── Dynamic risk management – base thresholds (neutral market) ────────────────
BASE_SL: float = 0.025                      # 2.5 % base stop loss
BASE_ACTIVATION_PCT: float = 0.020          # 2.0 % profit to activate trailing stop
BASE_TRAILING_DISTANCE: float = 0.020       # 2.0 % trailing distance

# ── Dynamic risk management – sentiment multiplier bounds ─────────────────────
_SENTIMENT_LOW_THRESHOLD: float = 0.30      # below this → scalping regime
_SENTIMENT_HIGH_THRESHOLD: float = 0.60    # above this → swing-trading regime
_MULTIPLIER_LOW: float = 0.8               # shrink thresholds in low-sentiment markets
_MULTIPLIER_HIGH: float = 1.8              # expand thresholds in high-sentiment markets
_MULTIPLIER_MIN: float = 0.5               # absolute floor (safety net)
_MULTIPLIER_MAX: float = 2.5               # absolute cap (safety net)
_SL_CAP: float = 0.05                      # maximum allowed stop-loss fraction (5 %)
_ACTIVATION_PCT_MIN: float = 0.005         # Floor lowered to 0.5 % to allow 1.0 % strategy (covers 0.08 % fees)


@dataclass
class DynamicThresholds:
    """Risk-management thresholds adjusted by an AI sentiment multiplier."""

    sl_pct: float
    activation_pct: float
    trailing_distance_pct: float
    multiplier: float


def get_execution_thresholds() -> DynamicThresholds:
    """Fixed execution thresholds without LLM sentiment (simplified live path).

    Uses :data:`BASE_SL`, :data:`BASE_ACTIVATION_PCT`, and
    :data:`BASE_TRAILING_DISTANCE` directly so stops are predictable and wide
    enough to avoid structural noise exits.
    """
    activation_pct = max(BASE_ACTIVATION_PCT, _ACTIVATION_PCT_MIN)
    return DynamicThresholds(
        sl_pct=min(BASE_SL, _SL_CAP),
        activation_pct=activation_pct,
        trailing_distance_pct=BASE_TRAILING_DISTANCE,
        multiplier=1.0,
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
        # Portfolio drawdown circuit-breaker – fixed reference, never modified
        self._initial_balance: float = initial_balance
        self._portfolio_dd_floor: float = initial_balance * (1.0 - MAX_PORTFOLIO_DD_PCT)
        # Daily-loss tracking
        self._daily_start_balance: float = initial_balance
        self._daily_loss: float = 0.0
        self._trading_halted_until: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        win_probability: float,  # noqa: ARG002
        *,
        risk_pct: float | None = None,
    ) -> float:
        """Return the position size in quote currency for the next trade.

        Uses the formula ``balance * risk_pct * LEVERAGE`` (default ``RISK_PER_TRADE``).

        The position size is also capped at ``balance / max_positions`` so
        that no single trade consumes more than a fair share of the
        portfolio.

        Parameters
        ----------
        win_probability:
            ML-predicted probability of the trade being profitable (0–1).
            Accepted for API compatibility; not used in the leverage-based
            formula.
        risk_pct:
            Optional per-symbol risk fraction; defaults to :data:`RISK_PER_TRADE`.
        """
        r = RISK_PER_TRADE if risk_pct is None else float(risk_pct)
        position_size = self.balance * r * LEVERAGE
        # Cap allocation to an equal share of the current balance
        max_allocation = self.balance / self.max_positions
        position_size = min(position_size, max_allocation)
        logger.debug(
            "position_size=%.2f  balance=%.2f  leverage=%d  risk_per_trade=%.4f  max_allocation=%.2f",
            position_size,
            self.balance,
            LEVERAGE,
            r,
            max_allocation,
        )
        return position_size

    def can_open_position(self) -> bool:
        """Return *True* if another position may be opened (below max_positions)."""
        return self._open_count < self.max_positions

    def is_sector_exposed(self, symbol: str, open_symbols: list[str]) -> bool:
        """Return *True* if any symbol in *open_symbols* shares the sector of *symbol*.

        Used to enforce a maximum of one open position per correlation group
        (e.g. at most one L1 Major at a time: BTC, ETH, SOL, BNB).

        Parameters
        ----------
        symbol:
            The candidate symbol being considered for a new position.
        open_symbols:
            Symbols that currently have an open position.
        """
        target_sector = get_sector(symbol)
        # Unclassified symbols are never blocked by sector exposure.
        if target_sector == _SECTOR_UNCLASSIFIED:
            return False
        return any(get_sector(s) == target_sector for s in open_symbols)

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

    # ------------------------------------------------------------------
    # Portfolio drawdown circuit-breaker
    # ------------------------------------------------------------------

    def is_portfolio_dd_exceeded(self) -> bool:
        """Return *True* when the account has lost more than ``MAX_PORTFOLIO_DD_PCT``
        of the balance recorded at instantiation time.

        This session-level guard fires on "black swan" events (e.g. a flash
        crash that triggers multiple stop-losses in quick succession).  Unlike
        the daily-loss halt it is **not** auto-lifted at midnight; a bot
        restart with fresh capital is required to resume trading.

        For a $50 live account the default 15 % threshold corresponds to a
        $7.50 maximum tolerable loss before the engine goes fully defensive.
        """
        if self.balance < self._portfolio_dd_floor:
            logger.critical(
                "🚨 [CIRCUIT BREAKER] Portfolio drawdown limit breached – "
                "balance=%.2f initial=%.2f threshold=%.0f%% (floor=%.2f). "
                "All new positions BLOCKED for this session.",
                self.balance,
                self._initial_balance,
                MAX_PORTFOLIO_DD_PCT * 100,
                self._portfolio_dd_floor,
            )
            return True
        return False

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

