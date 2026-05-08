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


# ── Portfolio risk parameters ──────────────────────────────────────────────────
MAX_PORTFOLIO_RISK_PCT: float = 0.10 # 10% maximum capital at risk across all positions
DRAWDOWN_HALFRISK_THRESHOLD: float = 0.05 # Reduce risk by 50% if weekly DD > 5%

class RiskManager:
    """Calculate position sizes for MetaTrader 5 with portfolio-level risk control.

    Parameters
    ----------
    initial_balance:
        Starting simulated balance in quote currency (e.g. USDT).
    max_positions:
        Maximum number of positions that may be open at the same time.
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
        
        # [PRO] Portfolio-level risk tracking
        self._weekly_start_balance: float = initial_balance
        self._total_risk_usd: float = 0.0 # Sum of (entry - SL) * lots for all open positions

    def get_current_risk_exposure(self) -> float:
        """Return the current percentage of capital at risk."""
        if self.balance <= 0: return 1.0
        return self._total_risk_usd / self.balance

    def calculate_position_size(
        self,
        win_probability: float,
        *,
        risk_pct: float | None = None,
        sl_distance_pct: float = 0.02,
    ) -> float:
        """Return the position size in quote currency for the next trade.

        Includes dynamic risk reduction based on drawdown and portfolio exposure.
        """
        base_r = RISK_PER_TRADE if risk_pct is None else float(risk_pct)
        
        # 1. Drawdown 'Thermometer' - reduce risk if weekly performance is poor
        weekly_dd = (self._weekly_start_balance - self.balance) / self._weekly_start_balance
        if weekly_dd > DRAWDOWN_HALFRISK_THRESHOLD:
            base_r *= 0.5
            logger.warning(
                "Risk reduced by 50%% due to weekly drawdown (%.2f%% > %.2f%%)",
                weekly_dd * 100, DRAWDOWN_HALFRISK_THRESHOLD * 100
            )

        # 2. Portfolio Exposure Check
        current_exposure = self.get_current_risk_exposure()
        if current_exposure + base_r > MAX_PORTFOLIO_RISK_PCT:
            # Scale down base_r to fit remaining risk budget
            base_r = max(0.0, MAX_PORTFOLIO_RISK_PCT - current_exposure)
            if base_r < 0.005: # Minimum 0.5% risk to bother opening
                logger.warning("Portfolio risk budget full (%.2f%%). Skipping trade.", current_exposure * 100)
                return 0.0
            logger.info("Risk scaled to %.2f%% to fit portfolio budget.", base_r * 100)

        # 3. Size by SL distance: risk_amount = balance * base_r
        # position_size = risk_amount / sl_distance_pct
        position_size = (self.balance * base_r) / max(0.001, sl_distance_pct)
        
        # Cap allocation to an equal share of the current balance
        max_allocation = (self.balance / self.max_positions) * LEVERAGE
        position_size = min(position_size, max_allocation)
        
        logger.debug(
            "position_size=%.2f  balance=%.2f  risk_per_trade=%.4f  max_allocation=%.2f",
            position_size, self.balance, base_r, max_allocation,
        )
        return position_size

    def register_open(self, risk_usd: float) -> None:
        """Increment open count and track total risk."""
        self._open_count += 1
        self._total_risk_usd += max(0.0, risk_usd)

    def register_close(self, risk_usd: float) -> None:
        """Decrement open count and release risk budget."""
        if self._open_count > 0:
            self._open_count -= 1
        self._total_risk_usd = max(0.0, self._total_risk_usd - risk_usd)

    @property
    def open_count(self) -> int:
        """Current number of open positions."""
        return self._open_count

    def can_open_position(self) -> bool:
        """Return *True* if another position may be opened (below max_positions)."""
        return self._open_count < self.max_positions

    def sync_open_count(self, count: int) -> None:
        """Set the open-position counter to *count*."""
        self._open_count = max(0, count)
        logger.info("Open-position counter synchronised to %d.", self._open_count)

    def recalc_total_risk(self, positions: dict) -> None:
        """Recalculate total risk USD from open positions."""
        self._total_risk_usd = sum(
            max(0.0, pos.position_size * pos.sl_pct)
            for pos in positions.values()
        )

    def is_trading_halted(self) -> bool:
        """Return *True* if trading has been halted due to the daily loss limit."""
        if self._trading_halted_until is None:
            return False
        now = datetime.now(tz=timezone.utc)
        if now >= self._trading_halted_until:
            self._trading_halted_until = None
            return False
        return True

    def is_portfolio_dd_exceeded(self) -> bool:
        """Return *True* when the account has lost more than MAX_PORTFOLIO_DD_PCT."""
        if self.balance < self._portfolio_dd_floor:
            return True
        return False

    def record_daily_loss(self, loss: float) -> None:
        """Accumulate a realised loss and trigger the safety break if needed."""
        if loss <= 0.0: return
        self._daily_loss += loss
        threshold = self._daily_start_balance * MAX_DAILY_LOSS_PCT
        if self._daily_loss >= threshold and self._trading_halted_until is None:
            self._trading_halted_until = datetime.now(tz=timezone.utc) + _HALT_DURATION

    def reset_daily_stats(self) -> None:
        """Reset daily-loss counters."""
        self._daily_start_balance = self.balance
        self._daily_loss = 0.0
        self._trading_halted_until = None

    def is_sector_exposed(self, symbol: str, open_symbols: list[str]) -> bool:
        """Return *True* if any symbol in *open_symbols* shares the sector of *symbol*."""
        target_sector = get_sector(symbol)
        if target_sector == _SECTOR_UNCLASSIFIED:
            return False
        return any(get_sector(s) == target_sector for s in open_symbols)

    def has_sufficient_balance(self, position_size: float) -> bool:
        """Return *True* if the current balance can cover the margin for *position_size*."""
        return self.balance >= (position_size / LEVERAGE) > 0.0

    def deduct(self, amount: float) -> None:
        """Subtract *amount* from the simulated balance (trade entry)."""
        self.balance -= amount

    def credit(self, amount: float) -> None:
        """Add *amount* to the simulated balance (trade close + PnL)."""
        self.balance += amount

