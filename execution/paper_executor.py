"""
execution.paper_executor
~~~~~~~~~~~~~~~~~~~~~~~~

Simulates order execution (paper trading).  A local trade journal
is persisted to ``logs/state.json`` and ``logs/trade_journal.csv``.

The entry logic computes stop-loss and trailing-stop thresholds
at position-open time.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.constants import DEBUG_LOG_HINT
from database.db_manager import DatabaseManager
from risk.risk_manager import RiskManager, get_execution_thresholds
from utils.telegram_notifier import send_telegram_alert

logger = logging.getLogger(__name__)

# ── Journal column headers ──────────────────────────────────────────────────
_JOURNAL_COLUMNS: list[str] = [
    "trade_id",
    "symbol",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "position_size",
    "gross_pnl",
    "net_pnl",
    "exit_reason",
    "ml_confidence_at_entry",
    "duration_minutes",
]

# ── Legacy compatibility constants ──────────────────────────────────────────
_TAKER_FEE_RATE: float = 0.0004
ATR_SL_MULTIPLIER: float = 1.5
ATR_TRAILING_MULTIPLIER: float = 2.0

# ── Error messages ───────────────────────────────────────────────────────────
_ERRORS = {
    "MARKET_CLOSED": "MARKET_CLOSED: La sesión de trading para este símbolo no está abierta.",
    "INSUFFICIENT_FUNDS": "INSUFFICIENT_FUNDS: No hay suficiente balance para abrir la posición.",
    "MAX_POSITIONS": "MAX_POSITIONS: Se ha alcanzado el límite máximo de posiciones abiertas.",
}


@dataclass
class OpenPosition:
    """State for one open paper-trading position."""

    trade_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size: float
    sl_price: float
    activation_price: float
    trailing_distance_pct: float
    peak_price: float
    trailing_stop_active: bool = False
    ml_confidence: float = 0.0

    @property
    def current_stop_loss(self) -> float:
        """Alias for sl_price to maintain Dashboard compatibility."""
        return self.sl_price

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpenPosition":
        d["entry_time"] = datetime.fromisoformat(d["entry_time"])
        return cls(**d)


def compute_dynamic_tp_hint(pos: OpenPosition) -> float | None:
    """Helper for dashboard rendering."""
    return None


class PaperExecutor:
    """Simulates order execution (paper trading)."""

    def __init__(
        self,
        db: DatabaseManager,
        risk_manager: RiskManager,
        exchange: Any | None = None,
    ) -> None:
        self._db = db
        self._risk = risk_manager
        self._exchange = exchange
        self.open_positions: dict[str, OpenPosition] = {}
        self.total_pnl: float = 0.0
        self._state_file = Path("logs/state.json")
        self._journal_file = Path("logs/trade_journal.csv")

    def save_state(self) -> None:
        state = {
            "total_pnl": self.total_pnl,
            "positions": {s: p.to_dict() for s, p in self.open_positions.items()},
        }
        try:
            self._state_file.parent.mkdir(exist_ok=True)
            with self._state_file.open("w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save state: %s", exc)

    def load_state(self) -> int:
        if not self._state_file.exists():
            return 0
        try:
            with self._state_file.open() as f:
                state = json.load(f)
            self.total_pnl = state.get("total_pnl", 0.0)
            raw_pos = state.get("positions", {})
            for s, d in raw_pos.items():
                self.open_positions[s] = OpenPosition.from_dict(d)
            self._risk.sync_open_count(len(self.open_positions))
            return len(self.open_positions)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load state: %s", exc)
            return 0

    def _append_to_journal(self, pos: OpenPosition, exit_price: float, exit_time: datetime, reason: str) -> None:
        gross_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.position_size
        net_pnl = gross_pnl
        duration = (exit_time - pos.entry_time).total_seconds() / 60.0
        row = [pos.trade_id, pos.symbol, pos.entry_time.isoformat(), exit_time.isoformat(), f"{pos.entry_price:.8f}", f"{exit_price:.8f}", f"{pos.position_size:.2f}", f"{gross_pnl:.4f}", f"{net_pnl:.4f}", reason, f"{pos.ml_confidence:.4f}", f"{duration:.1f}"]
        try:
            write_header = not self._journal_file.exists()
            with self._journal_file.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(_JOURNAL_COLUMNS)
                writer.writerow(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append to journal: %s", exc)

    async def try_open_trade(self, entry_price: float, win_probability: float, symbol: str, sentiment_score: float = 0.0, current_atr: float | None = None) -> bool:
        if symbol in self.open_positions: return False
        if not self._risk.can_open_position(): return False
        pos_size_quote = self._risk.calculate_position_size(win_probability)
        if not self._risk.has_sufficient_balance(pos_size_quote): return False
        thresh = get_execution_thresholds()
        sl_price = entry_price * (1.0 - thresh.sl_pct)
        act_price = entry_price * (1.0 + thresh.activation_pct)
        pos = OpenPosition(trade_id=uuid.uuid4().hex[:8], symbol=symbol, entry_time=datetime.now(tz=timezone.utc), entry_price=entry_price, position_size=pos_size_quote, sl_price=sl_price, activation_price=act_price, trailing_distance_pct=thresh.trailing_distance_pct, peak_price=entry_price, ml_confidence=win_probability)
        self.open_positions[symbol] = pos
        self._risk.register_open()
        self._risk.deduct(pos_size_quote)
        self.save_state()
        logger.info("🚀 [BUY] %s entry=%.4f size=%.2f SL=%.4f ACT=%.4f (ML=%.2f%%)", symbol, entry_price, pos_size_quote, sl_price, act_price, win_probability * 100)
        return True

    async def check_and_close(self, symbol: str, current_price: float) -> str | None:
        pos = self.open_positions.get(symbol)
        if not pos: return None
        if current_price > pos.peak_price:
            pos.peak_price = current_price
            if current_price >= pos.activation_price:
                if not pos.trailing_stop_active:
                    pos.trailing_stop_active = True
                    logger.info("📈 [TS] %s trailing stop ACTIVATED at %.4f", symbol, current_price)
                new_sl = pos.peak_price * (1.0 - pos.trailing_distance_pct)
                if new_sl > pos.sl_price: pos.sl_price = new_sl
        exit_reason = None
        if current_price <= pos.sl_price: exit_reason = "trailing_stop" if pos.trailing_stop_active else "stop_loss"
        if exit_reason:
            await self._close_position(symbol, current_price, exit_reason)
            return exit_reason
        return None

    async def check_ml_exit(self, current_price: float, ml_signal: str, ml_probability: float | None, symbol: str, min_confidence: float = 0.70) -> float | None:
        pos = self.open_positions.get(symbol)
        if not pos: return None
        now = datetime.now(tz=timezone.utc)
        duration_h = (now - pos.entry_time).total_seconds() / 3600.0
        if ml_signal == "SELL" and (ml_probability is None or ml_probability >= min_confidence):
            logger.info("📉 [SELL] %s ML Reversal triggered early exit.", symbol)
            return await self._close_position(symbol, current_price, "ml_reversal")
        if duration_h >= 12.0:
            logger.info("⏱️ [TTL] %s position reached max age (12h).", symbol)
            return await self._close_position(symbol, current_price, "ttl_expiry")
        return None

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        pos = self.open_positions.pop(symbol)
        exit_time = datetime.now(tz=timezone.utc)
        gross_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.position_size
        self.total_pnl += gross_pnl
        self._risk.credit(pos.position_size + gross_pnl)
        self._risk.register_close()
        if gross_pnl < 0: self._risk.record_daily_loss(abs(gross_pnl))
        self._append_to_journal(pos, exit_price, exit_time, reason)
        self.save_state()
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        logger.info("🏁 [CLOSE] %s exit=%.4f pnl=%.2f (%.2f%%) reason=%s", symbol, exit_price, gross_pnl, pnl_pct, reason)
        await send_telegram_alert(f"🏁 *CLOSE* {symbol}\nExit: {exit_price:.4f}\nPnL: {gross_pnl:+.2f} ({pnl_pct:+.2f}%)\nReason: {reason}")
        return gross_pnl

    async def sync_positions_with_exchange(self, confirmations_required: int = 1) -> int:
        return len(self.open_positions)

def _build_trade_report(*args: Any, **kwargs: Any) -> str:
    """Legacy stub for MT5Executor."""
    return ""

def record_trade(*args: Any, **kwargs: Any) -> None:
    """Legacy stub for MT5Executor."""
    pass
