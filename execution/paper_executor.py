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
import os
import uuid
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.constants import DEBUG_LOG_HINT
from database.db_manager import DatabaseManager
from risk import risk_manager as _rm_mod
from risk.risk_manager import RiskManager, get_execution_thresholds

_SL_CAP_FRAC = float(getattr(_rm_mod, "_SL_CAP", 0.05))
from strategy.ml_predictor import BUY_PROB_THRESHOLD, get_symbol_config
from utils.telegram_notifier import send_telegram_alert

logger = logging.getLogger(__name__)


def _ttl_hours_from_env() -> float:
    raw = os.environ.get("TTL_HOURS", os.environ.get("POSITION_TTL_HOURS", "")).strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 12.0


POSITION_TTL_HOURS: float = _ttl_hours_from_env()

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
# Stop inicial largos: distancia_sl ≈ ATR_SL_MULTIPLIER × ATR (tope risk_manager._SL_CAP).
ATR_SL_MULTIPLIER: float = 2.0
ATR_TRAILING_MULTIPLIER: float = 2.0

# ── Error messages ───────────────────────────────────────────────────────────
_ERRORS = {
    "MARKET_CLOSED": "MARKET_CLOSED: La sesión de trading para este símbolo no está abierta.",
    "INSUFFICIENT_FUNDS": "INSUFFICIENT_FUNDS: No hay suficiente balance para abrir la posición.",
    "MAX_POSITIONS": "MAX_POSITIONS: Se ha alcanzado el límite máximo de posiciones abiertas.",
}


@dataclass
class OpenPosition:
    """State for one open position (paper + MT5 bookkeeping)."""

    trade_id: str | int
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size: float
    sl_price: float
    activation_price: float
    trailing_distance_pct: float
    peak_price: float
    tp_price: float | None = None # [NEW] Dynamic Take Profit
    time_limit_reached: bool = False # [NEW] Time-based logic
    trailing_stop_active: bool = False
    ml_confidence: float = 0.0
    sl_pct: float = 0.025
    activation_pct: float = 0.02
    stop_loss_price: float = 0.0
    atr_trailing_distance: float = 0.0
    atr_at_entry: float | None = None # [NEW] Store ATR for history
    timeframe: str = "15m" # [NEW] Track timeframe
    sentiment_score: float = 0.0
    mt5_position_ticket: int | None = None
    current_stop_loss: float = 0.0
    last_broker_sl_synced: float = 0.0
    last_broker_tp_synced: float = 0.0
    last_mt5_modify_mono: float = 0.0
    close_pending: bool = False
    last_close_error: str | None = None

    def __post_init__(self) -> None:
        if self.stop_loss_price <= 0.0 and self.sl_price > 0.0:
            object.__setattr__(self, "stop_loss_price", self.sl_price)
        if self.current_stop_loss <= 0.0 and self.sl_price > 0.0:
            object.__setattr__(self, "current_stop_loss", self.sl_price)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpenPosition":
        raw = dict(d)
        raw["entry_time"] = datetime.fromisoformat(raw["entry_time"])
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in raw:
                kwargs[f.name] = raw[f.name]
            elif f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                raise KeyError(f.name)
        return cls(**kwargs)


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

    _save_state = save_state

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
            self._risk.recalc_total_risk(self.open_positions)
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
        cfg = get_symbol_config(symbol)
        
        # Risk Management: Portfolio level
        sl_frac = min(float(cfg.get("fixed_sl_pct", 0.025)), _SL_CAP_FRAC)
        pos_size_quote = self._risk.calculate_position_size(
            win_probability, 
            risk_pct=float(cfg.get("risk", 0.02)),
            sl_distance_pct=sl_frac
        )
        
        if pos_size_quote <= 0 or not self._risk.has_sufficient_balance(pos_size_quote): 
            return False
            
        thresh = get_execution_thresholds()
        sl_price = entry_price * (1.0 - sl_frac)
        act_price = entry_price * (1.0 + thresh.activation_pct)
        
        # [PRO] ATR-based Take Profit (default 2.5x ATR)
        atr_mult = float(os.environ.get("TP_ATR_MULTIPLIER", "2.5"))
        tp_price = entry_price + (current_atr * atr_mult) if current_atr else None
        
        trade_id = uuid.uuid4().hex[:8]
        now = datetime.now(tz=timezone.utc)
        tf = cfg.get("timeframe", "15m")

        pos = OpenPosition(
            trade_id=trade_id,
            symbol=symbol,
            entry_time=now,
            entry_price=entry_price,
            position_size=pos_size_quote,
            sl_price=sl_price,
            tp_price=tp_price,
            activation_price=act_price,
            trailing_distance_pct=thresh.trailing_distance_pct,
            peak_price=entry_price,
            ml_confidence=win_probability,
            sl_pct=sl_frac,
            activation_pct=thresh.activation_pct,
            stop_loss_price=sl_price,
            atr_trailing_distance=(current_atr or 0.0) * ATR_TRAILING_MULTIPLIER if current_atr else 0.0,
            atr_at_entry=current_atr,
            timeframe=tf,
            sentiment_score=sentiment_score,
            current_stop_loss=sl_price,
        )
        
        self.open_positions[symbol] = pos
        
        # Risk accounting
        risk_usd = pos_size_quote * sl_frac
        self._risk.register_open(risk_usd=risk_usd)
        self._risk.deduct(pos_size_quote)
        
        # DB Record
        try:
            await self._db.insert_trade_open(
                trade_id=trade_id,
                timestamp_open=now,
                symbol=symbol,
                timeframe=tf,
                side="LONG",
                entry_price=entry_price,
                lots=pos_size_quote / entry_price, # rough lots for paper
                win_probability=win_probability,
                atr=current_atr
            )
        except Exception as exc:
            logger.warning("Failed to log trade open to DB: %s", exc)

        self.save_state()
        tp_str = f"TP={tp_price:.4f}" if tp_price else "TP=NONE"
        logger.info("🚀 [BUY] %s entry=%.4f size=%.2f SL=%.4f %s (ML=%.2f%%)", symbol, entry_price, pos_size_quote, sl_price, tp_str, win_probability * 100)
        return True

    async def check_and_close(self, symbol: str, current_price: float) -> str | None:
        pos = self.open_positions.get(symbol)
        if not pos: return None
        
        # 1. Update Peak and Trailing Stop
        if current_price > pos.peak_price:
            pos.peak_price = current_price
            if current_price >= pos.activation_price:
                if not pos.trailing_stop_active:
                    pos.trailing_stop_active = True
                    logger.info("📈 [TS] %s trailing stop ACTIVATED at %.4f", symbol, current_price)
                new_sl = pos.peak_price * (1.0 - pos.trailing_distance_pct)
                if new_sl > pos.sl_price:
                    pos.sl_price = new_sl
                    pos.current_stop_loss = new_sl

        # 2. Check Exits
        exit_reason = None
        
        # Dynamic Take Profit (ATR-based)
        if pos.tp_price and current_price >= pos.tp_price:
            exit_reason = "take_profit"
        # Trailing / Stop Loss
        elif current_price <= pos.sl_price: 
            exit_reason = "trailing_stop" if pos.trailing_stop_active else "stop_loss"
        
        # [PRO] Time-based exit improvement: Move SL to entry after 4 hours if in profit
        now = datetime.now(tz=timezone.utc)
        duration_h = (now - pos.entry_time).total_seconds() / 3600.0
        if not pos.trailing_stop_active and duration_h >= 4.0 and current_price > pos.entry_price:
            if pos.sl_price < pos.entry_price:
                pos.sl_price = pos.entry_price
                pos.current_stop_loss = pos.entry_price
                logger.info("⏱️ [TIME] %s duration=%.1fh: SL moved to entry (Break-even).", symbol, duration_h)

        if exit_reason:
            await self._close_position(symbol, current_price, exit_reason)
            return exit_reason
        return None

    async def check_ml_exit(self, current_price: float, ml_signal: str, ml_probability: float | None, symbol: str, min_confidence: float = BUY_PROB_THRESHOLD) -> float | None:
        pos = self.open_positions.get(symbol)
        if not pos: return None
        now = datetime.now(tz=timezone.utc)
        duration_h = (now - pos.entry_time).total_seconds() / 3600.0
        if ml_signal == "SELL" and (ml_probability is None or ml_probability >= min_confidence):
            logger.info("📉 [SELL] %s ML Reversal triggered early exit.", symbol)
            return await self._close_position(symbol, current_price, "ml_reversal")
        if duration_h >= POSITION_TTL_HOURS:
            logger.info(
                "⏱️ [TTL] %s position reached max age (%.1fh).",
                symbol,
                POSITION_TTL_HOURS,
            )
            return await self._close_position(symbol, current_price, "ttl_expiry")
        return None

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        pos = self.open_positions.pop(symbol)
        exit_time = datetime.now(tz=timezone.utc)
        
        # PnL Calculation
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        gross_pnl = pnl_pct * pos.position_size
        
        self.total_pnl += gross_pnl
        self._risk.credit(pos.position_size + gross_pnl)
        
        # Risk accounting
        risk_usd = pos.position_size * pos.sl_pct
        self._risk.register_close(risk_usd=risk_usd)
        
        if gross_pnl < 0: 
            self._risk.record_daily_loss(abs(gross_pnl))
            
        # [NEW] Persistence to trade_history table
        try:
            await self._db.update_trade_exit(
                trade_id=pos.trade_id,
                timestamp_close=exit_time,
                exit_price=exit_price,
                pnl_usd=gross_pnl,
                pnl_pct=pnl_pct * 100,
                exit_reason=reason
            )
        except Exception as exc:
            logger.warning("Failed to update trade exit in DB: %s", exc)

        self._append_to_journal(pos, exit_price, exit_time, reason)
        self.save_state()
        
        pnl_pct_display = pnl_pct * 100
        logger.info("🏁 [CLOSE] %s exit=%.4f pnl=%.2f (%.2f%%) reason=%s", symbol, exit_price, gross_pnl, pnl_pct_display, reason)
        await send_telegram_alert(f"🏁 *CLOSE* {symbol}\nExit: {exit_price:.4f}\nPnL: {gross_pnl:+.2f} ({pnl_pct_display:+.2f}%)\nReason: {reason}")
        return gross_pnl

    async def sync_positions_with_exchange(self, confirmations_required: int = 1) -> int:
        return len(self.open_positions)

def _build_trade_report(*args: Any, **kwargs: Any) -> str:
    """Legacy stub for MT5Executor."""
    return ""

def record_trade(*args: Any, **kwargs: Any) -> None:
    """Legacy stub for MT5Executor."""
    pass
