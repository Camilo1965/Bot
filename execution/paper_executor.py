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
import time
import uuid
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bot.constants import DEBUG_LOG_HINT
from bot.observability import TradeMetrics, ExitMetrics, get_buffer
from database.db_manager import DatabaseManager
from risk import risk_manager as _rm_mod
from risk.risk_manager import RiskManager, BASE_SL, get_execution_thresholds

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
ATR_SL_MULTIPLIER: float = 2.0
ATR_TRAILING_MULTIPLIER: float = 2.0

def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

_DEFAULT_TP_PCT: float = _float_env("DEFAULT_TP_PCT", 0.03)
_BREAK_EVEN_HOURS: float = _float_env("BREAK_EVEN_HOURS", 4.0)
_ATR_TP_MULT: float = _float_env("ATR_TP_MULT", 2.5)

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
    direction: str = "long"  # "long" or "short"
    partial_exit_done: bool = False

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
            tmp = self._state_file.with_suffix(".tmp")
            with tmp.open("w") as f:
                json.dump(state, f, indent=2)
            tmp.replace(self._state_file)
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
            cleaned = 0
            for s, d in raw_pos.items():
                pos = OpenPosition.from_dict(d)
                # Discard orphan positions that were never persisted to DB
                # (numeric trade_id == legacy ghost position).
                tid = getattr(pos, "trade_id", None)
                if isinstance(tid, str) and tid.isdigit():
                    logger.debug("Discarding orphan position %s (trade_id=%s) from state.json.", s, tid)
                    cleaned += 1
                    continue
                self.open_positions[s] = pos
            if cleaned:
                logger.info("Cleaned %d orphan position(s) from state.json.", cleaned)
                self.save_state()
            self._risk.sync_open_count(len(self.open_positions))
            self._risk.recalc_total_risk(self.open_positions)
            return len(self.open_positions)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            backup = self._state_file.with_suffix(".json.corrupted")
            try:
                self._state_file.rename(backup)
                logger.warning(
                    "State file corrupted (%s) - backed up to %s, starting fresh.",
                    exc, backup,
                )
            except Exception:
                logger.warning("State file corrupted (%s) - could not backup, removing.", exc)
                try:
                    self._state_file.unlink()
                except Exception:
                    pass
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load state: %s", exc)
            return 0

    def _append_to_journal(self, pos: OpenPosition, exit_price: float, exit_time: datetime, reason: str) -> None:
        gross_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.position_size
        net_pnl = gross_pnl
        duration = (exit_time - pos.entry_time).total_seconds() / 60.0
        row = [pos.trade_id, pos.symbol, pos.entry_time.isoformat(), exit_time.isoformat(), f"{pos.entry_price:.8f}", f"{exit_price:.8f}", f"{pos.position_size:.2f}", f"{gross_pnl:.4f}", f"{net_pnl:.4f}", reason, f"{pos.ml_confidence:.4f}", f"{duration:.1f}"]
        try:
            self._journal_file.parent.mkdir(exist_ok=True)
            write_header = not self._journal_file.exists()
            with self._journal_file.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(_JOURNAL_COLUMNS)
                writer.writerow(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append to journal: %s", exc)

    async def try_open_trade(self, entry_price: float, win_probability: float, symbol: str, sentiment_score: float = 0.0, current_atr: float | None = None, vibe_quality: float = 1.0) -> bool:
        if symbol in self.open_positions: return False
        if not self._risk.can_open_position(): return False
        cfg = get_symbol_config(symbol)
        
        # Risk Management: Portfolio level
        # ATR-based dynamic SL: sl = clamp(ATR_SL_MULT * atr_pct, cfg_sl, SL_CAP)
        cfg_fixed_sl = float(cfg.get("fixed_sl_pct", BASE_SL))
        atr_pct = (current_atr / entry_price) if (current_atr and entry_price > 0.0) else None
        thresh = get_execution_thresholds(atr_pct=atr_pct)
        sl_frac = min(max(thresh.sl_pct, cfg_fixed_sl), _SL_CAP_FRAC)

        pos_size_quote = self._risk.calculate_position_size(
            win_probability,
            risk_pct=float(cfg.get("risk", 0.02)),
            sl_distance_pct=sl_frac,
            vibe_quality=vibe_quality,
        )

        if pos_size_quote <= 0 or not self._risk.has_sufficient_balance(pos_size_quote):
            return False

        sl_price = entry_price * (1.0 - sl_frac)
        act_price = entry_price * (1.0 + thresh.activation_pct)

        # Take Profit: max(fixed_tp_pct, ATR_TP_MULT × atr_pct) — widens TP in high-volatility
        _fixed_tp = float(cfg.get("fixed_tp_pct", _DEFAULT_TP_PCT))
        if current_atr and entry_price > 0.0:
            _atr_tp = (_ATR_TP_MULT * current_atr) / entry_price
            tp_frac = max(_fixed_tp, _atr_tp)
        else:
            tp_frac = _fixed_tp
        tp_price = entry_price * (1.0 + tp_frac)
        
        trade_id = str(uuid.uuid4())
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

        # Observability: trade opened
        get_buffer().emit_trade(
            TradeMetrics(
                symbol=symbol,
                timestamp=time.time(),
                side="LONG",
                entry_price=round(entry_price, 4),
                position_size=round(pos_size_quote, 4),
                win_probability=round(win_probability, 4),
                atr=round(current_atr, 6) if current_atr else None,
                vibe_quality=round(vibe_quality, 4),
                sl_distance_pct=round(sl_frac * 100, 2),
            )
        )

        self.save_state()
        tp_str = f"TP={tp_price:.4f}" if tp_price else "TP=NONE"
        logger.info("[BUY] %s entry=%.4f size=%.2f SL=%.4f %s (ML=%.2f%%)", symbol, entry_price, pos_size_quote, sl_price, tp_str, win_probability * 100)
        return True

    async def try_open_short_trade(
        self,
        entry_price: float,
        win_probability: float,
        symbol: str,
        current_atr: float | None = None,
        vibe_quality: float = 1.0,
    ) -> bool:
        """Open a SHORT (sell) position. Symmetric to try_open_trade but inverted TP/SL/trail."""
        if symbol in self.open_positions:
            return False
        if not self._risk.can_open_position():
            return False
        cfg = get_symbol_config(symbol)

        cfg_fixed_sl = float(cfg.get("fixed_sl_pct", BASE_SL))
        atr_pct = (current_atr / entry_price) if (current_atr and entry_price > 0.0) else None
        thresh = get_execution_thresholds(atr_pct=atr_pct)
        sl_frac = min(max(thresh.sl_pct, cfg_fixed_sl), _SL_CAP_FRAC)

        pos_size_quote = self._risk.calculate_position_size(
            win_probability,
            risk_pct=float(cfg.get("risk", 0.02)),
            sl_distance_pct=sl_frac,
            vibe_quality=vibe_quality,
        )
        if pos_size_quote <= 0 or not self._risk.has_sufficient_balance(pos_size_quote):
            return False

        # SHORT: SL is ABOVE entry, TP is BELOW entry
        sl_price = entry_price * (1.0 + sl_frac)
        _fixed_tp = float(cfg.get("fixed_tp_pct", _DEFAULT_TP_PCT))
        if current_atr and entry_price > 0.0:
            _atr_tp = (_ATR_TP_MULT * current_atr) / entry_price
            tp_frac = max(_fixed_tp, _atr_tp)
        else:
            tp_frac = _fixed_tp
        tp_price = entry_price * (1.0 - tp_frac)

        # Activation: price must fall by activation_pct before trailing kicks in
        activation_pct = thresh.activation_pct
        activation_price = entry_price * (1.0 - activation_pct)

        trade_id = str(uuid.uuid4())
        self._risk.debit(pos_size_quote)
        risk_usd = pos_size_quote * sl_frac
        self._risk.register_open(risk_usd=risk_usd)

        tf = cfg.get("timeframe", "15m")
        pos = OpenPosition(
            trade_id=trade_id,
            symbol=symbol,
            entry_time=datetime.now(tz=timezone.utc),
            entry_price=entry_price,
            position_size=pos_size_quote,
            sl_price=sl_price,
            activation_price=activation_price,
            trailing_distance_pct=thresh.trailing_pct,
            peak_price=entry_price,  # for SHORT: track minimum (starts at entry)
            tp_price=tp_price,
            ml_confidence=win_probability,
            sl_pct=sl_frac,
            activation_pct=activation_pct,
            stop_loss_price=sl_price,
            current_stop_loss=sl_price,
            atr_at_entry=current_atr,
            timeframe=tf,
            direction="short",
        )
        self.open_positions[symbol] = pos

        try:
            await self._db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                entry_price=entry_price,
                position_size=pos_size_quote,
                sl_price=sl_price,
                tp_price=tp_price,
                direction="short",
            )
        except Exception as exc:
            logger.warning("DB save_trade (short) failed: %s", exc)

        self.save_state()
        logger.info("[SELL SHORT] %s entry=%.4f size=%.2f SL=%.4f TP=%.4f (ML=%.2f%%)", symbol, entry_price, pos_size_quote, sl_price, tp_price, win_probability * 100)
        return True

    async def _partial_close_position(
        self,
        symbol: str,
        exit_price: float,
        close_fraction: float,
        reason: str,
    ) -> float:
        """Close *close_fraction* of an open position, credit partial PnL, move SL to break-even."""
        pos = self.open_positions.get(symbol)
        if not pos:
            return 0.0
        partial_size = pos.position_size * close_fraction
        remaining_size = pos.position_size * (1.0 - close_fraction)
        is_short = getattr(pos, "direction", "long") == "short"
        if pos.entry_price <= 0.0:
            pnl_pct = 0.0
        elif is_short:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        else:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        partial_pnl = pnl_pct * partial_size

        # Append journal entry BEFORE modifying pos.position_size
        partial_pos_snapshot = OpenPosition(
            trade_id=str(pos.trade_id) + "_p",
            symbol=pos.symbol,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            position_size=partial_size,
            sl_price=pos.sl_price,
            activation_price=pos.activation_price,
            trailing_distance_pct=pos.trailing_distance_pct,
            peak_price=pos.peak_price,
            ml_confidence=pos.ml_confidence,
            sl_pct=pos.sl_pct,
        )
        self._append_to_journal(partial_pos_snapshot, exit_price, datetime.now(tz=timezone.utc), reason)

        # Update accounting
        self.total_pnl += partial_pnl
        self._risk.credit(partial_size + partial_pnl)
        self._risk.register_close(risk_usd=partial_size * pos.sl_pct)
        if partial_pnl < 0:
            self._risk.record_daily_loss(abs(partial_pnl))

        # Shrink position and mark partial done
        pos.position_size = remaining_size
        pos.partial_exit_done = True

        # Move SL to break-even
        if is_short:
            if pos.sl_price > pos.entry_price:
                pos.sl_price = pos.entry_price
                pos.current_stop_loss = pos.entry_price
                pos.stop_loss_price = pos.entry_price
        else:
            if pos.sl_price < pos.entry_price:
                pos.sl_price = pos.entry_price
                pos.current_stop_loss = pos.entry_price
                pos.stop_loss_price = pos.entry_price

        self.save_state()
        logger.info(
            "📤 [PARTIAL 1R] %s exit=%.4f pnl=%.2f (%.2f%%) remaining=%.2f SL→BE=%.4f",
            symbol, exit_price, partial_pnl, pnl_pct * 100, remaining_size, pos.entry_price,
        )
        await send_telegram_alert(
            f"📤 *PARTIAL 1R* {symbol}\n"
            f"Exit {close_fraction:.0%}: {exit_price:.4f}\n"
            f"PnL: {partial_pnl:+.2f} ({pnl_pct * 100:+.2f}%)\n"
            f"Remaining: {remaining_size:.2f} | SL → Break-even"
        )
        return partial_pnl

    async def check_and_close(self, symbol: str, current_price: float) -> str | None:
        pos = self.open_positions.get(symbol)
        if not pos: return None
        
        # 1. Update Best Price and Trailing Stop (direction-aware)
        is_short = getattr(pos, "direction", "long") == "short"
        if is_short:
            # SHORT: "peak" = trough (lowest price = best for short)
            gain = (pos.entry_price - current_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            if current_price < pos.peak_price:
                pos.peak_price = current_price
                if current_price <= pos.activation_price:
                    if not pos.trailing_stop_active:
                        pos.trailing_stop_active = True
                        logger.info("[TS SHORT] %s trailing stop ACTIVATED at %.4f", symbol, current_price)
                    new_sl = pos.peak_price * (1.0 + pos.trailing_distance_pct)
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl
                        pos.current_stop_loss = new_sl
                        pos.stop_loss_price = new_sl
        else:
            gain = (current_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            logger.debug(
                "TRAIL CHECK sym=%s gain=%.4f activation=%.4f trail=%s peak=%.4f",
                symbol, gain, pos.activation_pct, pos.trailing_stop_active, pos.peak_price,
            )
            if current_price > pos.peak_price:
                pos.peak_price = current_price
                if current_price >= pos.activation_price:
                    if not pos.trailing_stop_active:
                        pos.trailing_stop_active = True
                        logger.info("[TS] %s trailing stop ACTIVATED at %.4f", symbol, current_price)
                    new_sl = pos.peak_price * (1.0 - pos.trailing_distance_pct)
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
                        pos.current_stop_loss = new_sl
                        logger.info("SL RATCHET %s: %.4f new_sl=%.4f (peak=%.4f)", symbol, pos.stop_loss_price, new_sl, pos.peak_price)
                        pos.stop_loss_price = new_sl

        # 1b. Partial exit at +1R: close 50% and move SL to break-even
        if not pos.partial_exit_done:
            if is_short:
                one_r_price = pos.entry_price * (1.0 - pos.sl_pct)
                if current_price <= one_r_price:
                    await self._partial_close_position(symbol, current_price, 0.5, "partial_1R")
            else:
                one_r_price = pos.entry_price * (1.0 + pos.sl_pct)
                if current_price >= one_r_price:
                    await self._partial_close_position(symbol, current_price, 0.5, "partial_1R")

        # 2. Check Exits
        exit_reason = None
        if is_short:
            if pos.tp_price and current_price <= pos.tp_price:
                exit_reason = "take_profit"
            elif current_price >= pos.sl_price:
                exit_reason = "trailing_stop" if pos.trailing_stop_active else "stop_loss"
        else:
            # Dynamic Take Profit (ATR-based)
            if pos.tp_price and current_price >= pos.tp_price:
                exit_reason = "take_profit"
            # Trailing / Stop Loss
            elif current_price <= pos.sl_price:
                exit_reason = "trailing_stop" if pos.trailing_stop_active else "stop_loss"

        # [PRO] Time-based exit: Move SL to entry after 4 hours if in profit
        now = datetime.now(tz=timezone.utc)
        duration_h = (now - pos.entry_time).total_seconds() / 3600.0
        in_profit = (current_price < pos.entry_price) if is_short else (current_price > pos.entry_price)
        if not pos.trailing_stop_active and duration_h >= _BREAK_EVEN_HOURS and in_profit:
            if is_short:
                if pos.sl_price > pos.entry_price:
                    pos.sl_price = pos.entry_price
                    pos.current_stop_loss = pos.entry_price
                    logger.info("[TIME SHORT] %s duration=%.1fh: SL moved to entry.", symbol, duration_h)
            else:
                if pos.sl_price < pos.entry_price:
                    pos.sl_price = pos.entry_price
                    pos.current_stop_loss = pos.entry_price
                    logger.info("[TIME] %s duration=%.1fh: SL moved to entry (Break-even).", symbol, duration_h)

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

        # Per-symbol TTL derived from horizon × timeframe so ETH/XRP expire at ~9h, not 12h
        cfg = get_symbol_config(symbol)
        tf_str: str = str(cfg.get("timeframe", "15m"))
        if tf_str.endswith("h"):
            tf_min = int(tf_str[:-1]) * 60
        else:
            tf_min = int(tf_str[:-1]) if tf_str[:-1].isdigit() else 15
        horizon_bars = int(cfg.get("horizon", 24))
        symbol_ttl_h = max(POSITION_TTL_HOURS, horizon_bars * tf_min / 60.0)

        if duration_h >= symbol_ttl_h:
            logger.info(
                "⏱️ [TTL] %s position reached max age (%.1fh, symbol-horizon=%.1fh).",
                symbol,
                symbol_ttl_h,
                horizon_bars * tf_min / 60.0,
            )
            return await self._close_position(symbol, current_price, "ttl_expiry")
        return None

    async def _close_position(self, symbol: str, exit_price: float, reason: str) -> float:
        pos = self.open_positions.pop(symbol)
        exit_time = datetime.now(tz=timezone.utc)
        
        # PnL Calculation (direction-aware)
        if pos.entry_price <= 0.0:
            logger.error("_close_position: entry_price=%s invalid for %s — PnL forced to 0.", pos.entry_price, symbol)
            pnl_pct = 0.0
        elif getattr(pos, "direction", "long") == "short":
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price  # SHORT: profit when price drops
        else:
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
        hours_held = (exit_time - pos.entry_time).total_seconds() / 3600.0

        # Observability: trade closed
        get_buffer().emit_exit(
            ExitMetrics(
                symbol=symbol,
                timestamp=time.time(),
                exit_price=round(exit_price, 4),
                pnl=round(gross_pnl, 4),
                reason=reason,
                hours_held=round(hours_held, 2),
            )
        )

        logger.info("🏁 [CLOSE] %s exit=%.4f pnl=%.2f (%.2f%%) reason=%s", symbol, exit_price, gross_pnl, pnl_pct_display, reason)
        await send_telegram_alert(f"🏁 *CLOSE* {symbol}\nExit: {exit_price:.4f}\nPnL: {gross_pnl:+.2f} ({pnl_pct_display:+.2f}%)\nReason: {reason}")
        return gross_pnl

    async def sync_positions_with_exchange(self, confirmations_required: int = 1) -> int:
        return len(self.open_positions)

    async def retry_close_pending_positions(self, latest_prices: dict[str, float]) -> int:
        """Retry closing positions marked as close_pending. Overridden in MT5Executor."""
        closed_count = 0
        for sym, price in latest_prices.items():
            pos = self.open_positions.get(sym)
            if pos and pos.close_pending:
                logger.info("[RECONCILER] Retrying paper close for %s", sym)
                reason = pos.last_close_error or "retry_close"
                await self._close_position(sym, price, reason)
                closed_count += 1
        return closed_count

def _build_trade_report(*args: Any, **kwargs: Any) -> str:
    """Legacy stub for MT5Executor."""
    return ""

def record_trade(*args: Any, **kwargs: Any) -> None:
    """Legacy stub for MT5Executor."""
    pass
