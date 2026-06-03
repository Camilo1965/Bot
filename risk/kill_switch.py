"""
risk.kill_switch
~~~~~~~~~~~~~~~~

Automated risk-circuit-breaker with 4 auto-triggers + persistent state.

Triggers
--------
T1  daily_pnl_pct ≤ -5%               → freeze entries 24h
T2  ≥3 consecutive losses (cross-sym)  → pause entries 4h
T3  rolling 7d drawdown ≥ 15%         → close_all flag + pause 48h
T4  per-symbol rolling WR < 30%        → demote symbol (threshold → 0.95 in-memory)

Legacy backward-compat (signal_emitter imports these module-level functions):
    from risk.kill_switch import is_halted, reason, halt, resume
All delegate to the thread-safe KillSwitch singleton.

State: data/kill_switch_state.json
Events: logs/kill_switch_events.log (JSONL)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_KILL_FILE = _REPO / "KILL"
_DEFAULT_STATE = _REPO / "data" / "kill_switch_state.json"
_DEFAULT_LOG = _REPO / "logs" / "kill_switch_events.log"

_instance: "KillSwitch | None" = None
_instance_lock = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso() -> str:
    return _utcnow().isoformat()


class KillSwitch:
    """Thread-safe kill-switch with auto-trigger evaluation and persistent state."""

    def __init__(
        self,
        state_path: Path | str = _DEFAULT_STATE,
        log_path: Path | str = _DEFAULT_LOG,
    ) -> None:
        self._state_path = Path(state_path)
        self._log_path = Path(log_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._mu = threading.Lock()
        self._state: dict[str, Any] = self._load_state()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.is_file():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "active": False,
            "since": None,
            "until": None,
            "reason": None,
            "close_all": False,
            "demoted_symbols": {},
            "history": [],
        }

    def _save(self) -> None:
        try:
            self._state_path.write_text(
                json.dumps(self._state, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("KillSwitch save failed: %s", exc)

    def _log(self, event: dict[str, Any]) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**event, "ts": _iso()}) + "\n")
        except Exception as exc:
            logger.error("KillSwitch log write failed: %s", exc)

    # ── Core state ───────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """Return True if any kill-switch mechanism is engaged."""
        if os.environ.get("KILL_SWITCH", "").strip() in ("1", "true", "yes"):
            return True
        if _KILL_FILE.is_file():
            return True
        with self._mu:
            return self._check_active_locked()

    def _check_active_locked(self) -> bool:
        if not self._state.get("active"):
            return False
        until_str = self._state.get("until")
        if until_str:
            try:
                until_dt = datetime.fromisoformat(until_str)
                if _utcnow() >= until_dt:
                    self._deactivate_locked()
                    return False
            except Exception:
                pass
        return True

    def get_reason(self) -> str | None:
        if os.environ.get("KILL_SWITCH", "").strip() in ("1", "true", "yes"):
            return os.environ.get("KILL_SWITCH_REASON") or "KILL_SWITCH env"
        if _KILL_FILE.is_file():
            try:
                return _KILL_FILE.read_text(encoding="utf-8").strip() or "KILL file"
            except Exception:
                return "KILL file"
        with self._mu:
            return self._state.get("reason")

    def activate(self, reason_str: str, duration_hours: float, close_all: bool = False) -> None:
        until = (_utcnow() + timedelta(hours=duration_hours)).isoformat()
        event = {
            "event": "activated",
            "reason": reason_str,
            "duration_hours": duration_hours,
            "close_all": close_all,
        }
        with self._mu:
            self._state.update({
                "active": True,
                "since": _iso(),
                "until": until,
                "reason": reason_str,
                "close_all": close_all,
            })
            self._state.setdefault("history", []).append({**event, "ts": _iso()})
            self._save()
        self._log(event)
        logger.warning(
            "[KILL_SWITCH] ACTIVATED — %s | %.1fh | close_all=%s",
            reason_str, duration_hours, close_all,
        )

    def deactivate(self) -> None:
        with self._mu:
            self._deactivate_locked()

    def _deactivate_locked(self) -> None:
        was_active = self._state.get("active", False)
        self._state.update({"active": False, "since": None, "until": None, "reason": None, "close_all": False})
        if was_active:
            self._state.setdefault("history", []).append({"event": "deactivated", "ts": _iso()})
            self._save()
        self._log({"event": "deactivated"})
        if was_active:
            logger.info("[KILL_SWITCH] Deactivated.")

    def needs_close_all(self) -> bool:
        with self._mu:
            return bool(self._state.get("active") and self._state.get("close_all"))

    def clear_close_all(self) -> None:
        with self._mu:
            self._state["close_all"] = False
            self._save()

    def status(self) -> dict[str, Any]:
        with self._mu:
            return dict(self._state)

    # ── Symbol demoting ──────────────────────────────────────────────────────

    def demote_symbol(self, symbol: str, threshold: float = 0.95) -> None:
        with self._mu:
            self._state.setdefault("demoted_symbols", {})[symbol] = threshold
            self._save()
        self._log({"event": "symbol_demoted", "symbol": symbol, "threshold": threshold})
        logger.warning("[KILL_SWITCH] Symbol demoted: %s → pt=%.2f", symbol, threshold)

    def undmote_symbol(self, symbol: str) -> None:
        with self._mu:
            self._state.get("demoted_symbols", {}).pop(symbol, None)
            self._save()
        logger.info("[KILL_SWITCH] Symbol un-demoted: %s", symbol)

    def get_effective_threshold(self, symbol: str, default: float) -> float:
        """Return demoted threshold if active, else the config default."""
        with self._mu:
            return self._state.get("demoted_symbols", {}).get(symbol, default)

    def is_symbol_demoted(self, symbol: str) -> bool:
        with self._mu:
            return symbol in self._state.get("demoted_symbols", {})

    # ── Trigger evaluation ───────────────────────────────────────────────────

    def check_triggers(
        self,
        *,
        daily_pnl_pct: float = 0.0,
        drawdown_7d_pct: float = 0.0,
        consecutive_losses: int = 0,
        symbol_rolling_stats: dict[str, dict] | None = None,
    ) -> list[str]:
        """
        Evaluate all 4 triggers. Returns list of fired trigger IDs.
        Mutates kill-switch state when triggers fire.

        Args:
            daily_pnl_pct      – today's PnL as % of balance (negative = loss)
            drawdown_7d_pct    – rolling 7d drawdown as % (positive number = drawdown)
            consecutive_losses – consecutive losing trades across all symbols
            symbol_rolling_stats – {sym: {"win_rate_rolling": 0.0..1.0, "trades": int}}
        """
        fired: list[str] = []
        currently_active = self.is_active()

        # T1: daily loss ≥ 5%
        if daily_pnl_pct <= -5.0 and not currently_active:
            self.activate(f"T1:daily_pnl={daily_pnl_pct:.1f}%", duration_hours=24.0)
            fired.append("T1_daily_loss")
            currently_active = True

        # T2: 3+ consecutive losses
        if consecutive_losses >= 3 and not currently_active:
            self.activate(f"T2:consec_losses={consecutive_losses}", duration_hours=4.0)
            fired.append("T2_consec_losses")
            currently_active = True

        # T3: 7d drawdown ≥ 15% → close_all + 48h
        if drawdown_7d_pct >= 15.0 and not currently_active:
            self.activate(
                f"T3:dd_7d={drawdown_7d_pct:.1f}%",
                duration_hours=48.0,
                close_all=True,
            )
            fired.append("T3_drawdown_7d")

        # T4: per-symbol rolling WR < 30% (requires ≥20 trades)
        if symbol_rolling_stats:
            for sym, stats in symbol_rolling_stats.items():
                n = stats.get("trades", 0)
                wr = stats.get("win_rate_rolling", 1.0)
                if n >= 20 and wr < 0.30 and not self.is_symbol_demoted(sym):
                    self.demote_symbol(sym, threshold=0.95)
                    fired.append(f"T4_wr_demote:{sym}")

        return fired


# ── Singleton ─────────────────────────────────────────────────────────────────

def get_kill_switch() -> KillSwitch:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = KillSwitch()
    return _instance


# ── Backward-compat module-level API (imported by signal_emitter.py) ──────────

def is_halted() -> bool:
    return get_kill_switch().is_active()


def reason() -> str | None:
    return get_kill_switch().get_reason()


def halt(reason_str: str = "manual", duration_hours: float = 24.0) -> None:
    get_kill_switch().activate(reason_str, duration_hours)


def resume() -> None:
    get_kill_switch().deactivate()
