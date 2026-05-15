"""
bot.observability
~~~~~~~~~~~~~~~~~

Métricas estructuradas de latencia, throughput y calidad de predicción por símbolo.

Escribe JSON Lines a logs/observability.jsonl para análisis post-mortem y
proporciona un snapshot en memoria para el health endpoint.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("clawdbot.observability")

_REPO = Path(__file__).resolve().parent.parent
_METRICS_FILE = _REPO / "logs" / "observability.jsonl"
_METRICS_FILE.parent.mkdir(exist_ok=True)

# In-memory ring buffer (últimos N eventos por tipo)
_MAX_RING = 1000


@dataclass
class SignalMetrics:
    symbol: str
    timestamp: float
    latency_ms: float
    probability: float
    signal: str
    features_valid: bool
    model_used: str
    vibe_gated: bool = False
    vibe_gate_reason: str | None = None


@dataclass
class TradeMetrics:
    symbol: str
    timestamp: float
    side: str
    entry_price: float
    position_size: float
    win_probability: float
    atr: float | None
    vibe_quality: float
    sl_distance_pct: float


@dataclass
class ExitMetrics:
    symbol: str
    timestamp: float
    exit_price: float
    pnl: float
    reason: str
    hours_held: float


class ObservabilityBuffer:
    """Buffer en memoria + append a JSONL."""

    def __init__(self, max_ring: int = _MAX_RING) -> None:
        self._ring: list[dict[str, Any]] = []
        self._max = max_ring
        self._counters: dict[str, int] = defaultdict(int)
        self._symbol_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "signals": 0,
                "buys": 0,
                "trades_opened": 0,
                "trades_closed": 0,
                "total_pnl": 0.0,
                "last_prob": 0.0,
                "last_signal": "HOLD",
                "last_latency_ms": 0.0,
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        self._ring.append(record)
        if len(self._ring) > self._max:
            self._ring.pop(0)
        # Write to JSONL
        try:
            with _METRICS_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.debug("observability write failed: %s", exc)

    def emit_signal(self, m: SignalMetrics) -> None:
        self._counters["signals_total"] += 1
        ss = self._symbol_stats[m.symbol]
        ss["signals"] += 1
        ss["last_prob"] = m.probability
        ss["last_signal"] = m.signal
        ss["last_latency_ms"] = m.latency_ms
        if m.signal == "BUY":
            self._counters["signals_buy"] += 1
            ss["buys"] += 1
        if m.vibe_gated:
            self._counters["vibe_gates"] += 1
        self._append({"type": "signal", **asdict(m)})

    def emit_trade(self, m: TradeMetrics) -> None:
        self._counters["trades_opened"] += 1
        self._symbol_stats[m.symbol]["trades_opened"] += 1
        self._append({"type": "trade", **asdict(m)})

    def emit_exit(self, m: ExitMetrics) -> None:
        self._counters["trades_closed"] += 1
        ss = self._symbol_stats[m.symbol]
        ss["trades_closed"] += 1
        ss["total_pnl"] += m.pnl
        self._append({"type": "exit", **asdict(m)})

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(self._counters),
            "symbol_stats": dict(self._symbol_stats),
            "recent_events": self._ring[-50:],
        }


# Singleton global
_BUFFER = ObservabilityBuffer()


def get_buffer() -> ObservabilityBuffer:
    return _BUFFER
