"""Append-only trade lifecycle audit log (JSON lines).

Authoritative execution state remains MT5 + ``PaperExecutor.open_positions`` +
TimescaleDB. This module is optional observability for post-mortems."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TradeState(str, Enum):
    SIGNAL_RECEIVED = "signal_received"
    ORDER_SENT = "order_sent"
    PENDING = "pending"
    EXECUTED = "executed"
    MANAGING = "managing"
    CLOSE_REQUESTED = "close_requested"
    CLOSED = "closed"
    ERROR = "error"


class OperationsTracker:
    """Writes one JSON object per line to *log_path* (safe for tail/grep)."""

    def __init__(self, log_path: str | Path = "logs/trade_events.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        trade_id: str,
        symbol: str,
        state: TradeState,
        price: float,
        **details: Any,
    ) -> None:
        payload = {
            "trade_id": trade_id,
            "symbol": symbol,
            "state": state.value,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "price": price,
            "details": details,
        }
        line = json.dumps(payload, default=str)
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning("operations_tracker: could not append: %s", e)
