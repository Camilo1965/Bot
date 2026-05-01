"""Persist startup + optional periodic snapshots of bot risk state for debugging."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execution.paper_executor import PaperExecutor, compute_dynamic_tp_hint
from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _REPO_ROOT / "logs"


def _positions_payload(paper_executor: PaperExecutor) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sym, pos in paper_executor.open_positions.items():
        tp_hint = compute_dynamic_tp_hint(pos)
        out.append({
            "symbol": sym,
            "entry_price": pos.entry_price,
            "peak_price": getattr(pos, "peak_price", pos.entry_price),
            "initial_stop_price": pos.stop_loss_price,
            "current_stop_loss": pos.current_stop_loss,
            "trailing_stop_active": bool(getattr(pos, "trailing_stop_active", False)),
            "activation_pct": pos.activation_pct,
            "trailing_distance_pct": pos.trailing_distance_pct,
            "sl_pct": pos.sl_pct,
            "dynamic_tp_hint": tp_hint,
            "mt5_ticket": getattr(pos, "mt5_position_ticket", None),
            "last_broker_sl_synced": getattr(pos, "last_broker_sl_synced", 0.0),
            "last_broker_tp_synced": getattr(pos, "last_broker_tp_synced", 0.0),
        })
    return out


def write_startup_snapshot(
    *,
    execution_mode: str,
    session_id: str,
    watchlist: list[str],
    risk_manager: RiskManager,
    paper_executor: PaperExecutor,
) -> Path:
    """Write ``logs/bot_startup_snapshot.json`` once positions/config are ready."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOGS_DIR / "bot_startup_snapshot.json"
    payload = {
        "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        "session_id": session_id,
        "execution_mode": execution_mode,
        "watchlist": list(watchlist),
        "balance": risk_manager.balance,
        "max_positions": risk_manager.max_positions,
        "open_positions": _positions_payload(paper_executor),
        "notes": (
            "SL sube con peak tras activation_pct (trailing). "
            "TP_hint = peak+gap si peak>entry (compute_dynamic_tp_hint). "
            "MT5 puede mostrar otros niveles hasta sync."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Runtime snapshot written: %s", path)
    return path


def append_runtime_metrics_jsonl(
    *,
    session_id: str,
    execution_mode: str,
    risk_manager: RiskManager,
    paper_executor: PaperExecutor,
    watchlist: list[str],
) -> None:
    """Append one JSON line to ``logs/runtime_metrics.jsonl``."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOGS_DIR / "runtime_metrics.jsonl"
    line = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "session_id": session_id,
        "execution_mode": execution_mode,
        "balance": risk_manager.balance,
        "open_count": len(paper_executor.open_positions),
        "total_pnl_session": paper_executor.total_pnl,
        "symbols_open": list(paper_executor.open_positions.keys()),
        "positions": _positions_payload(paper_executor),
        "watchlist_len": len(watchlist),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")


async def runtime_metrics_loop(
    session_id: str,
    execution_mode: str,
    risk_manager: RiskManager,
    paper_executor: PaperExecutor,
    watchlist: list[str],
    interval_s: float,
) -> None:
    """Background task: append metrics every *interval_s* seconds."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            append_runtime_metrics_jsonl(
                session_id=session_id,
                execution_mode=execution_mode,
                risk_manager=risk_manager,
                paper_executor=paper_executor,
                watchlist=watchlist,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_metrics_jsonl append failed: %s", exc)
