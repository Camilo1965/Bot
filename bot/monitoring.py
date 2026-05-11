"""Production monitoring alerts for ClawdBot.

Centralises alert rules so the health_monitor_loop stays readable.
All alerts are deduplicated by key and respect cooldowns.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from utils.telegram_notifier import send_priority_telegram_alert

logger = logging.getLogger("clawdbot.monitoring")

# ── Config from ENV ──────────────────────────────────────────────────────────
_STALE_FEED_ALERT_S: float = float(os.environ.get("MONITOR_STALE_FEED_S", "90").strip() or "90")
_DAILY_PNL_ALERT_PCT: float = float(os.environ.get("MONITOR_DAILY_PNL_ALERT_PCT", "0.015").strip() or "0.015")
_VIBE_DOWN_ALERT_S: float = float(os.environ.get("MONITOR_VIBE_DOWN_S", "300").strip() or "300")
_ALERT_COOLDOWN_S: float = float(os.environ.get("MONITOR_ALERT_COOLDOWN_S", "300").strip() or "300")

# ── Deduplication state ──────────────────────────────────────────────────────
_last_alert_at: dict[str, float] = {}


def _should_alert(key: str, cooldown: float | None = None) -> bool:
    """Return True if enough time passed since last alert for *key*."""
    now = time.monotonic()
    cool = cooldown if cooldown is not None else _ALERT_COOLDOWN_S
    if now - _last_alert_at.get(key, 0) >= cool:
        _last_alert_at[key] = now
        return True
    return False


async def check_stale_feed_alert(
    state: dict[str, Any],
    *,
    threshold_s: float | None = None,
) -> None:
    """Alert if no market message has been received recently."""
    thr = threshold_s if threshold_s is not None else _STALE_FEED_ALERT_S
    last_msg: datetime | None = state.get("last_market_message_at")
    if last_msg is None:
        return
    stale = (datetime.now(tz=timezone.utc) - last_msg).total_seconds()
    if stale > thr and _should_alert("stale_feed", cooldown=60):
        logger.warning("[MONITOR] Stale feed detected: %.1fs > %.1fs", stale, thr)
        asyncio.create_task(
            send_priority_telegram_alert(
                f"⚠️ *FEED STALE*\n"
                f"Último tick hace `{stale:.0f}s` (umbral `{thr:.0f}s`).\n"
                f"Verificar conexión MT5.",
                priority="critical",
                dedup_key="monitor:stale_feed",
                force=True,
            )
        )


async def check_daily_pnl_alert(
    risk_manager: Any,
    paper_executor: Any,
    *,
    threshold_pct: float | None = None,
) -> None:
    """Alert if daily realised loss crosses a fraction of the daily-loss limit."""
    thr = threshold_pct if threshold_pct is not None else _DAILY_PNL_ALERT_PCT
    # Use the RiskManager's internal daily tracking
    daily_loss = getattr(risk_manager, "_daily_loss", 0.0)
    daily_start = getattr(risk_manager, "_daily_start_balance", 1.0)
    if daily_start <= 0:
        return
    loss_pct = daily_loss / daily_start
    if loss_pct >= thr and _should_alert("daily_pnl", cooldown=600):
        logger.warning(
            "[MONITOR] Daily loss alert: %.2f%% of balance (threshold %.2f%%)",
            loss_pct * 100,
            thr * 100,
        )
        total_pnl = getattr(paper_executor, "total_pnl", 0.0)
        asyncio.create_task(
            send_priority_telegram_alert(
                f"🚨 *PÉRDIDA DIARIA*\n"
                f"Pérdida acumulada: `{loss_pct:.2%}` del balance inicial.\n"
                f"Total PnL sesión: `{total_pnl:+.2f}` USDT\n"
                f"Revisar estrategia o pausar bot.",
                priority="critical",
                dedup_key="monitor:daily_pnl",
                force=True,
            )
        )


async def check_vibe_down_alert(state: dict[str, Any]) -> None:
    """Alert if Vibe-Trading MCP client has been unavailable for too long."""
    vibe_client = state.get("vibe_client")
    if vibe_client is None:
        return
    available = getattr(vibe_client, "available", False)
    last_err = getattr(vibe_client, "last_error", None)
    # If client exists but is not available
    if not available and _should_alert("vibe_down", cooldown=_VIBE_DOWN_ALERT_S):
        logger.warning("[MONITOR] VIBE client unavailable: %s", last_err or "unknown")
        asyncio.create_task(
            send_priority_telegram_alert(
                f"🔌 *VIBE-TRADING DOWN*\n"
                f"MCP client no disponible hace >{_VIBE_DOWN_ALERT_S/60:.0f} min.\n"
                f"Error: `{str(last_err)[:200] or 'unknown'}`\n"
                f"El bot sigue operando con ML nativo.",
                priority="warning",
                dedup_key="monitor:vibe_down",
                force=True,
            )
        )


async def check_open_positions_alert(paper_executor: Any) -> None:
    """Alert if positions are stuck in close_pending for too long."""
    pending = [
        (sym, getattr(pos, "last_close_error", "unknown"))
        for sym, pos in paper_executor.open_positions.items()
        if getattr(pos, "close_pending", False)
    ]
    if pending and _should_alert("close_pending", cooldown=120):
        details = "\n".join(f"• {s}: {e}" for s, e in pending)
        asyncio.create_task(
            send_priority_telegram_alert(
                f"⏳ *CIERRE PENDIENTE*\n"
                f"Posiciones atascadas en close_pending:\n{details}",
                priority="warning",
                dedup_key="monitor:close_pending",
                force=True,
            )
        )


def get_health_metrics(
    state: dict[str, Any],
    market_queue: Any,
    paper_executor: Any,
    risk_manager: Any,
) -> dict[str, Any]:
    """Return a serialisable health snapshot for /api/health."""
    last_msg: datetime | None = state.get("last_market_message_at")
    now = datetime.now(tz=timezone.utc)
    stale_s: float | None = None
    if last_msg is not None:
        stale_s = round((now - last_msg).total_seconds(), 1)

    vibe_client = state.get("vibe_client")
    vibe_ok = bool(vibe_client.available) if vibe_client else False

    daily_loss = getattr(risk_manager, "_daily_loss", 0.0)
    daily_start = getattr(risk_manager, "_daily_start_balance", 1.0)
    loss_pct = round(daily_loss / daily_start, 4) if daily_start > 0 else 0.0

    return {
        "status": "ok" if stale_s is None or stale_s < _STALE_FEED_ALERT_S else "degraded",
        "timestamp": now.isoformat(),
        "stale_feed_seconds": stale_s,
        "queue_size": getattr(market_queue, "qsize", lambda: -1)(),
        "open_positions": len(paper_executor.open_positions),
        "max_positions": getattr(risk_manager, "max_positions", 0),
        "total_pnl_session": round(getattr(paper_executor, "total_pnl", 0.0), 2),
        "daily_loss_pct": loss_pct,
        "vibe_available": vibe_ok,
        "balance": round(getattr(risk_manager, "balance", 0.0), 2),
    }
