"""
vibe.notifier
~~~~~~~~~~~~~

High-level notification facade for ClawdBot events.

Wraps `utils.telegram_notifier` (which handles token, SSL, retry,
cooldown) and exposes structured helpers for each event type.

`notify_retrain` is called by `scripts/weekly_retrain.ps1` via:
    python -c "from vibe.notifier import notify_retrain; import asyncio; asyncio.run(notify_retrain(...))"
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from utils.telegram_notifier import send_telegram_alert, send_priority_telegram_alert

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("TELEGRAM_ENABLED", "1").strip() not in ("0", "false", "no")


class TelegramNotifier:
    """Simple async notifier that wraps the existing send_telegram_alert helper."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        if bot_token:
            os.environ["TELEGRAM_BOT_TOKEN"] = bot_token
        if chat_id:
            os.environ["TELEGRAM_CHAT_ID"] = chat_id

    async def notify(
        self,
        level: Literal["info", "warn", "critical"],
        msg: str,
        attachments: list | None = None,
    ) -> bool:
        if not _ENABLED:
            return False
        priority = {"info": "info", "warn": "warning_log", "critical": "critical"}.get(level, "info")
        return await send_priority_telegram_alert(msg, priority=priority)


# ── Structured event helpers ──────────────────────────────────────────────────

async def notify_trade_open(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float | None,
    size_usd: float,
    prob: float,
) -> None:
    if not _ENABLED:
        return
    tp_str = f"{tp:.4f}" if tp is not None else "NONE"
    msg = (
        f"*TRADE OPEN* {symbol} {side}\n"
        f"Entry: {entry:.4f} | SL: {sl:.4f} | TP: {tp_str}\n"
        f"Size: ${size_usd:.2f} | Prob: {prob:.1%}"
    )
    asyncio.create_task(send_telegram_alert(msg))


async def notify_trade_close(
    symbol: str,
    side: str,
    exit_price: float,
    pnl_usd: float,
    pnl_pct: float,
    reason: str,
    wr_rolling_7d: float | None = None,
) -> None:
    if not _ENABLED:
        return
    icon = "✅" if pnl_usd >= 0 else "❌"
    wr_line = f"\nWR 7d: {wr_rolling_7d:.1%}" if wr_rolling_7d is not None else ""
    msg = (
        f"{icon} *TRADE CLOSE* {symbol} {side}\n"
        f"Exit: {exit_price:.4f} | PnL: {pnl_usd:+.2f} ({pnl_pct:+.1%})\n"
        f"Reason: {reason}{wr_line}"
    )
    asyncio.create_task(send_telegram_alert(msg))


async def notify_kill_switch(trigger: str, action: str, duration_hours: float) -> None:
    if not _ENABLED:
        return
    msg = (
        f"*KILL SWITCH TRIGGERED*\n"
        f"Trigger: {trigger}\n"
        f"Action: {action} | Duration: {duration_hours:.0f}h"
    )
    asyncio.create_task(send_priority_telegram_alert(msg, priority="critical", force=True))


async def notify_drift(symbol: str, ks_stat: float, p_value: float) -> None:
    if not _ENABLED:
        return
    msg = (
        f"*DRIFT DETECTED* {symbol}\n"
        f"KS-stat: {ks_stat:.3f} | p-value: {p_value:.4f}\n"
        f"Symbol auto-paused 24h."
    )
    asyncio.create_task(send_priority_telegram_alert(msg, priority="critical", force=True))


async def notify_retrain(retrain_info: dict[str, Any]) -> None:
    """Called by weekly_retrain.ps1 after the retrain cycle completes.

    Args:
        retrain_info: dict with keys ts, retrain_exit, refit_exit, compare_exit
    """
    if not _ENABLED:
        return
    ts = retrain_info.get("ts", "?")
    r_exit = retrain_info.get("retrain_exit", -1)
    f_exit = retrain_info.get("refit_exit", -1)
    c_exit = retrain_info.get("compare_exit", -1)

    ok = r_exit == 0 and f_exit == 0
    rolled_back = c_exit == 2
    icon = "✅" if (ok and not rolled_back) else ("⚠️" if rolled_back else "❌")

    status = "OK — new models deployed" if (ok and not rolled_back) else (
        "ROLLED BACK — regression detected" if rolled_back else "FAILED"
    )
    msg = (
        f"{icon} *WEEKLY RETRAIN* {ts}\n"
        f"Status: {status}\n"
        f"retrain={r_exit} refit={f_exit} compare={c_exit}"
    )
    priority = "info" if (ok and not rolled_back) else "critical"
    await send_priority_telegram_alert(msg, priority=priority, force=True)
