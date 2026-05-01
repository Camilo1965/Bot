"""
utils.telegram_notifier
~~~~~~~~~~~~~~~~~~~~~~~

Lightweight async Telegram notification helper.

Usage (fire-and-forget from an async context)::

    import asyncio
    from utils.telegram_notifier import send_telegram_alert

    asyncio.create_task(send_telegram_alert("🚀 *OPEN LONG* | #BTC/USDT"))

The function reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from the
environment at call time so it works with any environment-variable management
strategy (``python-dotenv``, OS env, Docker secrets, etc.).

If either variable is missing or the HTTP call fails the error is logged and
execution continues; the notification is **never** allowed to interrupt the
main trading loop (fire-and-forget, no re-raise).  Returns ``False`` when
skipped or on failure, ``True`` when Telegram returns HTTP 200.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp

from bot.dashboard_helpers import mt5_dashboard_mark
from database.db_manager import db

try:
    import certifi
except ImportError:
    certifi = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Telegram Bot API base URL (token is interpolated at call time)
_TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Network timeout for the POST request (seconds).
_REQUEST_TIMEOUT: float = 5.0
_REPORT_TIMEZONE_ENV = "REPORT_TIMEZONE"
_DEFAULT_REPORT_TIMEZONE = "America/Bogota"
_PRIORITY_COOLDOWN_S = {
    "critical": 60.0,
    "summary": 300.0,
    "info": 120.0,
}
_LAST_ALERT_SENT_AT: dict[str, float] = {}


def _telegram_ssl_context() -> ssl.SSLContext | bool | None:
    """SSL context that uses certifi on Windows when the OS store fails verification."""
    if certifi is not None:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            pass
    return None


async def send_telegram_alert(message: str) -> bool:
    """Send *message* to the configured Telegram chat asynchronously.

    The function is designed for fire-and-forget usage via
    ``asyncio.create_task(send_telegram_alert(...))``.  All exceptions are
    caught and logged; the coroutine never raises to its caller.

    Returns
    -------
    bool
        ``True`` if Telegram accepted the message (HTTP 200), ``False`` if
        skipped, failed, or misconfigured.

    Parameters
    ----------
    message:
        Telegram message text.  Markdown formatting (``parse_mode="Markdown"``)
        is enabled so callers may use ``*bold*``, ``_italic_``, etc.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        logger.debug(
            "Telegram alert skipped – TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID "
            "not configured."
        )
        return False

    url = _TELEGRAM_API_URL.format(token=token)
    base_payload: dict[str, str | int] = {
        "chat_id": chat_id,
        "text": message,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        ssl_ctx = _telegram_ssl_context()
        connector = (
            aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx is not None else aiohttp.TCPConnector()
        )
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            payload = {**base_payload, "parse_mode": "Markdown"}
            async with session.post(url, json=payload) as resp:
                if resp.ok:
                    logger.info("Telegram alert sent successfully.")
                    return True
                body = await resp.text()
                # Bad entities / broken Markdown — retry as plain text
                if resp.status == 400 and (
                    "parse" in body.lower() or "markdown" in body.lower()
                ):
                    logger.warning(
                        "Telegram rejected Markdown; retrying as plain text. Body: %s",
                        body[:200],
                    )
                    async with session.post(url, json=base_payload) as resp2:
                        if resp2.ok:
                            logger.info("Telegram alert sent successfully (plain text).")
                            return True
                        body2 = await resp2.text()
                        logger.error(
                            "Telegram API returned HTTP %d: %s",
                            resp2.status,
                            body2[:200],
                        )
                        return False
                logger.error(
                    "Telegram API returned HTTP %d: %s",
                    resp.status,
                    body[:200],
                )
                return False
    except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
        return False


async def send_priority_telegram_alert(
    message: str,
    *,
    priority: str = "info",
    dedup_key: str | None = None,
    force: bool = False,
) -> bool:
    """Send Telegram alert with priority-based cooldown and dedup."""
    key = dedup_key or f"{priority}:{hash(message)}"
    now_mono = time.monotonic()
    cooldown = _PRIORITY_COOLDOWN_S.get(priority, _PRIORITY_COOLDOWN_S["info"])
    if not force:
        last_sent = _LAST_ALERT_SENT_AT.get(key, 0.0)
        if now_mono - last_sent < cooldown:
            logger.debug("Telegram alert skipped by cooldown (priority=%s key=%s).", priority, key)
            return False
    ok = await send_telegram_alert(message)
    if ok:
        _LAST_ALERT_SENT_AT[key] = now_mono
    return ok


def install_asyncio_critical_telegram_alerts() -> None:
    """Hook asyncio loop errors → Telegram (priority critical, dedup+cooldown).

    Call once from inside ``async def main()`` so ``get_running_loop()`` exists.
    Complements ``gather`` exception handling: catches task/callback failures that
    do not propagate to the main ``gather``.
    """
    loop = asyncio.get_running_loop()

    def _handler(async_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        async_loop.default_exception_handler(context)
        exc = context.get("exception")
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            return
        if isinstance(exc, asyncio.CancelledError):
            return
        msg = str(context.get("message", "") or "")
        task = context.get("task")
        task_label = ""
        if task is not None:
            try:
                task_label = task.get_name()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                task_label = repr(task)
        exc_line = (
            f"`{type(exc).__name__}`: `{str(exc)[:280]}`"
            if exc is not None
            else (msg[:380] if msg else "error sin excepción en contexto")
        )
        lines = [
            "🚨 *ERROR ASYNC (loop)* — revisar logs / posible fix requerido",
            "",
            exc_line,
        ]
        if task_label:
            lines.append(f"Task: `{task_label[:120]}`")
        text = "\n".join(lines)
        dedup_key = f"asyncio_ctx:{type(exc).__name__ if exc else 'nomsg'}:{hash(msg[:100]) % 100001}"
        try:
            asyncio.create_task(
                send_priority_telegram_alert(
                    text,
                    priority="critical",
                    dedup_key=dedup_key,
                )
            )
        except RuntimeError:
            pass

    loop.set_exception_handler(_handler)


def _report_timezone() -> ZoneInfo:
    tz_name = os.environ.get(_REPORT_TIMEZONE_ENV, _DEFAULT_REPORT_TIMEZONE).strip() or _DEFAULT_REPORT_TIMEZONE
    try:
        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid REPORT_TIMEZONE='%s', using %s.", tz_name, _DEFAULT_REPORT_TIMEZONE)
        return ZoneInfo(_DEFAULT_REPORT_TIMEZONE)


def _fmt_money(value: float) -> str:
    return f"{value:+,.2f} USDT"


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _fmt_profit_factor(summary: dict[str, Any]) -> str:
    losses = int(summary.get("losses", 0) or 0)
    wins = int(summary.get("wins", 0) or 0)
    total = int(summary.get("total_trades", 0) or 0)
    if total == 0:
        return "N/A (sin trades cerrados)"
    if losses == 0:
        if wins == 0:
            return "N/A"
        return "N/A (sin pérdidas cerradas)"
    return f"{float(summary.get('profit_factor', 0.0)):.2f}"


async def _build_period_report(period: str, tz_name: str) -> str:
    summary = await db.fetch_period_summary(period, tz_name=tz_name)
    icon = "🟢" if summary["pnl_total"] >= 0 else "🔴"
    best_trade = summary.get("best_trade")
    worst_trade = summary.get("worst_trade")
    best_line = "N/A"
    worst_line = "N/A"
    if isinstance(best_trade, dict):
        best_line = f"{best_trade.get('symbol', '-')} {_fmt_money(float(best_trade.get('pnl', 0.0)))}"
    if isinstance(worst_trade, dict):
        worst_line = f"{worst_trade.get('symbol', '-')} {_fmt_money(float(worst_trade.get('pnl', 0.0)))}"
    label = "SEMANA" if period == "week" else "MES" if period == "month" else "DÍA"
    pf_label = _fmt_profit_factor(summary)
    sample_note = (
        "Muestra baja: interpreta con cautela."
        if int(summary["total_trades"]) < 5
        else "Muestra estable."
    )
    return (
        f"📈 *RESUMEN {label}* ({tz_name})\n"
        f"────────────────────────\n"
        f"💰 *PnL:* {_fmt_money(summary['pnl_total'])} {icon}\n"
        f"📊 *Trades:* {summary['total_trades']} | Ganados: {summary['wins']} | Perdidos: {summary['losses']}\n"
        f"🎯 *Winrate:* {_fmt_pct(summary['winrate'])}\n"
        f"⚖️ *Profit Factor:* {pf_label}\n"
        f"🏆 *Mejor Trade:* {best_line}\n"
        f"📉 *Peor Trade:* {worst_line}\n"
        f"🧪 *Calidad muestra:* {sample_note}\n"
        f"────────────────────────"
    )


async def _build_history_report(days: int, tz_name: str) -> str:
    series = await db.fetch_daily_pnl_series(days=days, tz_name=tz_name)
    if not series:
        return f"📚 *HISTORIAL {days}D* ({tz_name})\nSin datos de trades cerrados."
    tz = _report_timezone()
    lines = [f"📚 *HISTORIAL {days}D* ({tz_name})", "────────────────────────"]
    for item in series[-days:]:
        day_utc = datetime.fromisoformat(str(item["day_utc"]))
        day_local = day_utc.astimezone(tz).strftime("%Y-%m-%d")
        pnl_total = float(item["pnl_total"])
        icon = "🟢" if pnl_total >= 0 else "🔴"
        lines.append(f"{day_local}: {_fmt_money(pnl_total)} {icon}")
    lines.append("────────────────────────")
    return "\n".join(lines)


async def telegram_command_poller(
    state: dict[str, Any],
    paper_executor: Any,
    risk_manager: Any,
    interval: int = 5,
) -> None:
    """Poll Telegram for incoming commands like /status."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    last_update_id = 0
    
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT + interval)
    ssl_ctx = _telegram_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_ctx) if ssl_ctx is not None else aiohttp.TCPConnector()
    
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            while True:
                await asyncio.sleep(interval)
                payload = {
                    "offset": last_update_id + 1,
                    "timeout": interval,
                    "allowed_updates": ["message"]
                }
                try:
                    async with session.post(url, json=payload) as resp:
                        if not resp.ok:
                            continue
                        data = await resp.json()
                        if not data.get("ok"):
                            continue
                        
                        updates = data.get("result", [])
                        for update in updates:
                            update_id = update["update_id"]
                            last_update_id = max(last_update_id, update_id)
                            
                            message = update.get("message")
                            if not message:
                                continue
                            
                            msg_chat_id = str(message.get("chat", {}).get("id", ""))
                            if msg_chat_id != chat_id:
                                continue  # Ignore messages from unauthorized chats
                                
                            text = message.get("text", "").strip()
                            if text.startswith("/status"):
                                num_open = len(paper_executor.open_positions)
                                max_pos = risk_manager.max_positions
                                balance = risk_manager.balance
                                realized_pnl = paper_executor.total_pnl
                                floating_total = 0.0
                                
                                report = (
                                    "📊 *ESTADO ACTUAL DEL BOT*\n"
                                    f"💰 *Balance Real:* {balance:.2f} USDT\n"
                                    f"📈 *PnL Realizado:* {realized_pnl:+.4f} USDT\n"
                                    f"🛒 *Posiciones Abiertas:* {num_open}/{max_pos}\n\n"
                                )
                                
                                if num_open > 0:
                                    for sym, pos in paper_executor.open_positions.items():
                                        # Intentar obtener el precio actual del estado compartido
                                        prices_buf = state.get("prices", {}).get(sym, [])
                                        candle_m = float(prices_buf[-1]) if prices_buf else pos.entry_price
                                        current_price = mt5_dashboard_mark(state, sym, candle_m) or pos.entry_price
                                        
                                        # Calcular PnL irrealizado
                                        qty = pos.position_size / pos.entry_price
                                        unrealized_pnl = (current_price - pos.entry_price) * qty
                                        floating_total += unrealized_pnl
                                        pnl_sign = "+" if unrealized_pnl >= 0 else ""
                                        
                                        report += (
                                            f"🔹 *{sym}*\n"
                                            f"   In: {pos.entry_price:.2f} | Out/Mark: {current_price:.2f}\n"
                                            f"   SL: {pos.current_stop_loss:.2f}\n"
                                            f"   PnL: {pnl_sign}{unrealized_pnl:.4f} USDT\n\n"
                                        )
                                else:
                                    report += "💤 _No hay posiciones abiertas._"
                                total_estimated = realized_pnl + floating_total
                                report += (
                                    "\n"
                                    f"🫧 *PnL Flotante:* {floating_total:+.4f} USDT\n"
                                    f"🧮 *PnL Total Estimado:* {total_estimated:+.4f} USDT"
                                )
                                
                                await send_priority_telegram_alert(
                                    report,
                                    priority="info",
                                    dedup_key="manual:status",
                                    force=True,
                                )
                            elif text.startswith("/stats") or text.startswith("/daily"):
                                stats = await db.fetch_daily_stats()
                                max_dd = state.get("max_drawdown", 0.0)
                                dd_sign = "-" if max_dd < 0 else ""
                                
                                wins = stats["wins"]
                                losses = stats["losses"]
                                daily_pnl = stats["daily_pnl"]
                                total_trades = stats["total_trades"]
                                floating_total = 0.0
                                for sym, pos in paper_executor.open_positions.items():
                                    prices_buf = state.get("prices", {}).get(sym, [])
                                    candle_m = float(prices_buf[-1]) if prices_buf else pos.entry_price
                                    current_price = mt5_dashboard_mark(state, sym, candle_m) or pos.entry_price
                                    qty = pos.position_size / pos.entry_price
                                    floating_total += (current_price - pos.entry_price) * qty
                                
                                pnl_sign = "+" if daily_pnl >= 0 else ""
                                pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
                                
                                report = (
                                    "📅 *RESUMEN DEL DÍA (HOY)*\n"
                                    "────────────────────────\n"
                                    f"📊 *Trades Cerrados:* {total_trades}\n"
                                    f"🏆 *Ganados / Perdidos:* {wins} / {losses}\n"
                                    f"💰 *PnL del Día:* {pnl_sign}{daily_pnl:.4f} USDT {pnl_emoji}\n"
                                    f"🫧 *PnL Flotante Actual:* {floating_total:+.4f} USDT\n"
                                    f"📉 *Máximo Drawdown:* {dd_sign}{abs(max_dd):.4f} USDT\n"
                                    "────────────────────────"
                                )
                                await send_priority_telegram_alert(
                                    report,
                                    priority="summary",
                                    dedup_key="manual:daily",
                                    force=True,
                                )
                            elif text.startswith("/weekly"):
                                tz_name = str(_report_timezone())
                                report = await _build_period_report("week", tz_name)
                                await send_priority_telegram_alert(
                                    report,
                                    priority="summary",
                                    dedup_key="manual:weekly",
                                    force=True,
                                )
                            elif text.startswith("/monthly"):
                                tz_name = str(_report_timezone())
                                report = await _build_period_report("month", tz_name)
                                await send_priority_telegram_alert(
                                    report,
                                    priority="summary",
                                    dedup_key="manual:monthly",
                                    force=True,
                                )
                            elif text.startswith("/history"):
                                tz_name = str(_report_timezone())
                                days = 7
                                try:
                                    parts = text.split()
                                    if len(parts) > 1:
                                        days = int(parts[1])
                                except Exception:
                                    days = 7
                                if days not in (7, 30):
                                    await send_priority_telegram_alert(
                                        "Uso: `/history 7` o `/history 30`",
                                        priority="info",
                                        dedup_key="manual:history:usage",
                                        force=True,
                                    )
                                else:
                                    report = await _build_history_report(days, tz_name)
                                    await send_priority_telegram_alert(
                                        report,
                                        priority="summary",
                                        dedup_key=f"manual:history:{days}",
                                        force=True,
                                    )
                            elif text.startswith("/ceo"):
                                tz_name = str(_report_timezone())
                                week = await db.fetch_period_summary("week", tz_name=tz_name)
                                month = await db.fetch_period_summary("month", tz_name=tz_name)
                                week_pf = _fmt_profit_factor(week)
                                month_pf = _fmt_profit_factor(month)
                                ceo_report = (
                                    f"🧭 *CEO SNAPSHOT* ({tz_name})\n"
                                    "────────────────────────\n"
                                    f"📅 Semana: {_fmt_money(week['pnl_total'])} | Winrate {_fmt_pct(week['winrate'])} | Trades {week['total_trades']}\n"
                                    f"🗓️ Mes: {_fmt_money(month['pnl_total'])} | Winrate {_fmt_pct(month['winrate'])} | Trades {month['total_trades']}\n"
                                    f"⚖️ PF Semana/Mes: {week_pf} / {month_pf}\n"
                                    "────────────────────────"
                                )
                                await send_priority_telegram_alert(
                                    ceo_report,
                                    priority="summary",
                                    dedup_key="manual:ceo",
                                    force=True,
                                )
                            elif text.startswith("/help") or text.startswith("/ayuda"):
                                help_report = (
                                    "🤖 *COMANDOS DISPONIBLES*\n"
                                    "────────────────────────\n"
                                    "/status -> Estado actual del bot y posiciones abiertas\n"
                                    "/daily o /stats -> Resumen de hoy\n"
                                    "/weekly -> Resumen semanal (America/Bogota)\n"
                                    "/monthly -> Resumen mensual (America/Bogota)\n"
                                    "/history 7 -> Historial diario últimos 7 días\n"
                                    "/history 30 -> Historial diario últimos 30 días\n"
                                    "/ceo -> Snapshot ejecutivo (7D/30D)\n"
                                    "────────────────────────"
                                )
                                await send_priority_telegram_alert(
                                    help_report,
                                    priority="info",
                                    dedup_key="manual:help",
                                    force=True,
                                )
                except asyncio.TimeoutError:
                    pass  # Normal long-polling timeout
                except Exception as exc:
                    logger.debug("Error polling Telegram commands: %s", exc)
    except Exception as exc:
        logger.debug("Fatal error in telegram_command_poller: %s", exc)

