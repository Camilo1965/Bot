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
from typing import Any

import aiohttp

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
                                pnl = paper_executor.total_pnl
                                
                                report = (
                                    "📊 *ESTADO ACTUAL DEL BOT*\n"
                                    f"💰 *Balance Real:* {balance:.2f} USDT\n"
                                    f"📈 *PnL Histórico:* {pnl:+.4f} USDT\n"
                                    f"🛒 *Posiciones Abiertas:* {num_open}/{max_pos}\n\n"
                                )
                                
                                if num_open > 0:
                                    for sym, pos in paper_executor.open_positions.items():
                                        # Intentar obtener el precio actual del estado compartido
                                        prices_buf = state.get("prices", {}).get(sym, [])
                                        current_price = float(prices_buf[-1]) if prices_buf else pos.entry_price
                                        
                                        # Calcular PnL irrealizado
                                        qty = pos.position_size / pos.entry_price
                                        unrealized_pnl = (current_price - pos.entry_price) * qty
                                        pnl_sign = "+" if unrealized_pnl >= 0 else ""
                                        
                                        report += (
                                            f"🔹 *{sym}*\n"
                                            f"   In: {pos.entry_price:.2f} | Out/Mark: {current_price:.2f}\n"
                                            f"   SL: {pos.current_stop_loss:.2f}\n"
                                            f"   PnL: {pnl_sign}{unrealized_pnl:.4f} USDT\n\n"
                                        )
                                else:
                                    report += "💤 _No hay posiciones abiertas._"
                                
                                await send_telegram_alert(report)
                            elif text.startswith("/stats") or text.startswith("/daily"):
                                stats = await db.fetch_daily_stats()
                                max_dd = state.get("max_drawdown", 0.0)
                                dd_sign = "-" if max_dd < 0 else ""
                                
                                wins = stats["wins"]
                                losses = stats["losses"]
                                daily_pnl = stats["daily_pnl"]
                                total_trades = stats["total_trades"]
                                
                                pnl_sign = "+" if daily_pnl >= 0 else ""
                                pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
                                
                                report = (
                                    "📅 *RESUMEN DEL DÍA (HOY)*\n"
                                    "────────────────────────\n"
                                    f"📊 *Trades Cerrados:* {total_trades}\n"
                                    f"🏆 *Ganados / Perdidos:* {wins} / {losses}\n"
                                    f"💰 *PnL del Día:* {pnl_sign}{daily_pnl:.4f} USDT {pnl_emoji}\n"
                                    f"📉 *Máximo Drawdown:* {dd_sign}{abs(max_dd):.4f} USDT\n"
                                    "────────────────────────"
                                )
                                await send_telegram_alert(report)
                except asyncio.TimeoutError:
                    pass  # Normal long-polling timeout
                except Exception as exc:
                    logger.debug("Error polling Telegram commands: %s", exc)
    except Exception as exc:
        logger.debug("Fatal error in telegram_command_poller: %s", exc)

