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
main trading loop (fire-and-forget, no re-raise).
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

# Telegram Bot API base URL (token is interpolated at call time)
_TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# Network timeout for the POST request (seconds).
_REQUEST_TIMEOUT: float = 5.0


async def send_telegram_alert(message: str) -> None:
    """Send *message* to the configured Telegram chat asynchronously.

    The function is designed for fire-and-forget usage via
    ``asyncio.create_task(send_telegram_alert(...))``.  All exceptions are
    caught and logged; the coroutine never raises to its caller.

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
        return

    url = _TELEGRAM_API_URL.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
    }

    try:
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if not resp.ok:
                    body = await resp.text()
                    logger.error(
                        "Telegram API returned HTTP %d: %s",
                        resp.status,
                        body[:200],
                    )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
