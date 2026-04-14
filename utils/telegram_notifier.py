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

import aiohttp

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
