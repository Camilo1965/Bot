"""
Run connectivity checks before ``main.py``:

* PostgreSQL (asyncpg) + schema / hypertables (TimescaleDB)
* MetaTrader 5 login (when ``EXECUTION_MODE=mt5``)
* Telegram Bot API (optional; needs token + chat id)

Usage::

    .\\.venv312\\Scripts\\python.exe preflight.py

Exit code ``0`` if all required checks pass, ``1`` otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv


async def _check_db() -> bool:
    from database.db_manager import DatabaseManager

    db = DatabaseManager()
    try:
        await db.connect()
        await db.close()
        return True
    except Exception as exc:  # noqa: BLE001
        logging.error("Database: %s", exc)
        return False


def _check_mt5() -> bool:
    mode = os.environ.get("EXECUTION_MODE", "mt5").strip().lower()
    if mode != "mt5":
        logging.info("MT5 check skipped (EXECUTION_MODE=%s).", mode)
        return True

    login_raw = os.environ.get("MT5_LOGIN", "").strip()
    password = os.environ.get("MT5_PASSWORD", "").strip()
    server = os.environ.get("MT5_SERVER", "").strip()
    if not login_raw or not password or not server:
        logging.error(
            "MT5: set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in .env "
            "(and ensure the MT5 terminal is running)."
        )
        return False
    try:
        login = int(login_raw)
    except ValueError:
        logging.error("MT5: MT5_LOGIN must be an integer account number.")
        return False

    from execution.mt5_executor import initialize_mt5, shutdown_mt5

    ok = initialize_mt5(account=login, password=password, server=server)
    if ok:
        shutdown_mt5()
    return ok


async def _check_telegram() -> bool:
    from utils.telegram_notifier import send_telegram_alert

    if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or not os.environ.get(
        "TELEGRAM_CHAT_ID", ""
    ).strip():
        logging.warning(
            "Telegram: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID empty — skipped."
        )
        return True

    return await send_telegram_alert(
        "*ClawdBot preflight* — si lees esto, Telegram responde OK."
    )


async def _async_main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)
    if not env_path.is_file():
        logging.warning(".env not found at %s — using process environment only.", env_path)

    ok_db = await _check_db()
    logging.info("Database: %s", "OK" if ok_db else "FAIL")

    ok_mt5 = _check_mt5()
    logging.info("MT5: %s", "OK" if ok_mt5 else "FAIL")

    ok_tg = await _check_telegram()
    logging.info("Telegram: %s", "OK" if ok_tg else "FAIL (or skipped with warning)")

    if not ok_db or not ok_mt5:
        return 1
    if not ok_tg and os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
