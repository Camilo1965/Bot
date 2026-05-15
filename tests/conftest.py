"""
conftest.py - Fixtures globales para ClawdBot test suite.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Asegurar que el proyecto está en sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────
# Fixture: estado de logging limpio
# ─────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_logging():
    """Limpiar handlers de logging entre tests para evitar duplicados."""
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    root.handlers.clear()
    for h in old_handlers:
        root.addHandler(h)
    root.setLevel(old_level)


# ─────────────────────────────────────────────────────────
# Fixture: env limpio
# ─────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Restaurar variables de entorno críticas después de cada test."""
    preserved = {
        k: os.environ.get(k)
        for k in (
            "EXECUTION_MODE",
            "MT5_LOGIN",
            "MT5_PASSWORD",
            "MT5_SERVER",
            "DATABASE_URL",
            "DB_USER",
            "DB_PASSWORD",
            "DB_HOST",
            "DB_PORT",
            "DB_NAME",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        )
    }
    yield
    for k, v in preserved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ─────────────────────────────────────────────────────────
# Fixture: event loop para tests async
# ─────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Proveer un event loop reusable para toda la sesión de tests."""
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─────────────────────────────────────────────────────────
# Fixture: mock de Telegram notifier
# ─────────────────────────────────────────────────────────
@pytest.fixture
def mock_telegram_alert(monkeypatch):
    """Mock de send_telegram_alert para evitar llamadas reales a la API."""
    async def _mock(message: str, parse_mode: str | None = None) -> bool:
        return bool(message and message.strip())

    monkeypatch.setattr(
        "utils.telegram_notifier.send_telegram_alert", _mock, raising=False
    )
    monkeypatch.setattr(
        "utils.telegram_notifier.send_priority_telegram_alert", _mock, raising=False
    )
    return _mock
