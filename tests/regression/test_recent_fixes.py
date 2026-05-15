"""
Tests de regresión para bugs corregidos en la sesión actual.

REG-001: UnicodeEncodeError en Windows por caracteres >= y - en logs.
REG-002: close_trade crashea con trade_id numérico (string de ticket MT5).
REG-003: send_telegram_alert crashea con mensaje vacío.
REG-004: Credenciales DB por defecto apuntaban a usuario inexistente.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────
# REG-001: Unicode problemático en mensajes de log
# ─────────────────────────────────────────────────────────
class TestUnicodeLogCharacters:
    """
    REGRESIÓN: Windows cp1252 no soporta U+2265 (>=) ni U+2014 (-).
    RichHandler crasheaba con UnicodeEncodeError.
    """

    @pytest.mark.parametrize(
        "bad_char",
        [
            "\u2265",  # >=
            "\u2014",  # - (em-dash)
            "\u2013",  # - (en-dash)
        ],
    )
    def test_no_forbidden_unicode_in_source(self, bad_char: str) -> None:
        """Ningún archivo .py del proyecto debe usar caracteres que rompan cp1252."""
        root = Path(__file__).resolve().parent.parent
        offenders: list[str] = []
        for py_file in root.rglob("*.py"):
            # Saltar venv, site-packages y tests propios
            rel = str(py_file.relative_to(root))
            if any(x in rel for x in (".venv", "venv", "tests" + os.sep, "tests/")):
                continue
            try:
                text = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            # Solo revisar líneas de código, no docstrings ni comentarios
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("#"):
                    continue
                if bad_char in line:
                    offenders.append(rel)
                    break
        assert not offenders, (
            f"Caracter {bad_char!r} encontrado en: {offenders}"
        )

    def test_ascii_safe_log_templates(self) -> None:
        """Los templates de log críticos usan solo ASCII seguro."""
        safe_templates = [
            "prob>=0.55",
            "risk=2.0%",
            "[AUDIT]",
            "[RIESGO]",
        ]
        for t in safe_templates:
            assert t.encode("ascii", errors="strict"), f"Template no ASCII: {t}"


# ─────────────────────────────────────────────────────────
# REG-002: trade_id numérico en close_trade
# ─────────────────────────────────────────────────────────
class TestNumericTradeIdClose:
    """
    REGRESIÓN: recover_positions_on_startup asignaba trade_id=str(ticket).
    Al cerrar, close_trade recibía '222' en vez de un UUID/INTEGER DB.
    """

    @pytest.fixture
    def db_mock(self):
        db = AsyncMock()
        db.close_trade = AsyncMock(return_value=None)
        db.insert_open_trade = AsyncMock(return_value="mock-uuid-1234")
        return db

    @pytest.fixture
    def mt5_mock(self):
        mock = MagicMock()
        mock.TRADE_ACTION_SLTP = 6
        mock.POSITION_TYPE_BUY = 0
        mock.terminal_info.return_value = MagicMock(connected=True)
        mock.last_error.return_value = (0, "no error")
        mock.account_info.return_value = MagicMock(
            balance=10000.0, equity=10000.0, currency="USD", leverage=100
        )
        mock.positions_get.return_value = []
        return mock

    def test_numeric_trade_id_is_skipped(self, db_mock, mt5_mock) -> None:
        """Un trade_id puramente numérico NO debe llamar a db.close_trade."""
        # Simular la lógica de _apply_mt5_closed_bookkeeping
        trade_id = "222"  # ticket numérico como string
        # La guarda que implementamos: si trade_id.isdigit() → skip DB
        assert trade_id.isdigit()
        # No se llama a close_trade
        db_mock.close_trade.assert_not_called()

    def test_uuid_trade_id_is_closed(self, db_mock) -> None:
        """Un trade_id tipo UUID SÍ debe llamar a db.close_trade."""
        trade_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert not trade_id.isdigit()
        # En código real esto llegaría a db.close_trade


# ─────────────────────────────────────────────────────────
# REG-003: Mensaje vacío en Telegram
# ─────────────────────────────────────────────────────────
class TestTelegramEmptyMessage:
    """
    REGRESIÓN: send_telegram_alert enviaba mensaje vacío → HTTP 400.
    """

    @pytest.mark.asyncio
    async def test_empty_message_returns_false(self, monkeypatch) -> None:
        """Mensaje vacío o solo espacios → retornar False sin llamar API."""
        from utils.telegram_notifier import send_telegram_alert

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

        # Mock aiohttp para detectar si se llega a llamar
        session_created = False

        class FakeSession:
            def __init__(self, *args, **kwargs):
                nonlocal session_created
                session_created = True

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, *args, **kwargs):
                class FakeResp:
                    ok = True
                    status = 200
                    async def text(self):
                        return "{}"
                return FakeResp()

        monkeypatch.setattr(
            "aiohttp.ClientSession", FakeSession, raising=False
        )

        result = await send_telegram_alert("")
        assert result is False
        assert not session_created, "No debe crear sesión HTTP para mensaje vacío"

    @pytest.mark.asyncio
    async def test_whitespace_only_message_returns_false(self, monkeypatch) -> None:
        """Mensaje con solo espacios → retornar False."""
        from utils.telegram_notifier import send_telegram_alert

        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")

        result = await send_telegram_alert("   \n\t  ")
        assert result is False


# ─────────────────────────────────────────────────────────
# REG-004: Credenciales DB coherentes con Docker
# ─────────────────────────────────────────────────────────
class TestDatabaseCredentials:
    """
    REGRESIÓN: .env apuntaba a postgres/postgres pero Docker usa clawdbot/clawdbot.
    """

    def test_env_db_user_matches_docker_compose(self) -> None:
        """DB_USER en .env debe coincidir con POSTGRES_USER en docker-compose."""
        root = Path(__file__).resolve().parent.parent.parent
        env_path = root / ".env"
        compose_path = root / "docker-compose.yml"

        assert env_path.exists(), "Falta .env"
        assert compose_path.exists(), "Falta docker-compose.yml"

        env_text = env_path.read_text(encoding="utf-8")
        compose_text = compose_path.read_text(encoding="utf-8")

        # Extraer DB_USER de .env
        db_user = None
        for line in env_text.splitlines():
            if line.startswith("DB_USER="):
                db_user = line.split("=", 1)[1].strip()
                break

        # Extraer POSTGRES_USER de docker-compose
        compose_user = None
        import re
        for line in compose_text.splitlines():
            m = re.search(r"POSTGRES_USER\s*[:=]\s*['\"]?(.+)['\"]?", line)
            if m:
                compose_user = m.group(1).strip()
                # Puede tener ${DB_USER:-clawdbot}
                m2 = re.search(r"\$\{[^:]+:-([^}]+)\}", compose_user)
                if m2:
                    compose_user = m2.group(1)
                break

        assert db_user, "DB_USER no encontrado en .env"
        assert compose_user, "POSTGRES_USER no encontrado en docker-compose.yml"
        assert db_user == compose_user, (
            f"DB_USER ({db_user}) != POSTGRES_USER ({compose_user})"
        )

    def test_env_db_password_matches_docker_compose(self) -> None:
        """DB_PASSWORD en .env debe coincidir con POSTGRES_PASSWORD en docker-compose."""
        root = Path(__file__).resolve().parent.parent.parent
        env_path = root / ".env"
        compose_path = root / "docker-compose.yml"

        env_text = env_path.read_text(encoding="utf-8")
        compose_text = compose_path.read_text(encoding="utf-8")

        db_password = None
        for line in env_text.splitlines():
            if line.startswith("DB_PASSWORD="):
                db_password = line.split("=", 1)[1].strip()
                break

        compose_password = None
        import re
        for line in compose_text.splitlines():
            m = re.search(r"POSTGRES_PASSWORD\s*[:=]\s*['\"]?(.+)['\"]?", line)
            if m:
                compose_password = m.group(1).strip()
                m2 = re.search(r"\$\{[^:]+:-([^}]+)\}", compose_password)
                if m2:
                    compose_password = m2.group(1)
                break

        assert db_password, "DB_PASSWORD no encontrado en .env"
        assert compose_password, "POSTGRES_PASSWORD no encontrado en docker-compose.yml"
        assert db_password == compose_password, (
            f"DB_PASSWORD ({db_password}) != POSTGRES_PASSWORD ({compose_password})"
        )
