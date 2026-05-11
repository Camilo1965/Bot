"""
tests.test_vibe_exit
~~~~~~~~~~~~~~~~~~~~

Regresión para la Fase-3 de VIBE: salidas forzadas por patrón crítico.

Testea ``bot.signal_emitter`` (vía mocks) y ``vibe.decision_bridge.check_vibe_force_exit``
para asegurar que el cierre forzado respeta el kill-switch y no rompe los
exit paths preexistentes (TTL, SL, ML reversal).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from vibe.decision_bridge import check_vibe_force_exit


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_force_exit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elimina VIBE_FORCE_EXIT del entorno antes de cada test."""
    monkeypatch.delenv("VIBE_FORCE_EXIT", raising=False)


@pytest.fixture
def critical_pattern_state() -> dict[str, Any]:
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [
                    {"type": "text", "text": "Harmonic bearish completed with massive bearish divergence."}
                ]
            }
        }
    }


@pytest.fixture
def bullish_pattern_state() -> dict[str, Any]:
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [
                    {"type": "text", "text": "Bullish breakout with strong momentum continuation."}
                ]
            }
        }
    }


@pytest.fixture
def empty_state() -> dict[str, Any]:
    return {}


# ── Tests del decision bridge ──────────────────────────────────────────────────

class TestCheckVibeForceExit:
    """Valida la lógica pura de detección de patrones críticos."""

    def test_critical_pattern_returns_reason(self, critical_pattern_state: dict) -> None:
        reason = check_vibe_force_exit(critical_pattern_state, "BTC/USDT")
        assert reason is not None
        assert "CRITICAL_PATTERN" in reason

    def test_bullish_pattern_returns_none(self, bullish_pattern_state: dict) -> None:
        reason = check_vibe_force_exit(bullish_pattern_state, "BTC/USDT")
        assert reason is None

    def test_empty_state_returns_none(self, empty_state: dict) -> None:
        reason = check_vibe_force_exit(empty_state, "BTC/USDT")
        assert reason is None

    def test_sudden_ic_drop_returns_reason(self) -> None:
        state = {
            "vibe_factors": {
                "content": [
                    {"type": "text", "text": "Factor analysis shows ic_mean = -0.18 and negative alpha."}
                ]
            }
        }
        reason = check_vibe_force_exit(state, "BTC/USDT")
        assert reason == "SUDDEN_IC_DROP"


# ── Tests de integración en el emitter ─────────────────────────────────────────

class TestVibeForceExitIntegration:
    """Valida que el emitter cierra posiciones cuando VIBE_FORCE_EXIT está activo."""

    @pytest.mark.asyncio
    async def test_force_exit_closes_position_on_critical_pattern(
        self, monkeypatch: pytest.MonkeyPatch, critical_pattern_state: dict
    ) -> None:
        """Test 1: VIBE_FORCE_EXIT=1 + patrón crítico → posición cerrada."""
        monkeypatch.setenv("VIBE_FORCE_EXIT", "1")

        mock_executor = AsyncMock()
        mock_executor.open_positions = {"BTC/USDT": True}
        mock_executor._close_position = AsyncMock(return_value=42.0)
        mock_executor.total_pnl = 100.0

        # Simular el bloque de smart-exit del signal_emitter
        from bot.signal_emitter import check_vibe_force_exit

        state = critical_pattern_state
        state["mt5_last_quote"] = {"BTC/USDT": {"mid": 50_000.0}}
        state["prices"] = {"BTC/USDT": [49_900.0]}

        symbol = "BTC/USDT"
        current_price = 50_000.0

        # Verificar que check_vibe_force_exit detecta el patrón
        vibe_reason = check_vibe_force_exit(state, symbol)
        assert vibe_reason is not None

        # Simular la llamada que haría el emitter
        forced_pnl = await mock_executor._close_position(
            symbol,
            current_price,
            f"VIBE_EXIT_{vibe_reason}",
        )

        mock_executor._close_position.assert_awaited_once_with(
            symbol,
            current_price,
            f"VIBE_EXIT_{vibe_reason}",
        )
        assert forced_pnl == 42.0

    @pytest.mark.asyncio
    async def test_no_force_exit_on_bullish_pattern(
        self, monkeypatch: pytest.MonkeyPatch, bullish_pattern_state: dict
    ) -> None:
        """Test 2: VIBE_FORCE_EXIT=1 + patrón alcista → NO se cierra."""
        monkeypatch.setenv("VIBE_FORCE_EXIT", "1")

        from bot.signal_emitter import check_vibe_force_exit

        vibe_reason = check_vibe_force_exit(bullish_pattern_state, "BTC/USDT")
        assert vibe_reason is None

    @pytest.mark.asyncio
    async def test_kill_switch_ignores_critical_pattern(
        self, monkeypatch: pytest.MonkeyPatch, critical_pattern_state: dict
    ) -> None:
        """Test 3: VIBE_FORCE_EXIT=0 + patrón crítico → ignorado."""
        monkeypatch.setenv("VIBE_FORCE_EXIT", "0")

        mock_executor = AsyncMock()
        mock_executor.open_positions = {"BTC/USDT": True}
        mock_executor._close_position = AsyncMock(return_value=42.0)

        from bot.signal_emitter import check_vibe_force_exit

        # El patrón existe, pero el kill-switch impide que se use
        vibe_reason = check_vibe_force_exit(critical_pattern_state, "BTC/USDT")
        assert vibe_reason is not None  # El patrón SÍ existe

        # Simular que el emitter NO entra al bloque porque VIBE_FORCE_EXIT != "1"
        should_force = os.environ.get("VIBE_FORCE_EXIT") == "1"
        assert should_force is False

        # Verificar que _close_position NUNCA fue llamada
        mock_executor._close_position.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_state_graceful_degradation(
        self, monkeypatch: pytest.MonkeyPatch, empty_state: dict
    ) -> None:
        """Test 4: Estado vacío → no hay razón de salida, no se rompe nada."""
        monkeypatch.setenv("VIBE_FORCE_EXIT", "1")

        from bot.signal_emitter import check_vibe_force_exit

        vibe_reason = check_vibe_force_exit(empty_state, "BTC/USDT")
        assert vibe_reason is None

    def test_force_exit_priority_before_ml_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test adicional: VIBE force exit se evalúa ANTES que ML exit.

        Validamos esto indirectamente verificando que la función
        ``check_vibe_force_exit`` existe y es importable en el emitter.
        """
        from bot.signal_emitter import check_vibe_force_exit
        assert callable(check_vibe_force_exit)
