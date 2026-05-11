"""
tests.test_vibe_swarm
~~~~~~~~~~~~~~~~~~~~~

Regresión para la Fase-5 de VIBE: Swarm Intelligence activa.

Valida ``vibe.swarm_loops.parse_swarm_recommendation`` y el ajuste dinámico
de thresholds en ``bot.signal_emitter._swarm_adjusted_threshold``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from vibe.swarm_loops import parse_swarm_recommendation
from bot.signal_emitter import _swarm_adjusted_threshold


# ── Tests de parsing de swarm ──────────────────────────────────────────────────

class TestParseSwarmRecommendation:
    def test_strong_buy_keywords(self) -> None:
        result = {"content": [{"type": "text", "text": "Strong buy signal with conviction."}]}
        assert parse_swarm_recommendation(result) == "STRONG_BUY"

    def test_buy_keywords(self) -> None:
        result = {"content": [{"type": "text", "text": "Bullish outlook, consider long entry."}]}
        assert parse_swarm_recommendation(result) == "BUY"

    def test_reduce_keywords(self) -> None:
        result = {"content": [{"type": "text", "text": "Trim positions and de-risk exposure."}]}
        assert parse_swarm_recommendation(result) == "REDUCE"

    def test_exit_keywords(self) -> None:
        result = {"content": [{"type": "text", "text": "Exit all longs and go flat."}]}
        assert parse_swarm_recommendation(result) == "EXIT"

    def test_neutral_when_no_keywords(self) -> None:
        result = {"content": [{"type": "text", "text": "Market is ranging with no clear direction."}]}
        assert parse_swarm_recommendation(result) == "NEUTRAL"

    def test_none_returns_neutral(self) -> None:
        assert parse_swarm_recommendation(None) == "NEUTRAL"


# ── Tests de ajuste dinámico de threshold ──────────────────────────────────────

class TestSwarmAdjustedThreshold:
    """Test 1-4 obligatorios de la Fase 5."""

    def test_strong_buy_reduces_threshold(self) -> None:
        """Test 1: STRONG_BUY reduce el threshold y permite un trade bloqueado."""
        state = {"vibe_swarm_recommendation": {"BTC/USDT": "STRONG_BUY"}}
        adjusted = _swarm_adjusted_threshold(0.62, state, "BTC/USDT")
        assert adjusted == pytest.approx(0.558, abs=0.001)  # 0.62 * 0.9
        assert adjusted < 0.62

    def test_exit_raises_threshold(self) -> None:
        """Test 2: EXIT sube el threshold y bloquea un trade que pasaría."""
        state = {"vibe_swarm_recommendation": {"BTC/USDT": "EXIT"}}
        adjusted = _swarm_adjusted_threshold(0.62, state, "BTC/USDT")
        assert adjusted == pytest.approx(0.93, abs=0.001)  # 0.62 * 1.5
        assert adjusted > 0.62

    def test_neutral_leaves_threshold_unchanged(self) -> None:
        """Test 3: NEUTRAL (o vacío) no altera el threshold."""
        state = {"vibe_swarm_recommendation": {}}
        adjusted = _swarm_adjusted_threshold(0.62, state, "BTC/USDT")
        assert adjusted == 0.62

    def test_bounds_respected(self) -> None:
        """Test 4: Bounds mínimo 0.50 y máximo 0.99 se respetan."""
        # Muy bajo
        state_low = {"vibe_swarm_recommendation": {"BTC/USDT": "STRONG_BUY"}}
        adjusted_low = _swarm_adjusted_threshold(0.52, state_low, "BTC/USDT")
        assert adjusted_low >= 0.50

        # Muy alto
        state_high = {"vibe_swarm_recommendation": {"BTC/USDT": "EXIT"}}
        adjusted_high = _swarm_adjusted_threshold(0.70, state_high, "BTC/USDT")
        assert adjusted_high <= 0.99

    def test_reduce_raises_threshold_moderately(self) -> None:
        state = {"vibe_swarm_recommendation": {"BTC/USDT": "REDUCE"}}
        adjusted = _swarm_adjusted_threshold(0.60, state, "BTC/USDT")
        assert adjusted == pytest.approx(0.72, abs=0.001)  # 0.60 * 1.2

    def test_buy_slightly_reduces(self) -> None:
        state = {"vibe_swarm_recommendation": {"BTC/USDT": "BUY"}}
        adjusted = _swarm_adjusted_threshold(0.60, state, "BTC/USDT")
        assert adjusted == pytest.approx(0.57, abs=0.001)  # 0.60 * 0.95

    def test_missing_symbol_defaults_to_neutral(self) -> None:
        state = {"vibe_swarm_recommendation": {"ETH/USDT": "EXIT"}}
        adjusted = _swarm_adjusted_threshold(0.62, state, "BTC/USDT")
        assert adjusted == 0.62


# ── Tests de integración del swarm loop ────────────────────────────────────────

class TestCryptoDeskSwarmLoop:
    @pytest.mark.asyncio
    async def test_loop_parses_and_stores_recommendation(self) -> None:
        """El loop debe llamar run_swarm, parsear y guardar en shared_state."""
        from vibe.swarm_loops import crypto_desk_swarm_loop

        mock_client = AsyncMock()
        mock_client.available = True
        mock_client.run_swarm = AsyncMock(
            return_value={"content": [{"type": "text", "text": "Strong buy conviction."}]}
        )

        state: dict[str, Any] = {}

        # Solo correr una iteración cancelando después
        task = asyncio.create_task(
            crypto_desk_swarm_loop(
                mock_client, state, ["BTC/USDT"],
                interval_s=3600, initial_delay_s=0,
            )
        )
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verificar que se guardó la recomendación
        assert "vibe_swarm_recommendation" in state
        assert state["vibe_swarm_recommendation"]["BTC/USDT"] == "STRONG_BUY"

    @pytest.mark.asyncio
    async def test_loop_handles_failure_gracefully(self) -> None:
        """El loop no debe morir si run_swarm falla."""
        from vibe.swarm_loops import crypto_desk_swarm_loop

        mock_client = AsyncMock()
        mock_client.available = True
        mock_client.run_swarm = AsyncMock(side_effect=Exception("LLM timeout"))

        state: dict[str, Any] = {}

        task = asyncio.create_task(
            crypto_desk_swarm_loop(
                mock_client, state, ["BTC/USDT"],
                interval_s=3600, initial_delay_s=0,
            )
        )
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # State debe seguir existiendo aunque vacío
        assert isinstance(state, dict)
