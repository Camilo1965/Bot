"""
tests.test_vibe_gating
~~~~~~~~~~~~~~~~~~~~~~

Regresión para la Fase-1 de VIBE: gating de entradas con veto probabilístico híbrido.

Testea ``bot.signal_emitter._apply_vibe_gating`` directamente (función pura,
sin necesidad de asyncio ni de arrancar el emitter completo).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from bot.signal_emitter import _apply_vibe_gating


@pytest.fixture(autouse=True)
def _clear_vibe_gate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elimina VIBE_GATE_TRADES del entorno antes de cada test para evitar
    interferencias del .env del usuario."""
    monkeypatch.delenv("VIBE_GATE_TRADES", raising=False)


class TestVibeGating:
    """Suite de 4 tests obligatorios para la Fase-1 de integración VIBE."""

    # ------------------------------------------------------------------
    # Test 1: Veto absoluto bloquea trade aunque win_prob = 0.99
    # ------------------------------------------------------------------
    def test_extreme_veto_blocks_even_99_prob(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIBE_GATE_TRADES", "1")

        state: dict[str, Any] = {"vibe_patterns": {"dummy": True}}

        with patch("bot.signal_emitter.is_extreme_macro_veto", return_value=True):
            gated_signal, reason = _apply_vibe_gating(state, "BTC/USDT", "BUY", 0.99)

        assert gated_signal == "HOLD"
        assert reason == "VIBE_EXTREME_VETO"

    # ------------------------------------------------------------------
    # Test 2: vibe_score bajo reduce win_prob por debajo del threshold → bloquea
    # ------------------------------------------------------------------
    def test_low_vibe_score_reduces_prob_below_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIBE_GATE_TRADES", "1")

        state: dict[str, Any] = {}

        # BTC/USDT threshold = 0.60  →  0.65 * 0.50 = 0.325 < 0.60
        with patch("bot.signal_emitter.is_extreme_macro_veto", return_value=False), \
             patch("bot.signal_emitter.compute_entry_score", return_value=0.50):
            gated_signal, reason = _apply_vibe_gating(state, "BTC/USDT", "BUY", 0.65)

        assert gated_signal == "HOLD"
        assert reason is not None
        assert "VIBE_SCORE_0.50" in reason

    # ------------------------------------------------------------------
    # Test 3: vibe_score neutral/alto permite que el trade pase
    # ------------------------------------------------------------------
    def test_high_vibe_score_allows_trade(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIBE_GATE_TRADES", "1")

        state: dict[str, Any] = {}

        # 0.65 * 1.0 = 0.65 >= 0.60 (threshold BTC/USDT)
        with patch("bot.signal_emitter.is_extreme_macro_veto", return_value=False), \
             patch("bot.signal_emitter.compute_entry_score", return_value=1.0):
            gated_signal, reason = _apply_vibe_gating(state, "BTC/USDT", "BUY", 0.65)

        assert gated_signal == "BUY"
        assert reason is None

    # ------------------------------------------------------------------
    # Test 4: VIBE_GATE_TRADES=0 ignora veto absoluto (kill-switch manual)
    # ------------------------------------------------------------------
    def test_gate_disabled_ignores_veto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIBE_GATE_TRADES", "0")

        state: dict[str, Any] = {}

        # Aunque haya veto extremo, si el gating está apagado debe pasar
        with patch("bot.signal_emitter.is_extreme_macro_veto", return_value=True):
            gated_signal, reason = _apply_vibe_gating(state, "BTC/USDT", "BUY", 0.99)

        assert gated_signal == "BUY"
        assert reason is None

    # ------------------------------------------------------------------
    # Tests adicionales de robustez
    # ------------------------------------------------------------------
    def test_non_buy_signal_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Si la señal ya es HOLD o SELL, el gating no debe tocarla."""
        monkeypatch.setenv("VIBE_GATE_TRADES", "1")

        for sig in ("HOLD", "SELL"):
            gated_signal, reason = _apply_vibe_gating({}, "BTC/USDT", sig, 0.99)
            assert gated_signal == sig
            assert reason is None

    def test_legacy_env_values_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Acepta 'true' y 'yes' como activadores además de '1'."""
        state: dict[str, Any] = {}

        for val in ("1", "true", "yes"):
            monkeypatch.setenv("VIBE_GATE_TRADES", val)
            with patch("bot.signal_emitter.is_extreme_macro_veto", return_value=True):
                gated_signal, _ = _apply_vibe_gating(state, "BTC/USDT", "BUY", 0.99)
            assert gated_signal == "HOLD", f"failed for env value {val!r}"

    def test_no_env_defaults_to_disabled(self) -> None:
        """Sin VIBE_GATE_TRADES definido, el gating está desactivado (legacy)."""
        # _clear_vibe_gate_env ya eliminó la variable
        gated_signal, reason = _apply_vibe_gating({}, "BTC/USDT", "BUY", 0.99)
        assert gated_signal == "BUY"
        assert reason is None
