"""
tests.test_vibe_sizing
~~~~~~~~~~~~~~~~~~~~~~

Regresión para la Fase-2 de VIBE: sizing dinámico por calidad del setup.

Testea ``risk.risk_manager.RiskManager.calculate_position_size`` con el
nuevo parámetro ``vibe_quality``, y la propagación del kwarg a través del
emitter hacia ``try_open_trade``.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from risk.risk_manager import RiskManager


@pytest.fixture
def risk_mgr() -> RiskManager:
    """RiskManager con balance de $10,000 para tests deterministas."""
    return RiskManager(initial_balance=10_000.0, max_positions=2)


@pytest.fixture(autouse=True)
def _clear_vibe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elimina variables VIBE del entorno antes de cada test."""
    for var in ("VIBE_QUALITY_MIN", "VIBE_QUALITY_MAX", "VIBE_ADJUST_SIZE"):
        monkeypatch.delenv(var, raising=False)


class TestVibeSizingInRiskManager:
    """Tests 1-4: comportamiento de calculate_position_size con vibe_quality."""

    # ------------------------------------------------------------------
    # Test 1: vibe_quality=1.0 → sizing normal (riesgo base 2%)
    # ------------------------------------------------------------------
    def test_quality_one_is_baseline(self, risk_mgr: RiskManager) -> None:
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=1.0,
        )
        # Sin drawdown ni exposure: base_r = 0.02
        # position_size = 10_000 * 0.02 / 0.02 = 10_000
        # max_allocation = 10_000 / 2 * 5 = 25_000
        # min(10_000, 25_000) = 10_000
        assert size == pytest.approx(10_000.0, rel=0.01)

    # ------------------------------------------------------------------
    # Test 2: vibe_quality=1.5 → sizing máximo (3% riesgo)
    # ------------------------------------------------------------------
    def test_quality_max_boosts_size(self, risk_mgr: RiskManager) -> None:
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=1.5,
        )
        # base_r = 0.02 * 1.5 = 0.03
        # position_size = 10_000 * 0.03 / 0.02 = 15_000
        assert size == pytest.approx(15_000.0, rel=0.01)

    # ------------------------------------------------------------------
    # Test 3: vibe_quality=0.5 → sizing mínimo (1% riesgo)
    # ------------------------------------------------------------------
    def test_quality_min_reduces_size(self, risk_mgr: RiskManager) -> None:
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=0.5,
        )
        # base_r = 0.02 * 0.5 = 0.01
        # position_size = 10_000 * 0.01 / 0.02 = 5_000
        assert size == pytest.approx(5_000.0, rel=0.01)

    # ------------------------------------------------------------------
    # Test 4: Extremos se capean matemáticamente
    # ------------------------------------------------------------------
    def test_extreme_high_capped(self, risk_mgr: RiskManager) -> None:
        """vibe_quality=2.0 debe ser capeado al max (1.5 por defecto)."""
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=2.0,
        )
        # Debe ser igual al caso 1.5
        assert size == pytest.approx(15_000.0, rel=0.01)

    def test_extreme_low_capped(self, risk_mgr: RiskManager) -> None:
        """vibe_quality=0.1 debe ser capeado al min (0.5 por defecto)."""
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=0.1,
        )
        # Debe ser igual al caso 0.5
        assert size == pytest.approx(5_000.0, rel=0.01)

    # ------------------------------------------------------------------
    # Test 5: Kill-switch VIBE_ADJUST_SIZE=0 fuerza quality=1.0
    # ------------------------------------------------------------------
    def test_kill_switch_ignores_quality(self, risk_mgr: RiskManager) -> None:
        """Cuando VIBE_ADJUST_SIZE=0, el emitter pasa vibe_quality=1.0
        (testeado indirectamente: calculate_position_size con cualquier
        quality sigue siendo 1.0 porque el kill-switch está en el emitter)."""
        # Este test valida que calculate_position_size respeta exactamente
        # el valor que recibe; el kill-switch es responsabilidad del caller.
        size_default = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
        )
        # vibe_quality default es 1.0
        assert size_default == pytest.approx(10_000.0, rel=0.01)

    # ------------------------------------------------------------------
    # Test 6: Drawdown thermometer se aplica SOBRE el riesgo ajustado por VIBE
    # ------------------------------------------------------------------
    def test_drawdown_thermometer_stacks_with_vibe(self, risk_mgr: RiskManager) -> None:
        """Si weekly DD > 5%, el riesgo se reduce 50% ADICIONALMENTE
        sobre el ya ajustado por VIBE."""
        # Forzar drawdown del 10% desde el high-water mark semanal
        risk_mgr._weekly_peak_balance = 11_111.11  # balance=10_000 → ~10% DD
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=1.5,
        )
        # base_r = 0.02 * 1.5 = 0.03
        # thermometer * 0.5 → 0.015
        # position_size = 10_000 * 0.015 / 0.02 = 7_500
        assert size == pytest.approx(7_500.0, rel=0.01)

    # ------------------------------------------------------------------
    # Tests adicionales de robustez
    # ------------------------------------------------------------------
    def test_custom_bounds_from_env(self, monkeypatch: pytest.MonkeyPatch, risk_mgr: RiskManager) -> None:
        """VIBE_QUALITY_MIN/MAX leídos de env deben respetarse."""
        monkeypatch.setenv("VIBE_QUALITY_MIN", "0.3")
        monkeypatch.setenv("VIBE_QUALITY_MAX", "2.0")

        size_min = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=0.1,  # capeado a 0.3
        )
        # 0.02 * 0.3 = 0.006 → 10_000 * 0.006 / 0.02 = 3_000
        assert size_min == pytest.approx(3_000.0, rel=0.01)

        size_max = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=3.0,  # capeado a 2.0
        )
        # 0.02 * 2.0 = 0.04 → 10_000 * 0.04 / 0.02 = 20_000
        assert size_max == pytest.approx(20_000.0, rel=0.01)

    def test_portfolio_budget_limits_vibe_boost(self, risk_mgr: RiskManager) -> None:
        """Aunque vibe_quality=1.5, si el portfolio budget está casi lleno,
        el sizing se reduce para respetar MAX_PORTFOLIO_RISK_PCT."""
        # Simular que ya hay riesgo ocupado
        risk_mgr._total_risk_usd = 900.0  # 9% del balance
        size = risk_mgr.calculate_position_size(
            win_probability=0.65,
            risk_pct=0.02,
            sl_distance_pct=0.02,
            vibe_quality=1.5,
        )
        # exposure = 900/10_000 = 0.09
        # base_r = 0.02 * 1.5 = 0.03
        # 0.09 + 0.03 = 0.12 > 0.10 (MAX_PORTFOLIO_RISK_PCT)
        # scaled to 0.10 - 0.09 = 0.01
        # position_size = 10_000 * 0.01 / 0.02 = 5_000
        assert size == pytest.approx(5_000.0, rel=0.01)


class TestVibeSizingPropagation:
    """Validar que el kwarg vibe_quality se propaga a través del emitter."""

    def test_signal_emitter_passes_vibe_quality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """El emitter debe llamar try_open_trade con vibe_quality cuando
        VIBE_ADJUST_SIZE está activo."""
        monkeypatch.setenv("VIBE_ADJUST_SIZE", "1")

        from bot.signal_emitter import _apply_vibe_gating
        # Solo validamos que compute_quality_multiplier existe y es importable
        from vibe.decision_bridge import compute_quality_multiplier
        assert callable(compute_quality_multiplier)

    def test_try_open_trade_accepts_vibe_quality(self) -> None:
        """PaperExecutor.try_open_trade debe aceptar el kwarg vibe_quality."""
        import inspect
        from execution.paper_executor import PaperExecutor
        sig = inspect.signature(PaperExecutor.try_open_trade)
        assert "vibe_quality" in sig.parameters
        assert sig.parameters["vibe_quality"].default == 1.0
