"""
tests.test_vibe_features
~~~~~~~~~~~~~~~~~~~~~~~~

Regresión para la Fase-4 de VIBE: features numéricas alimentando XGBoost.

Valida que ``vibe.feature_bridge.extract_vibe_features`` retorne vectores
estables y que ``strategy.ml_predictor`` los ensamble correctamente en el
vector de inferencia sin romper compatibilidad legacy.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from vibe.feature_bridge import extract_vibe_features, VIBE_FEATURE_NEUTRAL


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_vibe_features_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elimina VIBE_FEATURES_ENABLED del entorno antes de cada test."""
    monkeypatch.delenv("VIBE_FEATURES_ENABLED", raising=False)


@pytest.fixture
def rich_vibe_state() -> dict[str, Any]:
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Bullish breakout with higher highs."}]
            }
        },
        "vibe_factors": {
            "content": [{"type": "text", "text": "IC mean = 0.22, strong predictive factor."}]
        },
        "vibe_journal_analysis": {
            "content": [{"type": "text", "text": "Disciplined trading with consistent edge confirmed."}]
        },
        "vibe_backtest": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Sharpe ratio = 1.4, profitable strategy."}]
            }
        },
    }


@pytest.fixture
def empty_state() -> dict[str, Any]:
    return {}


# ── Tests del feature bridge ───────────────────────────────────────────────────

class TestExtractVibeFeatures:
    """Valida la extracción de features VIBE y el fallback neutral."""

    def test_returns_four_floats(self, rich_vibe_state: dict) -> None:
        vec = extract_vibe_features(rich_vibe_state, "BTC/USDT")
        assert len(vec) == 4
        assert all(isinstance(v, float) for v in vec)

    def test_empty_state_returns_neutral(self, empty_state: dict) -> None:
        vec = extract_vibe_features(empty_state, "BTC/USDT")
        assert vec == VIBE_FEATURE_NEUTRAL

    def test_none_state_returns_neutral(self) -> None:
        vec = extract_vibe_features(None, "BTC/USDT")
        assert vec == VIBE_FEATURE_NEUTRAL

    def test_values_are_finite(self, rich_vibe_state: dict) -> None:
        vec = extract_vibe_features(rich_vibe_state, "BTC/USDT")
        assert all(abs(v) < 10.0 for v in vec)


# ── Tests de integración con ml_predictor ──────────────────────────────────────

class TestMLPredictorVibeFeatures:
    """Valida que ml_predictor maneja correctamente el vector con/sin VIBE."""

    def test_predict_proba_with_vibe_enabled_appends_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test 1: Vector base (24) + VIBE activado = 28 features totales."""
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "1")

        # Recargar ml_predictor para que _FEATURE_COLS se calcule con VIBE=1
        import importlib
        import strategy.ml_predictor as _mp
        importlib.reload(_mp)

        predictor = _mp.MLPredictor()
        # Forzar warm_start con datos dummy para que el modelo esté entrenado
        prices = list(range(1000, 2000))
        highs = [p + 10 for p in prices]
        lows = [p - 10 for p in prices]
        predictor.warm_start(prices, highs=highs, lows=lows)

        vibe_vec = extract_vibe_features({}, "BTC/USDT")
        proba = predictor.predict_proba(
            prices,
            symbol="BTC/USDT",
            vibe_features=vibe_vec,
        )
        # Si el modelo está entrenado y las features coinciden, retorna float
        assert proba is not None
        assert isinstance(proba, float)
        assert 0.0 <= proba <= 1.0

    def test_predict_proba_with_empty_state_uses_neutral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test 2: State vacío + VIBE activado → 4 features neutrales añadidas."""
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "1")

        import importlib
        import strategy.ml_predictor as _mp
        importlib.reload(_mp)

        predictor = _mp.MLPredictor()
        prices = list(range(1000, 2000))
        highs = [p + 10 for p in prices]
        lows = [p - 10 for p in prices]
        predictor.warm_start(prices, highs=highs, lows=lows)

        # Pasar vibe_features=None debe auto-pad con neutrales
        proba = predictor.predict_proba(
            prices,
            symbol="BTC/USDT",
            vibe_features=None,
        )
        assert proba is not None
        assert isinstance(proba, float)

    def test_predict_proba_without_vibe_keeps_original_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test 3: VIBE desactivado (default) → vector mantiene tamaño original (24)."""
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "0")

        import importlib
        import strategy.ml_predictor as _mp
        importlib.reload(_mp)

        predictor = _mp.MLPredictor()
        prices = list(range(1000, 2000))
        highs = [p + 10 for p in prices]
        lows = [p - 10 for p in prices]
        predictor.warm_start(prices, highs=highs, lows=lows)

        proba = predictor.predict_proba(
            prices,
            symbol="BTC/USDT",
        )
        assert proba is not None
        assert isinstance(proba, float)

    def test_model_path_changes_with_vibe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cuando VIBE_FEATURES_ENABLED=1, el path default es _v3.json."""
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "1")

        import importlib
        import strategy.ml_predictor as _mp
        importlib.reload(_mp)

        # Use a symbol that has NO model on disk so fallback chain returns the default
        path = _mp.model_json_path_for_symbol("FAKE/TEST")
        assert "_v3.json" in str(path)

    def test_model_path_legacy_without_vibe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cuando VIBE desactivado, el path default es _v2.json."""
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "0")

        import importlib
        import strategy.ml_predictor as _mp
        importlib.reload(_mp)

        path = _mp.model_json_path_for_symbol("FAKE/TEST")
        assert "_v2.json" in str(path)

    def test_generate_signal_passes_vibe_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_signal acepta y pasa vibe_features a predict_proba."""
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "1")

        import importlib
        import strategy.ml_predictor as _mp
        importlib.reload(_mp)

        predictor = _mp.MLPredictor()
        prices = list(range(1000, 2000))
        highs = [p + 10 for p in prices]
        lows = [p - 10 for p in prices]
        predictor.warm_start(prices, highs=highs, lows=lows)

        vibe_vec = [0.5, 0.1, 0.8, 0.3]
        signal, prob = predictor.generate_signal(
            prices,
            highs=highs,
            lows=lows,
            symbol="BTC/USDT",
            vibe_features=vibe_vec,
        )
        assert signal in ("BUY", "HOLD")
        assert isinstance(prob, float)

    def test_feature_cols_dynamic_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_FEATURE_COLS cambia de tamaño según VIBE_FEATURES_ENABLED."""
        import importlib
        import strategy.ml_predictor as _mp
        import strategy.quant_features as _qf

        # Desactivado
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "0")
        importlib.reload(_mp)
        assert len(_mp._FEATURE_COLS) == len(_qf.QUANT_FEATURE_COLS)  # 24

        # Activado
        monkeypatch.setenv("VIBE_FEATURES_ENABLED", "1")
        importlib.reload(_mp)
        assert len(_mp._FEATURE_COLS) == len(_qf.QUANT_FEATURE_COLS) + len(_qf.VIBE_FEATURE_COLS)  # 28
