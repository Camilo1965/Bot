"""
tests.test_vibe_decision_bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regresión unitaria para ``vibe.decision_bridge``.

Asegura que los text-blobs del MCP de VIBE se traducen correctamente en
scores numéricos acotados y que el fallback neutral nunca bloquea el trading
cuando VIBE está ausente.
"""

from __future__ import annotations

import os
import pytest

from vibe.decision_bridge import (
    compute_entry_score,
    compute_quality_multiplier,
    is_extreme_macro_veto,
    parse_backtest_quality,
    parse_factor_quality,
    parse_journal_bias,
    parse_pattern_score,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_state() -> dict:
    """Estado sin ningún dato de VIBE."""
    return {}


@pytest.fixture
def neutral_state() -> dict:
    """Estado con claves de VIBE presentes pero vacías."""
    return {
        "vibe_patterns": {},
        "vibe_factors": None,
        "vibe_journal_analysis": None,
        "vibe_backtest": {},
    }


@pytest.fixture
def bullish_pattern_state() -> dict:
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Bullish breakout detected with higher highs and strong momentum up."}]
            }
        }
    }


@pytest.fixture
def bearish_pattern_state() -> dict:
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Double top bearish pattern with divergence and breakdown below support."}]
            }
        }
    }


@pytest.fixture
def mixed_vibe_state() -> dict:
    """Patrón bajista pero factor sólido y backtest positivo."""
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Bearish descending channel with lower lows."}]
            }
        },
        "vibe_factors": {
            "content": [{"type": "text", "text": "Factor close shows strong predictive power with IC mean = 0.18."}]
        },
        "vibe_journal_analysis": {
            "content": [{"type": "text", "text": "Trading profile shows disciplined risk control and positive expectancy."}]
        },
        "vibe_backtest": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Backtest profitable with Sharpe ratio = 1.4 and positive excess return."}]
            }
        },
    }


@pytest.fixture
def extreme_veto_state() -> dict:
    """Múltiples señales de riesgo extremo."""
    return {
        "vibe_patterns": {
            "BTC/USDT": {
                "content": [{"type": "text", "text": "Extreme fear detected in market structure. Flash crash signals."}]
            },
            "ETH/USDT": {
                "content": [{"type": "text", "text": "Capitulation pattern with extreme fear across all timeframes."}]
            }
        },
        "vibe_factors": {
            "content": [{"type": "text", "text": "Factor analysis shows negative alpha and ic_mean = -0.15."}]
        },
        "vibe_journal_analysis": {
            "content": [{"type": "text", "text": "Overtrading and revenge trading detected during losing streak."}]
        },
    }


# ── Tests de fallback neutral ─────────────────────────────────────────────────

class TestNeutralFallback:
    """Cuando VIBE no está disponible el bridge debe ser transparente."""

    def test_empty_state_returns_neutral_entry_score(self, empty_state: dict) -> None:
        assert compute_entry_score(empty_state, "BTC/USDT") == 1.0

    def test_empty_state_returns_neutral_multiplier(self, empty_state: dict) -> None:
        assert compute_quality_multiplier(empty_state, "BTC/USDT") == 1.0

    def test_empty_state_returns_no_veto(self, empty_state: dict) -> None:
        assert is_extreme_macro_veto(empty_state) is False

    def test_neutral_state_returns_neutral_values(self, neutral_state: dict) -> None:
        assert compute_entry_score(neutral_state, "BTC/USDT") == 1.0
        assert compute_quality_multiplier(neutral_state, "BTC/USDT") == 1.0
        assert is_extreme_macro_veto(neutral_state) is False


# ── Tests de parsing individual ────────────────────────────────────────────────

class TestParsePatternScore:
    def test_bullish_pattern_returns_positive_score(self, bullish_pattern_state: dict) -> None:
        score = parse_pattern_score(bullish_pattern_state["vibe_patterns"], "BTC/USDT")
        assert score > 0.0
        assert score <= 1.0

    def test_bearish_pattern_returns_negative_score(self, bearish_pattern_state: dict) -> None:
        score = parse_pattern_score(bearish_pattern_state["vibe_patterns"], "BTC/USDT")
        assert score < 0.0
        assert score >= -1.0

    def test_missing_symbol_returns_zero(self) -> None:
        assert parse_pattern_score({}, "BTC/USDT") == 0.0


class TestParseFactorQuality:
    def test_high_ic_returns_high_quality(self) -> None:
        data = {"content": [{"type": "text", "text": "IC mean = 0.22, strong predictive factor."}]}
        assert parse_factor_quality(data) > 0.7

    def test_negative_ic_returns_low_quality(self) -> None:
        data = {"content": [{"type": "text", "text": "IC mean = -0.12, weak predictive power."}]}
        assert parse_factor_quality(data) < 0.4

    def test_none_returns_neutral(self) -> None:
        assert parse_factor_quality(None) == 0.5


class TestParseJournalBias:
    def test_severe_biases_return_low_health(self) -> None:
        data = {"content": [{"type": "text", "text": "Overtrading and revenge trading detected. Loss aversion extreme."}]}
        assert parse_journal_bias(data) < 0.5

    def test_healthy_journal_returns_high_health(self) -> None:
        data = {"content": [{"type": "text", "text": "Disciplined trading with consistent edge confirmed."}]}
        assert parse_journal_bias(data) > 0.5

    def test_none_returns_perfect_health(self) -> None:
        # Sin journal aún no penalizamos
        assert parse_journal_bias(None) == 1.0


class TestParseBacktestQuality:
    def test_high_sharpe_returns_high_quality(self) -> None:
        data = {"BTC/USDT": {"content": [{"type": "text", "text": "Sharpe ratio = 1.8, profitable strategy."}]}}
        assert parse_backtest_quality(data, "BTC/USDT") > 0.8

    def test_low_sharpe_returns_low_quality(self) -> None:
        data = {"BTC/USDT": {"content": [{"type": "text", "text": "Sharpe ratio = 0.3, unprofitable with losses."}]}}
        assert parse_backtest_quality(data, "BTC/USDT") < 0.3

    def test_missing_symbol_returns_neutral(self) -> None:
        assert parse_backtest_quality({}, "BTC/USDT") == 0.5


# ── Tests de funciones públicas ───────────────────────────────────────────────

class TestComputeEntryScore:
    def test_bearish_pattern_lowers_score(self, bearish_pattern_state: dict) -> None:
        score = compute_entry_score(bearish_pattern_state, "BTC/USDT")
        assert score < 1.0
        # Con solo pattern bajista y resto ausente (neutral) → debe bajar del 1.0
        assert score < 0.85

    def test_bullish_pattern_helps_score(self, bullish_pattern_state: dict) -> None:
        score = compute_entry_score(bullish_pattern_state, "BTC/USDT")
        assert score > 0.5
        # Solo pattern alcista, resto neutral → score moderadamente alto
        assert score > 0.6

    def test_bounds_between_zero_and_one(self, mixed_vibe_state: dict) -> None:
        score = compute_entry_score(mixed_vibe_state, "BTC/USDT")
        assert 0.0 <= score <= 1.0

    def test_mixed_state_balanced(self, mixed_vibe_state: dict) -> None:
        # Pattern bajista pero factor/backtest/journal positivos
        score = compute_entry_score(mixed_vibe_state, "BTC/USDT")
        # Debe quedar en rango medio (ni muy alto ni muy bajo)
        assert 0.4 <= score <= 0.8


class TestComputeQualityMultiplier:
    @staticmethod
    def _reload_bridge_with_bounds(monkeypatch: pytest.MonkeyPatch, min_v: str, max_v: str):
        """Recarga vibe.decision_bridge con bounds forzados para hacer el test determinista.

        Elimina primero cualquier valor heredado del entorno del usuario para que
        el módulo caiga en los defaults esperados.
        """
        import importlib
        monkeypatch.delenv("VIBE_QUALITY_MIN", raising=False)
        monkeypatch.delenv("VIBE_QUALITY_MAX", raising=False)
        monkeypatch.setenv("VIBE_QUALITY_MIN", min_v)
        monkeypatch.setenv("VIBE_QUALITY_MAX", max_v)
        import vibe.decision_bridge as _bridge
        importlib.reload(_bridge)
        return _bridge

    def test_bounds_respected(self, mixed_vibe_state: dict) -> None:
        mult = compute_quality_multiplier(mixed_vibe_state, "BTC/USDT")
        assert 0.0 < mult <= 2.0  # rango amplio, solo sanity check

    def test_high_quality_boosts_multiplier(self) -> None:
        state = {
            "vibe_factors": {
                "content": [{"type": "text", "text": "IC mean = 0.25, robust predictive factor."}]
            },
            "vibe_backtest": {
                "BTC/USDT": {
                    "content": [{"type": "text", "text": "Sharpe ratio = 1.9, excellent performance."}]
                }
            }
        }
        mult = compute_quality_multiplier(state, "BTC/USDT")
        assert mult > 1.0

    def test_low_quality_reduces_multiplier(self) -> None:
        state = {
            "vibe_factors": {
                "content": [{"type": "text", "text": "IC mean = -0.05, noise factor."}]
            },
            "vibe_backtest": {
                "BTC/USDT": {
                    "content": [{"type": "text", "text": "Sharpe ratio = 0.2, unprofitable."}]
                }
            }
        }
        mult = compute_quality_multiplier(state, "BTC/USDT")
        assert mult < 1.0

    def test_empty_returns_one(self, empty_state: dict) -> None:
        assert compute_quality_multiplier(empty_state, "BTC/USDT") == 1.0

    def test_extreme_high_caps_at_max(self) -> None:
        """Con calidad máxima (raw_score=1.0) el multiplier debe tocar VIBE_QUALITY_MAX."""
        from unittest.mock import patch
        import vibe.decision_bridge as _bridge
        state = {
            "vibe_factors": {"content": [{"type": "text", "text": "dummy"}]},
            "vibe_backtest": {"BTC/USDT": {"content": [{"type": "text", "text": "dummy"}]}},
        }
        with patch.object(_bridge, "_VIBE_QUALITY_MIN", 0.5), \
             patch.object(_bridge, "_VIBE_QUALITY_MAX", 1.5), \
             patch.object(_bridge, "parse_factor_quality", return_value=1.0), \
             patch.object(_bridge, "parse_backtest_quality", return_value=1.0):
            mult = _bridge.compute_quality_multiplier(state, "BTC/USDT")
            assert mult == pytest.approx(1.5, abs=0.01)

    def test_extreme_low_caps_at_min(self) -> None:
        """Con calidad mínima (raw_score=0.0) el multiplier debe tocar VIBE_QUALITY_MIN."""
        from unittest.mock import patch
        import vibe.decision_bridge as _bridge
        state = {
            "vibe_factors": {"content": [{"type": "text", "text": "dummy"}]},
            "vibe_backtest": {"BTC/USDT": {"content": [{"type": "text", "text": "dummy"}]}},
        }
        with patch.object(_bridge, "_VIBE_QUALITY_MIN", 0.5), \
             patch.object(_bridge, "_VIBE_QUALITY_MAX", 1.5), \
             patch.object(_bridge, "parse_factor_quality", return_value=0.0), \
             patch.object(_bridge, "parse_backtest_quality", return_value=0.0):
            mult = _bridge.compute_quality_multiplier(state, "BTC/USDT")
            assert mult == pytest.approx(0.5, abs=0.01)


class TestIsExtremeMacroVeto:
    def test_extreme_state_triggers_veto(self, extreme_veto_state: dict) -> None:
        assert is_extreme_macro_veto(extreme_veto_state) is True

    def test_single_signal_no_veto(self, bearish_pattern_state: dict) -> None:
        # Solo un pattern bajista no es suficiente (necesita >= 2 señales)
        assert is_extreme_macro_veto(bearish_pattern_state) is False

    def test_empty_no_veto(self, empty_state: dict) -> None:
        assert is_extreme_macro_veto(empty_state) is False

    def test_journal_only_no_veto(self) -> None:
        state = {
            "vibe_journal_analysis": {
                "content": [{"type": "text", "text": "Overtrading detected during losing streak."}]
            }
        }
        # Solo journal (1 señal) no alcanza threshold de 2
        assert is_extreme_macro_veto(state) is False

    def test_factor_and_journal_triggers_veto(self) -> None:
        state = {
            "vibe_factors": {
                "content": [{"type": "text", "text": "IC mean = -0.15, negative alpha."}]
            },
            "vibe_journal_analysis": {
                "content": [{"type": "text", "text": "Overtrading and revenge trading detected during losing streak."}]
            }
        }
        # Factor negativo + journal severo = 2 señales → veto
        assert is_extreme_macro_veto(state) is True

    def test_extreme_keywords_in_journal_alone_not_enough(self) -> None:
        state = {
            "vibe_journal_analysis": {
                "content": [{"type": "text", "text": "Extreme fear in trading decisions."}]
            }
        }
        assert is_extreme_macro_veto(state) is False
