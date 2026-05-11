"""
vibe.feature_bridge
~~~~~~~~~~~~~~~~~~~

Convierte insights textuales de VIBE-Trading en features numéricas que
pueden alimentar el modelo XGBoost de ClawdBot (Fase 4).

REGLA DE ORO: Si el state está vacío o VIBE está caído, retorna valores
neutrales [0.0, 0.0, 1.0, 0.5] para evitar NaN que crasheen el booster.
"""

from __future__ import annotations

from typing import Any

from vibe.decision_bridge import (
    parse_backtest_quality,
    parse_factor_quality,
    parse_journal_bias,
    parse_pattern_score,
)

# Valores neutrales cuando VIBE no está disponible:
# [pattern_score, factor_ic, journal_health, backtest_sharpe]
VIBE_FEATURE_NEUTRAL: list[float] = [0.0, 0.0, 1.0, 0.5]


def extract_vibe_features(state: dict[str, Any] | None, symbol: str) -> list[float]:
    """Retorna un vector de 4 features numéricas basado en el estado de VIBE.

    Parameters
    ----------
    state:
        ``shared_state`` del bot.  Puede ser ``None`` o vacío.
    symbol:
        Símbolo a evaluar.

    Returns
    -------
    ``[pattern_score, factor_ic, journal_health, backtest_sharpe]``
    con valores en rangos seguros para XGBoost.
    """
    if state is None:
        return list(VIBE_FEATURE_NEUTRAL)

    vibe_patterns = state.get("vibe_patterns")
    vibe_factors = state.get("vibe_factors")
    vibe_journal = state.get("vibe_journal_analysis")
    vibe_backtest = state.get("vibe_backtest")

    # Si absolutamente nada de VIBE está poblado → neutrales
    if all(v is None or v == {} for v in (vibe_patterns, vibe_factors, vibe_journal, vibe_backtest)):
        return list(VIBE_FEATURE_NEUTRAL)

    p_score = parse_pattern_score(vibe_patterns, symbol)          # [-1, 1]
    f_quality = parse_factor_quality(vibe_factors)                # [0, 1]
    j_health = parse_journal_bias(vibe_journal)                   # [0, 1]
    b_quality = parse_backtest_quality(vibe_backtest, symbol)     # [0, 1]

    # Normalizar pattern score de [-1, 1] a [0, 1]
    p_norm = (p_score + 1.0) / 2.0

    # factor_ic: mapear IC calidad [0,1] a un rango más informativo [-0.5, 0.5]
    # para que el modelo vea dirección (positivo/negativo) del factor
    factor_ic = (f_quality - 0.5)  # [-0.5, 0.5]

    # journal_health ya está en [0, 1]
    journal_health = j_health

    # backtest_sharpe: mapear calidad [0,1] a [-1, 1] para dirección
    backtest_sharpe = (b_quality - 0.5) * 2.0  # [-1, 1]

    return [
        round(p_norm, 4),
        round(factor_ic, 4),
        round(journal_health, 4),
        round(backtest_sharpe, 4),
    ]
