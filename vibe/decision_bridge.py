"""
vibe.decision_bridge
~~~~~~~~~~~~~~~~~~~~

Traduce los resultados textuales de VIBE-Trading MCP en scores numéricos
que el bot puede usar para modular entradas, sizing y vetos.

Todas las funciones públicas son safe-call: si el estado de VIBE no está
poblado aún (o el cliente está caído) retornan valores neutrales para
no bloquear el trading.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger("clawdbot.vibe.bridge")

# ── Bounds leídos de env (Opción B Moderada) ──────────────────────────────────
_VIBE_QUALITY_MIN: float = float(os.environ.get("VIBE_QUALITY_MIN", "0.5").strip() or "0.5")
_VIBE_QUALITY_MAX: float = float(os.environ.get("VIBE_QUALITY_MAX", "1.5").strip() or "1.5")

# ── Keywords para parsing heurístico de textos VIBE ───────────────────────────
_BULLISH_PATTERN_KEYWORDS: set[str] = {
    "bullish", "ascending", "breakout", "higher low", "higher high",
    "golden cross", "cup and handle", "flag", "pennant", "wedge falling",
    "support", "accumulation", "demand zone", "buy signal", "long",
    " continuation", "uptrend", "momentum up",
}

_BEARISH_PATTERN_KEYWORDS: set[str] = {
    "bearish", "descending", "breakdown", "lower high", "lower low",
    "death cross", "head and shoulders", "double top", "triple top",
    "wedge rising", "resistance", "distribution", "supply zone",
    "sell signal", "short", "reversal", "downtrend", "momentum down",
    "divergence bearish", "overbought", "distribution",
}

_EXTREME_VETO_KEYWORDS: set[str] = {
    "extreme fear", "capitulation", "black swan", "liquidity crisis",
    "flash crash", "contagion", "systemic risk", "margin call cascade",
    "exchange insolvency", "regulatory ban", "stablecoin depeg",
}

_BULLISH_FACTOR_KEYWORDS: set[str] = {
    "strong predictive", "high ic", "positive alpha", "significant",
    "robust", "consistent",
}

_BEARISH_FACTOR_KEYWORDS: set[str] = {
    "weak predictive", "low ic", "negative alpha", "insignificant",
    "noise", "spurious", "decayed",
}

_JOURNAL_BIASES_SEVERE: set[str] = {
    "overtrading", "disposition effect", "revenge trading",
    "chasing momentum", "anchoring bias", "loss aversion extreme",
}

_JOURNAL_HEALTHY_KEYWORDS: set[str] = {
    "disciplined", "consistent", "edge confirmed", "risk control adequate",
    "profitable streak", "positive expectancy",
}


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _extract_text(data: Any) -> str:
    """Extrae texto plano de la respuesta típica del MCP de VIBE.

    El MCP suele devolver::

        {"content": [{"type": "text", "text": "..."}, ...]}

    o directamente un string.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data.lower()
    if isinstance(data, dict):
        parts: list[str] = []
        for item in data.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        text = " ".join(parts)
        return text.lower()
    return str(data).lower()


def _score_text(
    text: str,
    bullish_kw: set[str],
    bearish_kw: set[str],
) -> float:
    """Retorna un score en [-1.0, 1.0] basado en keyword matching.

    *  1.0 → fuertemente alcista
    *  0.0 → neutral
    * -1.0 → fuertemente bajista
    """
    if not text:
        return 0.0

    bull_count = sum(1 for kw in bullish_kw if kw in text)
    bear_count = sum(1 for kw in bearish_kw if kw in text)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _safe_float_from_text(text: str, pattern: str, default: float = 0.0) -> float:
    """Busca *pattern* en *text* y extrae el primer float numérico a su derecha."""
    if not text:
        return default
    m = re.search(pattern + r"[\s:=]+([-+]?\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


# ── Parsing individuales ───────────────────────────────────────────────────────

def parse_pattern_score(vibe_patterns: dict[str, Any] | None, symbol: str) -> float:
    """Score de patrón técnico para *symbol* en [-1.0, 1.0].

    *  1.0  → patrones fuertemente alcistas
    *  0.0  → sin datos o mixto
    * -1.0  → patrones fuertemente bajistas
    """
    if not vibe_patterns or symbol not in vibe_patterns:
        return 0.0
    text = _extract_text(vibe_patterns[symbol])
    return _score_text(text, _BULLISH_PATTERN_KEYWORDS, _BEARISH_PATTERN_KEYWORDS)


def parse_factor_quality(vibe_factors: Any | None) -> float:
    """Calidad del factor research en [0.0, 1.0].

    Basado en IC (Information Coefficient) cuando aparece en el texto.
    También considera keywords cualitativos.
    """
    if vibe_factors is None:
        return 0.5  # neutral cuando no hay datos

    text = _extract_text(vibe_factors)
    if not text:
        return 0.5

    # Intentar extraer IC numérico
    ic = _safe_float_from_text(text, r"ic(?:_mean)?", default=0.0)
    if ic != 0.0:
        # Normalizar IC típico [-0.3, 0.3] → [0, 1]
        normalized = (ic + 0.3) / 0.6
        return _clamp(normalized, 0.0, 1.0)

    # Fallback a keywords
    score = _score_text(text, _BULLISH_FACTOR_KEYWORDS, _BEARISH_FACTOR_KEYWORDS)
    # score ∈ [-1, 1] → mapear a [0, 1]
    return (score + 1.0) / 2.0


def parse_journal_bias(vibe_journal: Any | None) -> float:
    """Salud del journal en [0.0, 1.0].

    * 1.0 → trading disciplinado, sin sesgos graves
    * 0.0 → sesgos severos detectados (overtrading, revenge trading, etc.)
    """
    if vibe_journal is None:
        return 1.0  # sin journal aún = asumir saludable

    text = _extract_text(vibe_journal)
    if not text:
        return 1.0

    severe_count = sum(1 for kw in _JOURNAL_BIASES_SEVERE if kw in text)
    healthy_count = sum(1 for kw in _JOURNAL_HEALTHY_KEYWORDS if kw in text)

    if severe_count == 0 and healthy_count == 0:
        return 1.0

    # Penalizar fuertemente sesgos severos
    net = healthy_count - (severe_count * 2)
    # Normalizar aproximadamente a [0, 1]
    score = (net + 4) / 8.0
    return _clamp(score, 0.0, 1.0)


def parse_backtest_quality(
    vibe_backtest: dict[str, Any] | None, symbol: str
) -> float:
    """Calidad del backtest en [0.0, 1.0] basado en Sharpe ratio.

    * 1.0 → Sharpe >= 2.0 (excelente)
    * 0.5 → Sharpe ~ 1.0 (aceptable)
    * 0.0 → Sharpe <= 0 (malo)
    """
    if not vibe_backtest or symbol not in vibe_backtest:
        return 0.5  # neutral cuando no hay datos

    text = _extract_text(vibe_backtest[symbol])
    if not text:
        return 0.5

    sharpe = _safe_float_from_text(text, r"sharpe(?:[_\s]ratio)?", default=-999.0)
    if sharpe == -999.0:
        # Intentar buscar "sharpe" seguido de número sin label explícito
        m = re.search(r"sharpe[^\d]*?([-+]?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            try:
                sharpe = float(m.group(1))
            except ValueError:
                sharpe = -999.0

    if sharpe != -999.0:
        # Mapeo: Sharpe 0 → 0.0, Sharpe 1 → 0.5, Sharpe 2 → 1.0
        return _clamp((sharpe / 2.0), 0.0, 1.0)

    # Fallback a keywords
    if "profitable" in text or "positive" in text:
        return 0.7
    if "unprofitable" in text or "negative" in text or "loss" in text:
        return 0.3
    return 0.5


# ── Funciones públicas principales ─────────────────────────────────────────────

def compute_entry_score(state: dict[str, Any], symbol: str) -> float:
    """Score compuesto para autorizar una entrada en [0.0, 1.0].

    * 1.0 → todas las señales de VIBE son favorables
    * 0.0 → condiciones adversas claras (pero no veto extremo)

    Si VIBE no tiene datos retorna 1.0 (neutral, no bloquea).
    """
    vibe_patterns = state.get("vibe_patterns")
    vibe_factors = state.get("vibe_factors")
    vibe_journal = state.get("vibe_journal_analysis")
    vibe_backtest = state.get("vibe_backtest")

    # Si nada de VIBE está disponible → neutral
    if all(v is None or v == {} for v in (vibe_patterns, vibe_factors, vibe_journal, vibe_backtest)):
        return 1.0

    p_score = parse_pattern_score(vibe_patterns, symbol)          # [-1, 1]
    f_quality = parse_factor_quality(vibe_factors)                # [0, 1]
    j_health = parse_journal_bias(vibe_journal)                   # [0, 1]
    b_quality = parse_backtest_quality(vibe_backtest, symbol)     # [0, 1]

    # Normalizar pattern score de [-1, 1] a [0, 1]
    p_norm = (p_score + 1.0) / 2.0

    # Pesos: pattern 30%, factor 25%, journal 20%, backtest 25%
    raw = (p_norm * 0.30) + (f_quality * 0.25) + (j_health * 0.20) + (b_quality * 0.25)

    return _clamp(raw, 0.0, 1.0)


def compute_quality_multiplier(state: dict[str, Any], symbol: str) -> float:
    """Multiplicador de calidad para ajustar el sizing en [VIBE_QUALITY_MIN, VIBE_QUALITY_MAX].

    * 1.0 → calidad neutral (sin ajuste)
    * >1.0 → aumentar exposición (hasta VIBE_QUALITY_MAX)
    * <1.0 → reducir exposición (hasta VIBE_QUALITY_MIN)

    Basado principalmente en backtest Sharpe y factor IC.
    """
    vibe_factors = state.get("vibe_factors")
    vibe_backtest = state.get("vibe_backtest")

    if vibe_factors is None and (vibe_backtest is None or symbol not in (vibe_backtest or {})):
        return 1.0

    f_quality = parse_factor_quality(vibe_factors)                # [0, 1]
    b_quality = parse_backtest_quality(vibe_backtest, symbol)     # [0, 1]

    # Peso: 40% factor, 60% backtest
    raw_score = (f_quality * 0.40) + (b_quality * 0.60)

    # Mapear [0, 1] → [VIBE_QUALITY_MIN, VIBE_QUALITY_MAX]
    # 0.0 → MIN, 0.5 → 1.0, 1.0 → MAX
    multiplier = _VIBE_QUALITY_MIN + (raw_score * (_VIBE_QUALITY_MAX - _VIBE_QUALITY_MIN))

    return _clamp(multiplier, _VIBE_QUALITY_MIN, _VIBE_QUALITY_MAX)


def is_extreme_macro_veto(state: dict[str, Any]) -> bool:
    """Retorna *True* solo cuando múltiples señales de VIBE indican riesgo macro extremo.

    Condiciones que activan veto (requiere al menos 2 de 3):

    1. Shadow report o journal indica "rule violations críticas" en streak perdedora prolongada.
    2. Factor analysis muestra IC sostenidamente negativo (< -0.1).
    3. Pattern recognition detecta patrones de colapso/extreme fear en múltiples símbolos.
    """
    vibe_journal = state.get("vibe_journal_analysis")
    vibe_shadow = state.get("vibe_shadow_report")
    vibe_factors = state.get("vibe_factors")
    vibe_patterns = state.get("vibe_patterns", {})

    signals = 0

    # 1. Journal / Shadow: streak perdedora + sesgos severos
    journal_text = _extract_text(vibe_journal)
    shadow_text = _extract_text(vibe_shadow)
    combined_text = journal_text + " " + shadow_text

    if any(kw in combined_text for kw in _EXTREME_VETO_KEYWORDS):
        signals += 1
    if "losing streak" in combined_text and any(
        kw in combined_text for kw in _JOURNAL_BIASES_SEVERE
    ):
        signals += 1

    # 2. Factor IC negativo sostenido
    factor_text = _extract_text(vibe_factors)
    ic = _safe_float_from_text(factor_text, r"ic(?:[_\s]mean)?", default=0.0)
    if ic < -0.1:
        signals += 1
    elif "negative" in factor_text and "predictive" in factor_text:
        signals += 1

    # 3. Patrones extremos en múltiples símbolos (>=2)
    extreme_pattern_count = 0
    for sym, data in (vibe_patterns or {}).items():
        txt = _extract_text(data)
        if any(kw in txt for kw in _EXTREME_VETO_KEYWORDS):
            extreme_pattern_count += 1
    if extreme_pattern_count >= 2:
        signals += 1

    # Veto solo si hay al menos 2 señales concurrentes
    return signals >= 2


# ── Patrones críticos de reversión para Fase 3 (Exit Augmentation) ────────────

_CRITICAL_EXIT_PATTERNS: list[str] = [
    "harmonic bearish completed",
    "massive bearish divergence",
    "head and shoulders top",
    "double top confirmed",
    "triple top",
    "capitulation",
    "flash crash",
    "extreme fear",
    "distribution completed",
    "supply zone rejection",
]

_SUDDEN_IC_DROP_THRESHOLD: float = -0.15


def check_vibe_force_exit(state: dict[str, Any], symbol: str) -> str | None:
    """Retorna una razón de salida forzada si VIBE detecta un riesgo inminente.

    Evalúa ``vibe_patterns`` en busca de patrones de reversión críticos y
    ``vibe_factors`` para caídas súbitas del IC.  Si no hay peligro retorna
    ``None``.

    Parameters
    ----------
    state:
        El ``shared_state`` del bot.
    symbol:
        Símbolo a evaluar.

    Returns
    -------
    ``str`` con la razón (ej. ``"CRITICAL_BEARISH_PATTERN"``) o ``None``.
    """
    vibe_patterns = state.get("vibe_patterns", {})
    vibe_factors = state.get("vibe_factors")

    # 1. Patrones técnicos críticos
    if symbol in vibe_patterns:
        text = _extract_text(vibe_patterns[symbol])
        for pattern in _CRITICAL_EXIT_PATTERNS:
            if pattern in text:
                return f"CRITICAL_PATTERN_{pattern.replace(' ', '_').upper()}"

    # 2. Caída súbita del factor IC
    if vibe_factors is not None:
        factor_text = _extract_text(vibe_factors)
        ic = _safe_float_from_text(factor_text, r"ic(?:[_\s]mean)?", default=0.0)
        if ic < _SUDDEN_IC_DROP_THRESHOLD:
            return "SUDDEN_IC_DROP"

    return None
