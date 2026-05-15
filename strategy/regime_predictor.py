"""
strategy.regime_predictor
~~~~~~~~~~~~~~~~~~~~~~~~~

Predicts market regime (TRENDING vs RANGING) using pre-trained XGBoost models.

Regime prediction is much more accurate (AUC ~0.85-0.99) than direction
prediction (AUC ~0.52), so we use it as the primary trading filter:
- TRENDING → allow entries in direction of trend
- RANGING → block new entries, close existing positions
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from strategy.quant_features import MIN_OHLC_ROWS, QUANT_FEATURE_COLS, compute_quant_vector_from_lists

logger = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
_REGIME_CACHE: dict[str, XGBClassifier] = {}

# Probability threshold to consider market as trending
TRENDING_PROB_THRESHOLD: float = float(os.environ.get("REGIME_TRENDING_THRESHOLD", "0.65"))


def load_regime_model(symbol: str) -> XGBClassifier | None:
    """Load regime XGB model from disk (cached)."""
    key = symbol
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]
    path = MODELS_DIR / f"{symbol.replace('/', '_')}_regime.json"
    if not path.is_file():
        logger.debug("No regime model found at %s", path)
        return None
    try:
        model = XGBClassifier()
        model.load_model(str(path))
        _REGIME_CACHE[key] = model
        logger.info("[REGIME] Loaded model for %s from %s", symbol, path)
        return model
    except Exception as exc:
        logger.warning("[REGIME] Failed to load model for %s: %s", symbol, exc)
        return None


def predict_regime(
    symbol: str,
    prices: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> tuple[str, float] | None:
    """Predict market regime for *symbol*.

    Returns ``("TRENDING", probability)`` or ``("RANGING", probability)``,
    or ``None`` if model unavailable or insufficient data.
    """
    model = load_regime_model(symbol)
    if model is None:
        return None

    vec = compute_quant_vector_from_lists(prices, highs or prices, lows or prices, volumes)
    if vec is None:
        return None

    try:
        X = pd.DataFrame([vec], columns=QUANT_FEATURE_COLS)
        proba: float = float(model.predict_proba(X)[0][1])
    except Exception as exc:
        logger.warning("[REGIME] Inference failed for %s: %s", symbol, exc)
        return None

    regime = "TRENDING" if proba >= TRENDING_PROB_THRESHOLD else "RANGING"
    return regime, round(proba, 4)
