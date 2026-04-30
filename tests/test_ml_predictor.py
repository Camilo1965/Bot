"""Regression tests for :class:`~strategy.ml_predictor.MLPredictor.generate_signal`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strategy.ml_predictor import (
    HTF_TREND_BEARISH,
    HTF_TREND_BULLISH,
    HTF_TREND_NEUTRAL,
    MLPredictor,
)


@pytest.fixture
def prices_25() -> list[float]:
    """Enough bars for inference (>= 20) with mild uptrend."""
    return [100.0 + i * 0.05 for i in range(25)]


def test_generate_signal_hold_when_model_not_ready(prices_25: list[float]) -> None:
    pred = MLPredictor()
    assert pred.generate_signal(prices_25) == "HOLD"


def test_generate_signal_buy_base_case(prices_25: list[float]) -> None:
    pred = MLPredictor()
    pred.predict_proba = MagicMock(return_value=0.65)
    sig = pred.generate_signal(
        prices_25,
        sentiment_score=0.2,
        htf_trend_4h=HTF_TREND_BULLISH,
        htf_trend_1h=HTF_TREND_BULLISH,
    )
    assert sig == "BUY"


def test_generate_signal_buy_muted_when_4h_bearish(prices_25: list[float]) -> None:
    pred = MLPredictor()
    pred.predict_proba = MagicMock(return_value=0.65)
    sig = pred.generate_signal(
        prices_25,
        sentiment_score=0.2,
        htf_trend_4h=HTF_TREND_BEARISH,
        htf_trend_1h=HTF_TREND_BULLISH,
    )
    assert sig == "HOLD"


def test_generate_signal_buy_muted_when_1h_not_bullish(prices_25: list[float]) -> None:
    pred = MLPredictor()
    pred.predict_proba = MagicMock(return_value=0.65)
    sig = pred.generate_signal(
        prices_25,
        sentiment_score=0.2,
        htf_trend_4h=HTF_TREND_BULLISH,
        htf_trend_1h=HTF_TREND_NEUTRAL,
    )
    assert sig == "HOLD"


def test_generate_signal_sell_base_case(prices_25: list[float]) -> None:
    pred = MLPredictor()
    pred.predict_proba = MagicMock(return_value=0.25)
    sig = pred.generate_signal(prices_25, sentiment_score=-0.5)
    assert sig == "SELL"


def test_generate_signal_buy_penalised_by_extreme_greed_funding(prices_25: list[float]) -> None:
    pred = MLPredictor()
    pred.predict_proba = MagicMock(return_value=0.65)
    sig = pred.generate_signal(
        prices_25,
        sentiment_score=0.2,
        funding_rate=0.001,
        htf_trend_4h=HTF_TREND_BULLISH,
        htf_trend_1h=HTF_TREND_BULLISH,
    )
    assert sig == "HOLD"
