"""
strategy.ml_predictor
~~~~~~~~~~~~~~~~~~~~~~

XGBoost-based predictor for the probability of a BTC price increase in the
next 5 price ticks (order-book snapshots).

Signal generation rules
-----------------------
* probability > 0.7 AND sentiment_score > 0.3   → **BUY**
* probability < 0.3 AND sentiment_score < -0.3  → **SELL**
* otherwise                                      → **HOLD**

The model supports "warm-starting" from historical price data already stored
in TimescaleDB via :meth:`MLPredictor.warm_start`.

Usage
-----
    from strategy.ml_predictor import MLPredictor

    predictor = MLPredictor()
    predictor.warm_start(prices=[...])
    signal = predictor.generate_signal(prices=[...], sentiment_score=0.4)
    # → "BUY"
"""

from __future__ import annotations

import logging

import pandas as pd
from xgboost import XGBClassifier

from strategy.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

# Signal thresholds
_BUY_PROB_THRESHOLD = 0.7
_BUY_SENTIMENT_THRESHOLD = 0.3
_SELL_PROB_THRESHOLD = 0.3
_SELL_SENTIMENT_THRESHOLD = -0.3

# Default prediction horizon (price-tick steps)
_PREDICTION_HORIZON = 5

Signal = str  # literal: "BUY" | "SELL" | "HOLD"


class MLPredictor:
    """Predicts price-direction probability and generates trading signals.

    The underlying model is an XGBoost binary classifier trained on features
    produced by :class:`~strategy.feature_engineer.FeatureEngineer`.
    """

    def __init__(self) -> None:
        self._model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self._engineer = FeatureEngineer()
        self._is_trained = False
        logger.info("MLPredictor initialised (XGBoost).")

    # ------------------------------------------------------------------
    # Training / warm-start
    # ------------------------------------------------------------------

    def warm_start(
        self,
        prices: list[float],
        sentiment_scores: list[float] | None = None,
        horizon: int = _PREDICTION_HORIZON,
    ) -> bool:
        """Train (or re-train) the model from historical price data.

        This method is idempotent – calling it multiple times is safe and will
        simply replace the previously fitted model.

        Parameters
        ----------
        prices:           Historical mid-prices in chronological order.
        sentiment_scores: Optional per-tick sentiment scores (same length as
                          *prices*).  Defaults to ``0.0`` for all ticks.
        horizon:          Number of ticks ahead to use as the prediction target.

        Returns
        -------
        ``True`` on success, ``False`` when there is insufficient data to
        produce at least 10 labelled training samples.
        """
        X, y = self._engineer.build_feature_matrix(prices, sentiment_scores, horizon)
        if X is None or len(X) < 10:
            logger.warning(
                "warm_start: not enough labelled samples (%s). Model not trained.",
                0 if X is None else len(X),
            )
            return False

        self._model.fit(X, y)
        self._is_trained = True
        logger.info(
            "MLPredictor warm-started on %d samples (%d features).",
            len(X),
            X.shape[1],
        )
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        prices: list[float],
        sentiment_score: float = 0.0,
    ) -> float | None:
        """Return the probability of an upward price move.

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
                         At least 50 values are required.
        sentiment_score: Current VADER compound score in [-1, +1].

        Returns
        -------
        A float in [0, 1] representing the predicted probability, or
        ``None`` when the model is untrained or there is insufficient history.
        """
        if not self._is_trained:
            logger.debug("predict_proba called before model is trained.")
            return None

        features = self._engineer.build_features(prices, sentiment_score)
        if features is None:
            return None

        X = pd.DataFrame([features])
        proba: float = float(self._model.predict_proba(X)[0][1])
        logger.debug("Predicted probability=%.4f", proba)
        return proba

    def generate_signal(
        self,
        prices: list[float],
        sentiment_score: float = 0.0,
    ) -> Signal:
        """Generate a trading signal (BUY / SELL / HOLD).

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
        sentiment_score: Current VADER compound score in [-1, +1].

        Returns
        -------
        ``"BUY"``, ``"SELL"``, or ``"HOLD"`` according to the rules:

        * probability > 0.7 AND sentiment > 0.3   → BUY
        * probability < 0.3 AND sentiment < -0.3  → SELL
        * otherwise                               → HOLD
        """
        probability = self.predict_proba(prices, sentiment_score)

        if probability is None:
            logger.info("Signal=HOLD (model not ready or insufficient data)")
            return "HOLD"

        if probability > _BUY_PROB_THRESHOLD and sentiment_score > _BUY_SENTIMENT_THRESHOLD:
            signal: Signal = "BUY"
        elif probability < _SELL_PROB_THRESHOLD and sentiment_score < _SELL_SENTIMENT_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        logger.info(
            "Signal=%s  probability=%.4f  sentiment=%.4f",
            signal,
            probability,
            sentiment_score,
        )
        return signal
