"""
strategy.ml_predictor
~~~~~~~~~~~~~~~~~~~~~~

XGBoost-based predictor for the probability of a BTC price increase in the
next 5 price ticks (order-book snapshots).

Signal generation rules
-----------------------
* probability > 0.65 AND sentiment_score > 0.3   → **BUY**
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
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Signal thresholds
_BUY_PROB_THRESHOLD = 0.65
_BUY_SENTIMENT_THRESHOLD = 0.3
_SELL_PROB_THRESHOLD = 0.3
_SELL_SENTIMENT_THRESHOLD = -0.3

# Default prediction horizon (price-tick steps)
_PREDICTION_HORIZON = 5

# Technical indicator parameters
_SMA_PERIOD = 20
_RSI_PERIOD = 14
_MOMENTUM_PERIOD = 5

# Minimum prices needed for inference (SMA_20 requires 20 observations)
_MIN_PRICES_FOR_INFERENCE = _SMA_PERIOD

# Feature column order expected by the model
_FEATURE_COLS = ["sentiment", "rsi", "sma_ratio", "volatility", "momentum"]

Signal = str  # literal: "BUY" | "SELL" | "HOLD"


class MLPredictor:
    """Predicts price-direction probability and generates trading signals.

    The underlying model is an XGBoost binary classifier trained on a
    five-element feature vector: [sentiment, rsi, sma_ratio, volatility,
    momentum].
    """

    def __init__(self) -> None:
        self._model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        )
        self._is_trained = False
        logger.info("MLPredictor initialised (XGBoost).")

    @property
    def is_trained(self) -> bool:
        """Return *True* when the model has been fitted or loaded."""
        return self._is_trained

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_features(self, prices: list[float], sentiment: float) -> list[float]:
        """Compute technical indicators and return a feature vector.

        Parameters
        ----------
        prices:    Recent mid-prices in chronological order.
        sentiment: Current VADER compound score in [-1, +1].

        Returns
        -------
        A five-element list ``[sentiment, rsi, sma_ratio, volatility, momentum]``
        where NaN values are replaced with ``0.0``.
        """
        series = pd.Series(prices, dtype=float)

        # SMA_20 ratio: current_price / SMA_20 (normalised)
        sma_20 = series.rolling(_SMA_PERIOD).mean()
        sma_20_val = float(sma_20.iloc[-1])
        current_price = float(series.iloc[-1])
        if np.isnan(sma_20_val) or sma_20_val == 0.0:
            sma_ratio = 0.0
        else:
            sma_ratio = current_price / sma_20_val

        # Volatility: rolling standard deviation of the last 20 periods
        vol_series = series.rolling(_SMA_PERIOD).std()
        vol_val = float(vol_series.iloc[-1])
        volatility = 0.0 if np.isnan(vol_val) else vol_val

        # RSI_14: Relative Strength Index using Wilder's exponential smoothing.
        # When avg_loss=0 and avg_gain>0, RS=inf → RSI=100 (pure uptrend).
        # When both are 0 (no movement), RS=NaN → fill with 50 (neutral).
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=_RSI_PERIOD - 1, min_periods=_RSI_PERIOD).mean()
        avg_loss = loss.ewm(com=_RSI_PERIOD - 1, min_periods=_RSI_PERIOD).mean()
        rs = avg_gain / avg_loss
        rsi_series = (100 - (100 / (1 + rs))).fillna(50.0)
        rsi = float(rsi_series.iloc[-1])

        # Momentum: percentage change of current price vs 5 periods ago
        mom_series = series.pct_change(_MOMENTUM_PERIOD) * 100
        mom_val = float(mom_series.iloc[-1])
        momentum = 0.0 if np.isnan(mom_val) else mom_val

        return [float(sentiment), rsi, sma_ratio, volatility, momentum]

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
        series = pd.Series(prices, dtype=float)

        # Compute all indicators across the full price history
        sma_20 = series.rolling(_SMA_PERIOD).mean()
        vol_series = series.rolling(_SMA_PERIOD).std()

        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=_RSI_PERIOD - 1, min_periods=_RSI_PERIOD).mean()
        avg_loss = loss.ewm(com=_RSI_PERIOD - 1, min_periods=_RSI_PERIOD).mean()
        rs = avg_gain / avg_loss
        rsi_series = (100 - (100 / (1 + rs))).fillna(50.0)

        mom_series = series.pct_change(_MOMENTUM_PERIOD) * 100
        sma_ratio_series = series / sma_20

        sentiment_series = (
            pd.Series(sentiment_scores, dtype=float)
            if sentiment_scores is not None
            else pd.Series(0.0, index=series.index)
        )

        df = pd.DataFrame(
            {
                "sentiment": sentiment_series,
                "rsi": rsi_series,
                "sma_ratio": sma_ratio_series,
                "volatility": vol_series,
                "momentum": mom_series,
                "price": series,
            }
        )

        # Fill NaN from rolling warm-up with 0
        df[["sma_ratio", "volatility", "momentum"]] = df[
            ["sma_ratio", "volatility", "momentum"]
        ].fillna(0.0)

        # Label: 1 if price rises after `horizon` ticks, else 0
        df["label"] = (series.shift(-horizon) > series).astype(int)

        # Drop rows without a future label (last `horizon` rows)
        df = df.dropna(subset=["label"])

        X = df[_FEATURE_COLS]
        y = df["label"]

        if len(X) < 10:
            logger.warning(
                "warm_start: not enough labelled samples (%d). Model not trained.",
                len(X),
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
    # Model persistence
    # ------------------------------------------------------------------

    def load_model(self, filepath: str | Path) -> bool:
        """Load a previously saved XGBoost model from *filepath*.

        The model file must have been created by XGBoost's native
        ``save_model`` (JSON or binary format).

        Parameters
        ----------
        filepath: Path to the saved model file (e.g. ``models/xgb_live.json``).

        Returns
        -------
        ``True`` on success, ``False`` if the file does not exist or cannot
        be loaded.
        """
        path = Path(filepath)
        if not path.exists():
            logger.info("load_model: file not found at %s.", path)
            return False
        try:
            self._model.load_model(str(path))
            self._is_trained = True
            logger.info("MLPredictor loaded pre-trained model from %s.", path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_model failed (%s): %s", path, exc)
            return False

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
                         At least ``_MIN_PRICES_FOR_INFERENCE`` (20) values are
                         required.
        sentiment_score: Current VADER compound score in [-1, +1].

        Returns
        -------
        A float in [0, 1] representing the predicted probability, or
        ``None`` when the model is untrained or there is insufficient history.
        """
        if not self._is_trained:
            logger.debug("predict_proba called before model is trained.")
            return None

        if len(prices) < _MIN_PRICES_FOR_INFERENCE:
            logger.debug(
                "predict_proba: not enough prices (%d < %d).",
                len(prices),
                _MIN_PRICES_FOR_INFERENCE,
            )
            return None

        features = self._compute_features(prices, sentiment_score)
        X = pd.DataFrame([features], columns=_FEATURE_COLS)
        proba: float = float(self._model.predict_proba(X)[0][1])

        rsi = features[1]
        volatility = features[3]
        momentum = features[4]
        logger.info(
            "[INDICATORS] RSI: %.1f | Volatility: %.1f | Momentum: %.1f%%",
            rsi,
            volatility,
            momentum,
        )
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

        * probability > 0.65 AND sentiment > 0.3   → BUY
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
