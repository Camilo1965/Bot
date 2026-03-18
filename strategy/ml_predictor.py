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

# [ELITE] ADX thresholds for market-regime detection
_ADX_RANGING_THRESHOLD = 20   # ADX < 20 → ranging market (reduce momentum weight)
_ADX_TRENDING_THRESHOLD = 25  # ADX > 25 → trending market (increase RSI weight)
_ADX_PERIOD = 14

# [ELITE] Funding-rate penalty: large positive funding → market overheated (penalise BUY)
_FUNDING_RATE_GREED_THRESHOLD = 0.001   # 0.1% per 8h – above this, apply greed penalty
_FUNDING_RATE_PENALTY = 0.05            # subtract from probability when greedy

# [ELITE] OBI scaling: each unit of imbalance shifts raw probability by this amount
_OBI_ADJUSTMENT_FACTOR = 0.05          # e.g. OBI=1.0 → probability shifts ±0.05

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

    def _compute_adx(self, prices: list[float]) -> float:
        """[ELITE] Compute the Average Directional Index (ADX) for *prices*.

        Uses Wilder's smoothing over ``_ADX_PERIOD`` bars.  Returns ``0.0``
        when there is insufficient history.
        """
        if len(prices) < _ADX_PERIOD * 2:
            # Wilder's ADX needs at least 2 × _ADX_PERIOD bars: one period for the
            # directional movement and a second period for the ADX smoothing pass.
            return 0.0

        close = pd.Series(prices, dtype=float)

        # True Range (using close-only proxy: TR = abs(close_t - close_t-1))
        tr = close.diff().abs()

        # Directional Movement (close-only proxy)
        dm_plus = close.diff().clip(lower=0)
        dm_minus = (-close.diff()).clip(lower=0)

        # Wilder smoothing
        atr = tr.ewm(alpha=1 / _ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean()
        smooth_plus = dm_plus.ewm(alpha=1 / _ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean()
        smooth_minus = dm_minus.ewm(alpha=1 / _ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean()

        di_plus = (smooth_plus / atr.replace(0, np.nan)) * 100
        di_minus = (smooth_minus / atr.replace(0, np.nan)) * 100

        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)) * 100
        adx_series = dx.ewm(alpha=1 / _ADX_PERIOD, min_periods=_ADX_PERIOD, adjust=False).mean()

        adx_val = float(adx_series.iloc[-1])
        return 0.0 if np.isnan(adx_val) else adx_val

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

        # [ELITE] ADX-based market-regime adjustment
        adx = self._compute_adx(prices)
        if adx > 0:
            if adx < _ADX_RANGING_THRESHOLD:
                # Ranging market – reduce momentum influence
                momentum *= 0.5
                logger.debug(
                    "[ELITE] ADX=%.1f (Ranging) – Momentum weight reduced to 50%%.", adx
                )
            elif adx > _ADX_TRENDING_THRESHOLD:
                # Trending market – amplify RSI contribution
                rsi_distance = rsi - 50.0  # signed distance from neutral
                rsi = 50.0 + rsi_distance * 1.25
                logger.debug(
                    "[ELITE] ADX=%.1f (Trending) – RSI weight increased by 25%%.", adx
                )

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

    def save_model(self, filepath: str | Path) -> bool:
        """Save the current XGBoost model to *filepath* in JSON format.

        Parameters
        ----------
        filepath: Destination path (e.g. ``models/xgb_live.json``).

        Returns
        -------
        ``True`` on success, ``False`` if the model is untrained or an error
        occurs.
        """
        if not self._is_trained:
            logger.warning("save_model: model is not trained – nothing to save.")
            return False
        path = Path(filepath)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save_model(str(path))
            logger.info("[PRO] Model saved to %s.", path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PRO] save_model failed (%s): %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        prices: list[float],
        sentiment_score: float = 0.0,
        obi_ratio: float = 0.0,
        funding_rate: float = 0.0,
    ) -> float | None:
        """Return the probability of an upward price move.

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
                         At least ``_MIN_PRICES_FOR_INFERENCE`` (20) values are
                         required.
        sentiment_score: Current VADER compound score in [-1, +1].
        obi_ratio:       [ELITE] Order Book Imbalance in [-1, +1].
                         Positive values indicate more bid volume (buying pressure).
        funding_rate:    [ELITE] Current perpetual funding rate.
                         A high positive rate signals market overheating (greed penalty).

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

        # [ELITE] OBI adjustment: shift probability toward buy/sell pressure
        if obi_ratio != 0.0:
            proba = max(0.0, min(1.0, proba + obi_ratio * _OBI_ADJUSTMENT_FACTOR))
            logger.debug("[ELITE] OBI=%.4f applied – adjusted proba=%.4f", obi_ratio, proba)

        # [ELITE] Funding rate Greed/Fear penalty
        if funding_rate > _FUNDING_RATE_GREED_THRESHOLD:
            proba = max(0.0, proba - _FUNDING_RATE_PENALTY)
            logger.info(
                "[ELITE] Funding rate=%.5f (Greed) – BUY probability penalised by %.2f.",
                funding_rate,
                _FUNDING_RATE_PENALTY,
            )

        adx = self._compute_adx(prices)
        rsi = features[1]
        volatility = features[3]
        momentum = features[4]
        logger.info(
            "[INDICATORS] RSI: %.1f | Volatility: %.1f | Momentum: %.1f%% | ADX: %.1f",
            rsi,
            volatility,
            momentum,
            adx,
        )
        logger.debug("Predicted probability=%.4f", proba)
        return proba

    def generate_signal(
        self,
        prices: list[float],
        sentiment_score: float = 0.0,
        obi_ratio: float = 0.0,
        funding_rate: float = 0.0,
    ) -> Signal:
        """Generate a trading signal (BUY / SELL / HOLD).

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
        sentiment_score: Current VADER compound score in [-1, +1].
        obi_ratio:       [ELITE] Order Book Imbalance in [-1, +1].
        funding_rate:    [ELITE] Current perpetual funding rate.

        Returns
        -------
        ``"BUY"``, ``"SELL"``, or ``"HOLD"`` according to the rules:

        * probability > 0.65 AND sentiment > 0.3   → BUY
        * probability < 0.3 AND sentiment < -0.3  → SELL
        * otherwise                               → HOLD
        """
        probability = self.predict_proba(prices, sentiment_score, obi_ratio, funding_rate)

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
            "Signal=%s  probability=%.4f  sentiment=%.4f  obi=%.4f  funding=%.6f",
            signal,
            probability,
            sentiment_score,
            obi_ratio,
            funding_rate,
        )
        return signal
