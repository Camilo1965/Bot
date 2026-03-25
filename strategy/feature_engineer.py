"""
strategy.feature_engineer
~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates technical indicators (RSI, Moving Averages) from a price history
and combines them with a sentiment score to build ML feature vectors.

Usage
-----
    from strategy.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    features = fe.build_features(prices=[...], sentiment_score=0.35)
    # → {"rsi": 58.2, "sma_short": 42100.0, ..., "sentiment_score": 0.35}

    X, y = fe.build_feature_matrix(prices=[...])
    # → (DataFrame of features, Series of binary labels)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Look-back window sizes
_RSI_PERIOD = 14
_SMA_SHORT = 20
_SMA_LONG = 50
_EMA_SHORT = 20

# ATR period
ATR_PERIOD = 14

# Minimum number of price observations required for a feature vector
MIN_PRICES = _SMA_LONG


class FeatureEngineer:
    """Builds ML feature vectors from price history and sentiment scores."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
        """Return RSI values for *prices* using Wilder's exponential smoothing."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def compute_atr(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = ATR_PERIOD,
    ) -> float | None:
        """Compute the Average True Range (ATR) for the latest bar.

        Uses Wilder's smoothing (RMA) over *period* bars.  Requires at least
        ``period + 1`` values in each of *highs*, *lows*, and *closes* (the
        extra value is needed to compute the previous-close component of the
        first True Range).

        Parameters
        ----------
        highs:  Per-bar high prices in chronological order.
        lows:   Per-bar low prices in chronological order.
        closes: Per-bar close prices in chronological order.
        period: Smoothing period (default 14, as in ATR_14).

        Returns
        -------
        The latest ATR value as a float, or ``None`` when there is
        insufficient data.
        """
        n = min(len(highs), len(lows), len(closes))
        if n < period + 1:
            logger.debug(
                "compute_atr: insufficient data (%d bars, need %d).",
                n,
                period + 1,
            )
            return None

        h = highs[-n:]
        lw = lows[-n:]
        c = closes[-n:]

        # Compute True Range for each bar starting at index 1
        trs: list[float] = []
        for i in range(1, n):
            tr = max(
                h[i] - lw[i],
                abs(h[i] - c[i - 1]),
                abs(lw[i] - c[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return None

        # Seed the ATR with the simple mean of the first `period` TRs
        atr = sum(trs[:period]) / period
        # Apply Wilder's smoothing for remaining TRs
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period

        return atr

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_features(
        self,
        prices: list[float],
        sentiment_score: float = 0.0,
    ) -> dict[str, float] | None:
        """Return a feature dict for the *latest* observation.

        Parameters
        ----------
        prices:          Historical mid-prices in chronological order.
                         Requires at least ``MIN_PRICES`` (50) values.
        sentiment_score: VADER compound score in [-1, +1].

        Returns
        -------
        A dict of named feature values, or ``None`` when there are too
        few prices to compute all indicators.
        """
        if len(prices) < MIN_PRICES:
            logger.debug(
                "Not enough price data (%d < %d) to compute features.",
                len(prices),
                MIN_PRICES,
            )
            return None

        series = pd.Series(prices, dtype=float)
        rsi = self._compute_rsi(series)
        sma_short = series.rolling(_SMA_SHORT).mean()
        sma_long = series.rolling(_SMA_LONG).mean()
        ema_short = series.ewm(span=_EMA_SHORT, adjust=False).mean()

        latest_price = float(series.iloc[-1])
        sma_short_val = float(sma_short.iloc[-1])
        sma_long_val = float(sma_long.iloc[-1])

        features: dict[str, float] = {
            "rsi": float(rsi.iloc[-1]),
            "sma_short": sma_short_val,
            "sma_long": sma_long_val,
            "ema_short": float(ema_short.iloc[-1]),
            "price_sma_short_ratio": latest_price / sma_short_val if sma_short_val else 1.0,
            "price_sma_long_ratio": latest_price / sma_long_val if sma_long_val else 1.0,
            "sma_ratio": sma_short_val / sma_long_val if sma_long_val else 1.0,
            "sentiment_score": float(sentiment_score),
        }
        logger.debug("Features computed: %s", features)
        return features

    def build_feature_matrix(
        self,
        prices: list[float],
        sentiment_scores: list[float] | None = None,
        horizon: int = 5,
    ) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
        """Build a labelled feature matrix for training.

        Each row is one time step; the label is ``1`` if the price
        *horizon* steps ahead is higher than the current price, else ``0``.

        Parameters
        ----------
        prices:           Historical mid-prices in chronological order.
        sentiment_scores: Optional per-tick sentiment scores (same length as
                          *prices*).  Defaults to ``0.0`` for all ticks.
        horizon:          Number of ticks ahead to use as the prediction target.

        Returns
        -------
        ``(X, y)`` DataFrames ready for XGBoost training, or ``(None, None)``
        when there is insufficient data.
        """
        min_required = MIN_PRICES + horizon
        if len(prices) < min_required:
            logger.debug(
                "Not enough price data (%d < %d) to build feature matrix.",
                len(prices),
                min_required,
            )
            return None, None

        series = pd.Series(prices, dtype=float)
        rsi = self._compute_rsi(series)
        sma_short = series.rolling(_SMA_SHORT).mean()
        sma_long = series.rolling(_SMA_LONG).mean()
        ema_short = series.ewm(span=_EMA_SHORT, adjust=False).mean()

        sentiment = (
            pd.Series(sentiment_scores, dtype=float)
            if sentiment_scores is not None
            else pd.Series(0.0, index=series.index)
        )

        df = pd.DataFrame(
            {
                "price": series,
                "rsi": rsi,
                "sma_short": sma_short,
                "sma_long": sma_long,
                "ema_short": ema_short,
                "price_sma_short_ratio": series / sma_short,
                "price_sma_long_ratio": series / sma_long,
                "sma_ratio": sma_short / sma_long,
                "sentiment_score": sentiment,
            }
        )

        # Label: 1 if price rises after `horizon` ticks
        df["label"] = (series.shift(-horizon) > series).astype(int)

        # Drop rows with NaN values (rolling window warm-up) or missing future labels
        df = df.dropna()

        feature_cols = [c for c in df.columns if c not in ("price", "label")]
        X = df[feature_cols]
        y = df["label"]
        return X, y
