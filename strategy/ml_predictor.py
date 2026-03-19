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

Elite Quant additions
----------------------
* **Order Book Imbalance (OBI)**: ``obi_ratio`` = total bid volume / total ask
  volume across the top-5 depth levels.  Values > 1 indicate buy-side
  pressure; values < 1 indicate sell-side pressure.

* **Market Regime Classifier** (ADX-based):

  - ADX > 25 (trending market): Trend-Following mode – BUY requires RSI > 50
    AND positive momentum; SELL requires RSI < 50 AND negative momentum.
  - ADX < 20 (range-bound market): Mean-Reversion mode – BUY triggered by RSI
    oversold (< 35); SELL triggered by RSI overbought (> 65).
  - 20 ≤ ADX ≤ 25: standard ML signal is used without regime adjustment.

* **Funding Rate Bias**:

  - lastFundingRate > +0.0003 (> +0.03 %, extreme greed): Short-Bias penalty –
    a BUY signal is demoted to HOLD.
  - lastFundingRate < -0.0003 (< -0.03 %, extreme fear): Long-Bias bonus – the
    BUY signal is retained and flagged as ``[ELITE]``.

When any of the Elite factors influences the final signal, the log line is
prefixed with ``[ELITE]``.

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
_ADX_PERIOD = 14
_ATR_PERIOD = 14

# Market Regime thresholds (ADX-based)
_ADX_TREND_THRESHOLD = 25.0    # above → trending market (trend-following mode)
_ADX_RANGE_THRESHOLD = 20.0    # below → range-bound market (mean-reversion mode)
_RSI_OVERSOLD = 35.0           # mean-reversion BUY trigger
_RSI_OVERBOUGHT = 65.0         # mean-reversion SELL trigger

# Funding rate bias thresholds (decimal form: 0.0003 = 0.03 %)
_FUNDING_RATE_EXTREME_GREED = 0.0003   # > this → Short-Bias penalty on BUY
_FUNDING_RATE_EXTREME_FEAR = -0.0003   # < this → Long-Bias bonus on BUY

# Minimum prices needed for inference (SMA_20 requires 20 observations)
_MIN_PRICES_FOR_INFERENCE = _SMA_PERIOD

# Default ADX value used when OHLCV data is unavailable (neutral – no regime)
_ADX_NEUTRAL_DEFAULT = 25.0

# HTF trend labels
HTF_TREND_BULLISH = "bullish"
HTF_TREND_BEARISH = "bearish"
HTF_TREND_NEUTRAL = "neutral"

# Minimum number of HTF candles required to compute a trend
_HTF_MIN_CANDLES = 3

# Feature column order expected by the model
_FEATURE_COLS = [
    "sentiment",
    "rsi",
    "sma_ratio",
    "volatility",
    "momentum",
    "obi_ratio",
    "adx",
    "atr",
]

Signal = str  # literal: "BUY" | "SELL" | "HOLD"
TrendStatus = str  # literal: "bullish" | "bearish" | "neutral"


def compute_htf_trend(
    closes: list[float],
    opens: list[float] | None = None,
) -> TrendStatus:
    """Determine the higher-timeframe trend direction.

    The trend is considered **bullish** when either of the following holds:
    * The latest close is above the 20-period EMA of all available close prices.
    * The last two candles are individually bullish (close ≥ open).

    The trend is considered **bearish** when:
    * The latest close is below the 20-period EMA, AND
    * At least the last candle is bearish (close < open).

    Returns ``"neutral"`` when there is insufficient data.

    Parameters
    ----------
    closes: Close prices in chronological order (oldest first).
    opens:  Open prices in chronological order (same length as *closes*).
            When ``None`` or empty the candle-body check is skipped.

    Returns
    -------
    ``"bullish"``, ``"bearish"``, or ``"neutral"``.
    """
    if len(closes) < _HTF_MIN_CANDLES:
        return HTF_TREND_NEUTRAL

    series = pd.Series(closes, dtype=float)
    ema_period = min(_SMA_PERIOD, len(closes))
    ema = series.ewm(span=ema_period, adjust=False).mean()
    current_close = float(series.iloc[-1])
    current_ema = float(ema.iloc[-1])

    above_ema = current_close > current_ema

    # Check whether the last two candles are bullish (close ≥ open)
    last_two_bullish = False
    if opens and len(opens) >= 2 and len(closes) >= 2:
        last_two_bullish = (
            closes[-1] >= opens[-1]
            and closes[-2] >= opens[-2]
        )

    if above_ema or last_two_bullish:
        return HTF_TREND_BULLISH

    # Bearish: below EMA and last candle is bearish
    last_candle_bearish = bool(opens and len(opens) >= 1 and closes[-1] < opens[-1])
    if not above_ema and last_candle_bearish:
        return HTF_TREND_BEARISH

    return HTF_TREND_NEUTRAL


class MLPredictor:
    """Predicts price-direction probability and generates trading signals.

    The underlying model is an XGBoost binary classifier trained on an
    eight-element feature vector:
    [sentiment, rsi, sma_ratio, volatility, momentum, obi_ratio, adx, atr].
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

    @staticmethod
    def _compute_adx_atr(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = _ADX_PERIOD,
    ) -> tuple[float, float]:
        """Compute ADX and ATR from OHLC data.

        Parameters
        ----------
        highs:   High prices in chronological order.
        lows:    Low prices in chronological order.
        closes:  Close prices in chronological order.
        period:  Smoothing period (default 14).

        Returns
        -------
        ``(adx, atr)`` – both are floats; defaults are ``(25.0, 0.0)`` when
        the series is too short to produce a valid result.
        """
        n = len(closes)
        if n < period + 1:
            return _ADX_NEUTRAL_DEFAULT, 0.0

        h = pd.Series(highs, dtype=float)
        l = pd.Series(lows, dtype=float)
        c = pd.Series(closes, dtype=float)
        prev_c = c.shift(1)
        prev_h = h.shift(1)
        prev_l = l.shift(1)

        # True Range
        tr = pd.concat(
            [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
        ).max(axis=1)

        # ATR (Wilder's EMA)
        atr_series = tr.ewm(com=period - 1, min_periods=period).mean()
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val):
            atr_val = 0.0

        # Directional Movement
        up_move = h - prev_h
        down_move = prev_l - l

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        smoothed_plus_dm = plus_dm.ewm(com=period - 1, min_periods=period).mean()
        smoothed_minus_dm = minus_dm.ewm(com=period - 1, min_periods=period).mean()
        smoothed_atr = tr.ewm(com=period - 1, min_periods=period).mean()

        # Avoid division by zero
        safe_atr = smoothed_atr.replace(0.0, np.nan)
        plus_di = 100 * smoothed_plus_dm / safe_atr
        minus_di = 100 * smoothed_minus_dm / safe_atr

        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = (100 * di_diff / di_sum.replace(0.0, np.nan)).fillna(0.0)

        adx_series = dx.ewm(com=period - 1, min_periods=period).mean()
        adx_val = float(adx_series.iloc[-1])
        if np.isnan(adx_val):
            adx_val = _ADX_NEUTRAL_DEFAULT  # neutral default

        return adx_val, atr_val

    def _compute_features(
        self,
        prices: list[float],
        sentiment: float,
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        obi_ratio: float = 1.0,
    ) -> list[float]:
        """Compute technical indicators and return a feature vector.

        Parameters
        ----------
        prices:    Recent mid-prices in chronological order.
        sentiment: Current VADER compound score in [-1, +1].
        highs:     Optional high prices (same length as *prices*); used for
                   ADX / ATR calculation.  Defaults to ``None`` (ADX=25, ATR=0).
        lows:      Optional low prices (same length as *prices*); used for
                   ADX / ATR calculation.  Defaults to ``None`` (ADX=25, ATR=0).
        obi_ratio: Order Book Imbalance ratio (bid_vol / ask_vol, top-5 depth).
                   Defaults to ``1.0`` (perfectly balanced book).

        Returns
        -------
        An eight-element list
        ``[sentiment, rsi, sma_ratio, volatility, momentum, obi_ratio, adx, atr]``
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

        # ADX / ATR: require matching highs and lows series
        if highs is not None and lows is not None and len(highs) >= _ATR_PERIOD + 1:
            adx, atr = self._compute_adx_atr(highs, lows, prices)
        else:
            adx, atr = _ADX_NEUTRAL_DEFAULT, 0.0  # neutral defaults when OHLCV unavailable

        return [float(sentiment), rsi, sma_ratio, volatility, momentum, float(obi_ratio), adx, atr]

    # ------------------------------------------------------------------
    # Training / warm-start
    # ------------------------------------------------------------------

    def warm_start(
        self,
        prices: list[float],
        sentiment_scores: list[float] | None = None,
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        obi_ratios: list[float] | None = None,
        horizon: int = _PREDICTION_HORIZON,
    ) -> bool:
        """Train (or re-train) the model from historical price data.

        This method is idempotent – calling it multiple times is safe and will
        simply replace the previously fitted model.

        Parameters
        ----------
        prices:           Historical mid/close-prices in chronological order.
        sentiment_scores: Optional per-tick sentiment scores (same length as
                          *prices*).  Defaults to ``0.0`` for all ticks.
        highs:            Optional high prices per tick (for ADX/ATR features).
                          When ``None``, ADX defaults to 25.0 and ATR to 0.0.
        lows:             Optional low prices per tick (for ADX/ATR features).
                          When ``None``, ADX defaults to 25.0 and ATR to 0.0.
        obi_ratios:       Optional per-tick OBI ratios.  Defaults to ``1.0``.
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

        obi_series = (
            pd.Series(obi_ratios, dtype=float)
            if obi_ratios is not None
            else pd.Series(1.0, index=series.index)
        )

        # ADX / ATR series (row-by-row computation for the full history)
        if highs is not None and lows is not None and len(highs) == len(prices):
            h_series = pd.Series(highs, dtype=float)
            l_series = pd.Series(lows, dtype=float)
            prev_c = series.shift(1)

            tr = pd.concat(
                [h_series - l_series,
                 (h_series - prev_c).abs(),
                 (l_series - prev_c).abs()],
                axis=1,
            ).max(axis=1)

            up_move = h_series - h_series.shift(1)
            down_move = l_series.shift(1) - l_series
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

            smth_atr = tr.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean()
            smth_pdm = plus_dm.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean()
            smth_mdm = minus_dm.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean()

            safe_atr = smth_atr.replace(0.0, np.nan)
            plus_di = 100 * smth_pdm / safe_atr
            minus_di = 100 * smth_mdm / safe_atr
            di_sum = plus_di + minus_di
            dx = (100 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)).fillna(0.0)
            adx_series = dx.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean().fillna(_ADX_NEUTRAL_DEFAULT)
            atr_series = smth_atr.fillna(0.0)
        else:
            adx_series = pd.Series(25.0, index=series.index)
            atr_series = pd.Series(0.0, index=series.index)

        df = pd.DataFrame(
            {
                "sentiment": sentiment_series,
                "rsi": rsi_series,
                "sma_ratio": sma_ratio_series,
                "volatility": vol_series,
                "momentum": mom_series,
                "obi_ratio": obi_series,
                "adx": adx_series,
                "atr": atr_series,
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
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        obi_ratio: float = 1.0,
    ) -> float | None:
        """Return the probability of an upward price move.

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
                         At least ``_MIN_PRICES_FOR_INFERENCE`` (20) values are
                         required.
        sentiment_score: Current VADER compound score in [-1, +1].
        highs:           Optional high prices for ADX/ATR computation.
        lows:            Optional low prices for ADX/ATR computation.
        obi_ratio:       Order Book Imbalance ratio (bid_vol / ask_vol, top-5).

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

        features = self._compute_features(prices, sentiment_score, highs, lows, obi_ratio)
        X = pd.DataFrame([features], columns=_FEATURE_COLS)
        try:
            proba: float = float(self._model.predict_proba(X)[0][1])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "predict_proba: XGBoost inference failed (%s) – "
                "model may need retraining with the new feature set.",
                exc,
            )
            return None

        rsi = features[1]
        volatility = features[3]
        momentum = features[4]
        obi = features[5]
        adx = features[6]
        atr = features[7]
        logger.info(
            "[INDICATORS] RSI: %.1f | Volatility: %.4f | Momentum: %.1f%% | "
            "OBI: %.4f | ADX: %.1f | ATR: %.4f",
            rsi,
            volatility,
            momentum,
            obi,
            adx,
            atr,
        )
        logger.debug("Predicted probability=%.4f", proba)
        return proba

    def generate_signal(
        self,
        prices: list[float],
        sentiment_score: float = 0.0,
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        obi_ratio: float = 1.0,
        funding_rate: float = 0.0,
        htf_trend_4h: TrendStatus = HTF_TREND_NEUTRAL,
        htf_trend_1h: TrendStatus = HTF_TREND_NEUTRAL,
    ) -> Signal:
        """Generate a trading signal (BUY / SELL / HOLD).

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
        sentiment_score: Current VADER compound score in [-1, +1].
        highs:           Optional high prices for ADX/ATR computation.
        lows:            Optional low prices for ADX/ATR computation.
        obi_ratio:       Order Book Imbalance ratio (bid_vol / ask_vol, top-5).
        funding_rate:    Current perpetual-futures funding rate as a decimal
                         (e.g. 0.0001 = 0.01 % per 8-hour window).
        htf_trend_4h:    4-hour trend status (``"bullish"``, ``"bearish"``, or
                         ``"neutral"``).  A ``"bearish"`` value causes any BUY
                         signal to be muted to HOLD ("General" filter).
        htf_trend_1h:    1-hour trend status (same values).  A ``"bearish"``
                         value also mutes BUY signals ("Colonel" filter).

        Returns
        -------
        ``"BUY"``, ``"SELL"``, or ``"HOLD"`` with the following logic:

        Base ML signal:
          * probability > 0.65 AND sentiment > 0.3   → BUY
          * probability < 0.3  AND sentiment < -0.3  → SELL
          * otherwise                                → HOLD

        HTF Filter ("General" + "Colonel"):
          * 4H trend bearish                         → BUY → HOLD
          * 1H trend bearish                         → BUY → HOLD

        Market Regime override (ADX-based):
          * ADX > 25 (trending): BUY only when RSI > 50 AND momentum > 0;
            SELL only when RSI < 50 AND momentum < 0.
          * ADX < 20 (range-bound): BUY when RSI < 35; SELL when RSI > 65.

        Funding Rate bias:
          * rate > +0.03 % (extreme greed): BUY → HOLD (Short-Bias penalty).
          * rate < -0.03 % (extreme fear):  BUY retained + Long-Bias bonus.

        ``[ELITE]`` is logged whenever the regime or funding-rate logic alters
        the base ML signal.
        """
        probability = self.predict_proba(prices, sentiment_score, highs, lows, obi_ratio)

        if probability is None:
            logger.info("Signal=HOLD (model not ready or insufficient data)")
            return "HOLD"

        # ── Base ML signal ────────────────────────────────────────────────
        if probability > _BUY_PROB_THRESHOLD and sentiment_score > _BUY_SENTIMENT_THRESHOLD:
            base_signal: Signal = "BUY"
        elif probability < _SELL_PROB_THRESHOLD and sentiment_score < _SELL_SENTIMENT_THRESHOLD:
            base_signal = "SELL"
        else:
            base_signal = "HOLD"

        # Retrieve the just-computed features so we can inspect regime inputs
        features = self._compute_features(prices, sentiment_score, highs, lows, obi_ratio)
        rsi = features[1]
        momentum = features[4]
        adx = features[6]

        # ── Market Regime Classifier ──────────────────────────────────────
        elite_factors: list[str] = []
        signal: Signal = base_signal

        if adx > _ADX_TREND_THRESHOLD:
            # Trend-Following mode: trust RSI and Momentum
            if base_signal == "BUY" and not (rsi > 50 and momentum > 0):
                signal = "HOLD"
                elite_factors.append(f"ADX={adx:.1f}>25 trend-following: RSI/momentum not aligned")
            elif base_signal == "SELL" and not (rsi < 50 and momentum < 0):
                signal = "HOLD"
                elite_factors.append(f"ADX={adx:.1f}>25 trend-following: RSI/momentum not aligned")
            elif base_signal != "HOLD":
                elite_factors.append(f"ADX={adx:.1f}>25 trend-following confirmed")

        elif adx < _ADX_RANGE_THRESHOLD:
            # Mean-Reversion mode: oversold/overbought bounces
            if rsi < _RSI_OVERSOLD and sentiment_score >= 0:
                if signal != "BUY":
                    signal = "BUY"
                    elite_factors.append(
                        f"ADX={adx:.1f}<20 mean-reversion: RSI oversold ({rsi:.1f})"
                    )
                else:
                    elite_factors.append(
                        f"ADX={adx:.1f}<20 mean-reversion: RSI oversold ({rsi:.1f}) confirmed"
                    )
            elif rsi > _RSI_OVERBOUGHT and sentiment_score <= 0:
                if signal != "SELL":
                    signal = "SELL"
                    elite_factors.append(
                        f"ADX={adx:.1f}<20 mean-reversion: RSI overbought ({rsi:.1f})"
                    )
                else:
                    elite_factors.append(
                        f"ADX={adx:.1f}<20 mean-reversion: RSI overbought ({rsi:.1f}) confirmed"
                    )

        # ── Funding Rate Bias ─────────────────────────────────────────────
        if signal == "BUY" and funding_rate > _FUNDING_RATE_EXTREME_GREED:
            signal = "HOLD"
            elite_factors.append(
                f"Funding rate={funding_rate:.6f} > +0.03% extreme greed: Short-Bias penalty"
            )
        elif signal == "BUY" and funding_rate < _FUNDING_RATE_EXTREME_FEAR:
            elite_factors.append(
                f"Funding rate={funding_rate:.6f} < -0.03% extreme fear: Long-Bias bonus"
            )

        # ── HTF Trend Filter ("General" 4H + "Colonel" 1H) ───────────────
        if signal == "BUY" and htf_trend_4h == HTF_TREND_BEARISH:
            signal = "HOLD"
            elite_factors.append(
                f"[MTA] General filter: 4H trend is bearish – BUY muted to HOLD"
            )
        elif signal == "BUY" and htf_trend_1h == HTF_TREND_BEARISH:
            signal = "HOLD"
            elite_factors.append(
                f"[MTA] Colonel filter: 1H trend is bearish – BUY muted to HOLD"
            )

        # ── Logging ───────────────────────────────────────────────────────
        if elite_factors:
            logger.info(
                "[ELITE] Signal=%s (base=%s)  probability=%.4f  sentiment=%.4f  "
                "4H=%s  1H=%s  factors=[%s]",
                signal,
                base_signal,
                probability,
                sentiment_score,
                htf_trend_4h,
                htf_trend_1h,
                " | ".join(elite_factors),
            )
        else:
            logger.info(
                "Signal=%s  probability=%.4f  sentiment=%.4f  4H=%s  1H=%s",
                signal,
                probability,
                sentiment_score,
                htf_trend_4h,
                htf_trend_1h,
            )
        return signal
