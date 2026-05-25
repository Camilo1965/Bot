"""
strategy.ml_predictor
~~~~~~~~~~~~~~~~~~~~~~

Dedicated XGBoost classifier (quant OHLCV features: RSI, MACD, ATR, BB, vol delta, log returns).

Entry rule
----------
* ``symbol`` must be in ``ALLOWED_SYMBOLS`` (otherwise **HOLD**).
* ``probability >= BUY_PROB_THRESHOLD`` (default 0.50 max-performance profile; override ``BUY_PROB_THRESHOLD`` env) → **BUY**, else **HOLD**.

No sentiment, HTF, funding, or regime overrides - the loaded model is the only
entry gate.  :meth:`MLPredictor.warm_start` still retrains from history when no
artifact is present; weekly retrainer may refresh the active model path in this file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from strategy.prob_calibration import calibrate_probability
from strategy.quant_features import (
    DEFAULT_LABEL_ROUND_TRIP,
    FINAL_FEATURE_ORDER,
    MIN_OHLC_ROWS,
    QUANT_FEATURE_COLS,
    VIBE_FEATURE_COLS,
    add_quant_features,
    compute_quant_vector_from_lists,
    forward_return_label,
    htf_sma200_1h_allows_long,
)
from vibe.feature_bridge import extract_vibe_features, VIBE_FEATURE_NEUTRAL

logger = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Optimization for MT5 known symbols with higher Profit Factor and daily consistency.
SYMBOL_CONFIG: dict[str, dict[str, Any]] = {
    "BTC/USDT": {
        "prob_threshold": 0.65,     # Exhaust search: thresh=0.65 only viable combo. Signal marginal.
        "fixed_sl_pct": 0.010,      # 1.0% SL — tight because BTC model has low signal strength
        "fixed_tp_pct": 0.075,      # 7.5% TP (R:R = 7.5:1 needed for 15% WR to be profitable)
        "use_sma_filter": False,
        "risk": 0.015,
        "timeframe": "30m",
        "horizon": 24,              # 12h max hold (was 8 bars = 4h)
    },
    "ETH/USDT": {
        "prob_threshold": 0.40,     # Exhaust search: 3.6x alpha vs random at 0.40. 555 fires/156d = 3.6/day.
        "fixed_sl_pct": 0.020,      # 2.0% SL — wider needed for 9h hold without random whipsaw
        "fixed_tp_pct": 0.075,      # 7.5% TP. EV=+0.586%/trade. WR=15.1% vs random 4.2%.
        "use_sma_filter": True,
        "risk": 0.015,
        "timeframe": "15m",
        "horizon": 36,              # 9h max hold (was 8 bars = 2h) — path sim optimal
    },
    "XRP/USDT": {
        "prob_threshold": 0.45,     # Exhaust search: 8.9x alpha vs random. EV=+1.206%/trade (best signal).
        "fixed_sl_pct": 0.015,      # 1.5% SL
        "fixed_tp_pct": 0.100,      # 10% TP. EV(no-timeout)=+0.235%. Model selects real 10% moves.
        "use_sma_filter": True,
        "risk": 0.015,
        "timeframe": "15m",
        "horizon": 36,              # 9h max hold — needed for 10% TP to be reachable
        # Direction model v2 has precision=0 (p99=0.38 < threshold=0.45).
        # Use regime model (AUC=0.94) as fallback entry signal until retrain.
        "regime_only": True,
        "regime_entry_threshold": 0.82,
    },
}

ALLOWED_SYMBOLS: tuple[str, ...] = tuple(SYMBOL_CONFIG.keys())
BUY_PROB_THRESHOLD: float = _float_env("BUY_PROB_THRESHOLD", 0.50)
_XGB_MODEL_PATH = MODELS_DIR / "ETH_USDT_v1.json"
_booster_cache: dict[str, XGBClassifier] = {}

_LABEL_ROUND_TRIP: float = DEFAULT_LABEL_ROUND_TRIP

# Back-test / warm_start labelling only (not used for live entry)
_SELL_PROB_THRESHOLD = 0.35
_SELL_SENTIMENT_THRESHOLD = -0.3


def model_json_path_for_symbol(symbol: str) -> Path:
    """Resolve the newest available LONG model path with fallback chain."""
    base = MODELS_DIR / symbol.replace("/", "_")
    candidates = [f"{base}_v3.json", f"{base}_v2.json", f"{base}_v1.json"] if _VIBE_ENABLED else [f"{base}_v2.json", f"{base}_v1.json"]
    for p in candidates:
        if Path(p).is_file():
            return Path(p)
    default_suffix = "_v3" if _VIBE_ENABLED else "_v2"
    return Path(f"{base}{default_suffix}.json")


def short_model_json_path_for_symbol(symbol: str) -> Path | None:
    """Resolve SHORT model path. Returns None if no short model exists."""
    base = MODELS_DIR / (symbol.replace("/", "_") + "_short")
    for suffix in ["_v2.json", "_v1.json"]:
        p = Path(str(base) + suffix)
        if p.is_file():
            return p
    return None


_short_booster_cache: dict[str, XGBClassifier] = {}


def load_short_booster(symbol: str) -> XGBClassifier | None:
    """Load SHORT model for symbol. Returns None if no model file exists."""
    if symbol in _short_booster_cache:
        return _short_booster_cache[symbol]
    path = short_model_json_path_for_symbol(symbol)
    if path is None:
        return None
    try:
        m = XGBClassifier()
        m.load_model(str(path))
        _short_booster_cache[symbol] = m
        logger.info("SHORT model loaded: %s (%s)", symbol, path.name)
        return m
    except Exception as exc:
        logger.warning("SHORT model load failed %s: %s", symbol, exc)
        return None


def get_symbol_config(symbol: str) -> dict[str, Any]:
    """Per-symbol ML / risk knobs; unknown symbols fall back to env defaults."""
    base: dict[str, Any] = {
        "prob_threshold": BUY_PROB_THRESHOLD,
        "fixed_sl_pct": 0.025,
        "use_sma_filter": True,
        "risk": _float_env("RISK_PER_TRADE", 0.02),
    }
    if symbol in SYMBOL_CONFIG:
        return {**base, **SYMBOL_CONFIG[symbol]}
    return base


def load_booster_from_disk(path: Path | None = None) -> XGBClassifier:
    """Load XGB JSON from *path* or default ``_XGB_MODEL_PATH`` (cached per resolved path)."""
    p = (path or _XGB_MODEL_PATH).resolve()
    key = str(p)
    if key in _booster_cache:
        return _booster_cache[key]
    if not p.is_file():
        raise FileNotFoundError(str(p))
    booster = XGBClassifier()
    booster.load_model(str(p))
    _booster_cache[key] = booster
    return booster

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

# Minimum bars for quant feature stack (MACD/BB warm-up)
_MIN_PRICES_FOR_INFERENCE = MIN_OHLC_ROWS

# Default ADX value used when OHLCV data is unavailable (neutral - no regime)
_ADX_NEUTRAL_DEFAULT = 25.0

# HTF trend labels
HTF_TREND_BULLISH = "bullish"
HTF_TREND_BEARISH = "bearish"
HTF_TREND_NEUTRAL = "neutral"

# Minimum number of HTF candles required to compute a trend
_HTF_MIN_CANDLES = 3

_VIBE_ENABLED: bool = os.environ.get("VIBE_FEATURES_ENABLED", "0").strip() in ("1", "true", "yes")
_FEATURE_COLS = list(FINAL_FEATURE_ORDER) if _VIBE_ENABLED else list(QUANT_FEATURE_COLS)

Signal = str  # literal: "BUY" | "SELL" | "HOLD"
TrendStatus = str  # literal: "bullish" | "bearish" | "neutral"


def compute_htf_trend(
    closes: list[float],
    opens: list[float] | None = None,
) -> TrendStatus:
    """Determine the higher-timeframe trend direction.

    The trend is considered **bullish** when either of the following holds:
    * The latest close is above the 20-period EMA of all available close prices.
    * The last two candles are individually bullish (close >= open).

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

    # Check whether the last two candles are bullish (close >= open)
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
    """XGBoost binary classifier on :data:`~strategy.quant_features.QUANT_FEATURE_COLS`."""

    def __init__(self) -> None:
        self._model = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
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
        ``(adx, atr)`` - both are floats; defaults are ``(25.0, 0.0)`` when
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
        volumes: list[float] | None = None,
        symbol: str = "ETH/USDT",
    ) -> list[float] | None:
        """Compute quant features (same pipeline as ``scripts/quant_sweep.py``)."""
        del sentiment, obi_ratio
        if highs is None or lows is None:
            highs = prices
            lows = prices
        
        cfg = get_symbol_config(symbol)
        tf_str = cfg.get("timeframe", "15m")
        tf_min = int(tf_str[:-1]) if tf_str[:-1].isdigit() else 15
        
        vec = compute_quant_vector_from_lists(prices, highs, lows, volumes, base_timeframe_min=tf_min)
        if vec is None:
            return None
        return vec

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
        volumes: list[float] | None = None,
        horizon: int = _PREDICTION_HORIZON,
    ) -> bool:
        """Train from OHLC(V); labels match sweep (forward return vs round-trip cost)."""
        del sentiment_scores, obi_ratios
        n = len(prices)
        if highs is None:
            highs = list(prices)
        if lows is None:
            lows = list(prices)
        vol = volumes if volumes is not None and len(volumes) == n else [0.0] * n

        df = pd.DataFrame(
            {
                "open": prices,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": vol[:n],
            }
        )
        feat = add_quant_features(df)
        feat["label"] = forward_return_label(feat["close"], horizon, _LABEL_ROUND_TRIP)

        # [VIBE FASE 4] Add neutral VIBE columns to historical training data
        if _VIBE_ENABLED:
            for col, val in zip(VIBE_FEATURE_COLS, VIBE_FEATURE_NEUTRAL):
                if col not in feat.columns:
                    feat[col] = val

        feat = feat.dropna(subset=_FEATURE_COLS + ["label"])

        X = feat[_FEATURE_COLS]
        y = feat["label"]

        if len(X) < 10:
            logger.warning(
                "warm_start: not enough labelled samples (%d). Model not trained.",
                len(X),
            )
            return False

        self._model.fit(X, y)
        self._is_trained = True
        logger.info(
            "MLPredictor warm-started on %d samples (%d quant features).",
            len(X),
            X.shape[1],
        )
        return True

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def load_model(self, filepath: str | Path | None = None) -> bool:
        """Load XGB JSON from *filepath* or default ``_XGB_MODEL_PATH``."""
        path = Path(filepath) if filepath else _XGB_MODEL_PATH
        try:
            self._model = load_booster_from_disk(path)
            self._is_trained = True
            logger.info("MLPredictor loaded model from %s.", path)
            return True
        except FileNotFoundError:
            logger.info("load_model: file not found at %s.", path)
            return False
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
            logger.warning("save_model: model is not trained - nothing to save.")
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
        volumes: list[float] | None = None,
        *,
        symbol: str | None = None,
        precomputed_features: list[float] | None = None,
        vibe_features: list[float] | None = None,
    ) -> float | None:
        """Return the probability of an upward price move.

        Parameters
        ----------
        prices:          Recent mid-prices in chronological order.
                         At least ``_MIN_PRICES_FOR_INFERENCE`` (20) values are
                         required.
        sentiment_score: Ignored (API compatibility).
        highs:           Optional high prices for ADX/ATR computation.
        lows:            Optional low prices for ADX/ATR computation.
        obi_ratio:       Ignored (API compatibility).
        symbol:          When set and not in ``ALLOWED_SYMBOLS``, returns ``None``.
        precomputed_features: Optional precomputed feature vector to avoid redundant calculation.
        vibe_features:   Optional 4 VIBE-derived features (Fase 4).  When ``None``
                         and ``VIBE_FEATURES_ENABLED=1``, neutral values are padded.

        Returns
        -------
        A float in [0, 1] representing the predicted probability, or
        ``None`` when the model is untrained or there is insufficient history.
        """
        if symbol is not None and symbol not in ALLOWED_SYMBOLS:
            return None

        if precomputed_features is not None:
            features = list(precomputed_features)
        else:
            if len(prices) < _MIN_PRICES_FOR_INFERENCE:
                logger.debug(
                    "predict_proba: not enough prices (%d < %d).",
                    len(prices),
                    _MIN_PRICES_FOR_INFERENCE,
                )
                return None

            features = self._compute_features(
                prices, sentiment_score, highs, lows, obi_ratio, volumes=volumes
            )

        if features is None:
            return None

        # [VIBE FASE 4] Append VIBE features if enabled
        if _VIBE_ENABLED:
            if vibe_features is not None and len(vibe_features) == len(VIBE_FEATURE_COLS):
                features.extend(vibe_features)
            else:
                features.extend(VIBE_FEATURE_NEUTRAL)

        # ------------------------------------------------------------------
        # Resolve booster and determine expected feature count
        # ------------------------------------------------------------------
        sym = symbol or "ETH/USDT"
        booster: XGBClassifier | None = None
        mp = model_json_path_for_symbol(sym)
        if mp.is_file():
            try:
                booster = load_booster_from_disk(mp)
            except Exception as exc:  # noqa: BLE001
                logger.warning("predict_proba: load %s failed: %s", mp, exc)
        if booster is None and self._is_trained:
            booster = self._model
        if booster is None:
            logger.debug("predict_proba: no model file for %s and predictor untrained.", sym)
            return None

        # Expected feature count from the loaded model
        try:
            model_features = int(booster.n_features_in_)
        except AttributeError:
            try:
                model_features = int(booster.get_booster().num_features())
            except Exception:
                model_features = len(_FEATURE_COLS)

        input_features = len(features)

        # Auto-Padding: V2 model (16) with only 12 input features
        if model_features == 16 and input_features == 12:
            logger.error(
                "[XGBOOST] Mismatch detectado. Esperadas: %d, Recibidas: %d. Aplicando corrección...",
                model_features,
                input_features,
            )
            features.extend(VIBE_FEATURE_NEUTRAL)

        # Generic length safety net
        if len(features) < model_features:
            features.extend([0.0] * (model_features - len(features)))
        elif len(features) > model_features:
            features = features[:model_features]

        # Force column order using FINAL_FEATURE_ORDER
        cols = FINAL_FEATURE_ORDER[:model_features]
        X = pd.DataFrame([features], columns=cols)
        try:
            proba: float = float(booster.predict_proba(X)[0][1])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "predict_proba: XGBoost inference failed (%s) - "
                "model may need retraining with the new feature set.",
                exc,
            )
            return None

        # Apply isotonic calibration if a calibration file exists for this symbol
        proba = calibrate_probability(proba, sym)

        logger.debug(
            "[QUANT_FEAT] rsi=%.2f macd_hist=%.6f atr=%.6f vol_rel=%.3f",
            features[0],
            features[3],
            features[4],
            features[11],
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
        volumes: list[float] | None = None,
        *,
        symbol: str = "ETH/USDT",
        vibe_features: list[float] | None = None,
    ) -> tuple[Signal, float]:
        """BUY only when HTF (precio > SMA200 1h) passes, then ``probability >= BUY_PROB_THRESHOLD``.
        
        Returns tuple of (Signal, Probability).
        """
        del htf_trend_4h, htf_trend_1h
        if symbol not in ALLOWED_SYMBOLS:
            logger.debug("Signal=HOLD (symbol %s not in ALLOWED_SYMBOLS)", symbol)
            return "HOLD", 0.0

        cfg = get_symbol_config(symbol)
        tf_str = cfg.get("timeframe", "15m")
        tf_min = int(tf_str[:-1]) if tf_str[:-1].isdigit() else 15
        
        # Precompute features once
        features = self._compute_features(
            prices, sentiment_score, highs, lows, obi_ratio, volumes=volumes
        )
        
        probability = self.predict_proba(
            prices,
            sentiment_score,
            highs,
            lows,
            obi_ratio,
            volumes=volumes,
            symbol=symbol,
            precomputed_features=features,
            vibe_features=vibe_features,
        )

        if probability is None:
            logger.debug("Signal=HOLD (model not ready or insufficient data)")
            return "HOLD", 0.0

        # Funding-rate bias: extreme positive → overleveraged longs, penalise BUY
        #                    extreme negative → shorts squeezed out, small boost
        if funding_rate > _FUNDING_RATE_EXTREME_GREED:
            probability = max(0.0, probability * 0.90)
            logger.debug("[FUNDING] %s rate=%.6f → prob penalised to %.4f", symbol, funding_rate, probability)
        elif funding_rate < _FUNDING_RATE_EXTREME_FEAR:
            probability = min(0.99, probability * 1.10)
            logger.debug("[FUNDING] %s rate=%.6f → prob boosted to %.4f", symbol, funding_rate, probability)

        if cfg["use_sma_filter"] and not htf_sma200_1h_allows_long(prices, base_timeframe_min=tf_min):
            logger.debug(
                "Signal=HOLD (precio <= SMA200 1h) prob=%.4f symbol=%s",
                probability,
                symbol,
            )
            return "HOLD", probability

        th = float(cfg["prob_threshold"])
        if probability >= th:
            logger.debug(
                "Signal=BUY  probability=%.4f  symbol=%s (th=%.2f)",
                probability,
                symbol,
                th,
            )
            return "BUY", probability

        logger.debug("Signal=HOLD  probability=%.4f  symbol=%s", probability, symbol)
        return "HOLD", probability

    def generate_short_signal(
        self,
        prices: list[float],
        highs: list[float] | None = None,
        lows: list[float] | None = None,
        volumes: list[float] | None = None,
        *,
        symbol: str = "ETH/USDT",
    ) -> tuple[Signal, float]:
        """Generate SHORT signal using the per-symbol SHORT model.

        Returns ("SHORT", probability) when model fires, else ("HOLD", probability).
        SHORT fires when price is BELOW SMA200 1h (downtrend confirmed).
        """
        if symbol not in ALLOWED_SYMBOLS:
            return "HOLD", 0.0

        short_model = load_short_booster(symbol)
        if short_model is None:
            return "HOLD", 0.0

        cfg = get_symbol_config(symbol)
        tf_str = cfg.get("timeframe", "15m")
        tf_min = int(tf_str[:-1]) if tf_str[:-1].isdigit() else 15

        features = self._compute_features(prices, 0.0, highs, lows, 1.0, volumes=volumes)
        if features is None:
            return "HOLD", 0.0

        try:
            proba = float(short_model.predict_proba([features])[0, 1])
        except Exception as exc:
            logger.debug("SHORT predict_proba error %s: %s", symbol, exc)
            return "HOLD", 0.0

        # SHORT only fires when price is BELOW SMA200 (downtrend)
        if htf_sma200_1h_allows_long(prices, base_timeframe_min=tf_min):
            logger.debug("SHORT HOLD (price > SMA200 1h) prob=%.4f %s", proba, symbol)
            return "HOLD", proba

        th = float(cfg.get("short_prob_threshold", cfg["prob_threshold"]))
        if proba >= th:
            logger.debug("SHORT signal prob=%.4f %s (th=%.2f)", proba, symbol, th)
            return "SHORT", proba

        return "HOLD", proba
