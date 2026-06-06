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


# Production SYMBOL_CONFIG — tuned 2026-06-02 via disk-loaded backtest + post-retrain
# calibration refit + per-symbol param sweep on 60d OOS.
# Baseline snapshot (pre-2026-06-02): logs/symbol_config_baseline.json.
SYMBOL_CONFIG: dict[str, dict[str, Any]] = {
    "BTC/USDT": {
        # 2026-06-06 retrain: AUC 0.7148, but calibrated probs max at 0.069 — class imbalance
        # (1.5% move in 4h is rare for BTC@100k). Threshold lowered to match calibrated range.
        # 60d backtest pt=0.05: 9 trades, WR 55.6%, PnL +4.62%, Sharpe 0.89. Marginal, monitor.
        "prob_threshold": 0.05,
        "fixed_sl_pct": 0.020,
        "fixed_tp_pct": 0.040,
        "use_sma_filter": True,
        "risk": 0.015,
        "timeframe": "30m",
        "horizon": 36,
        "skip_regime": False,  # E3: ADX regime gate active
        "exec_costs": {"spread_bps": 2.0, "slippage_atr_mult": 0.05},
    },
    "ETH/USDT": {
        # PT raised 2026-06-03: portfolio backtest 60d showed WR 32% PnL -$353
        # with pt=0.50. Calibration shifted post v3 retrain; raising threshold
        # filters weak signals. Reevaluate after 30d demo.
        "prob_threshold": 0.75,
        "fixed_sl_pct": 0.020,
        "fixed_tp_pct": 0.030,
        "use_sma_filter": True,
        "risk": 0.015,
        "timeframe": "15m",
        "horizon": 20,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 2.0, "slippage_atr_mult": 0.05},
    },
    # XRP/USDT DEPRECATED 2026-06-04 — demoted (backtest -9.2% over 30d, risk 0.005 simbólico).
    # Not in active WATCHLIST. Re-evaluate only after model rebuild. Do not re-enable.
    "XRP/USDT": {
        "prob_threshold": 0.95,
        "fixed_sl_pct": 0.025,
        "fixed_tp_pct": 0.040,
        "use_sma_filter": True,
        "risk": 0.001,
        "timeframe": "15m",
        "horizon": 16,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 2.0, "slippage_atr_mult": 0.05},
    },
    "SOL/USDT": {
        # Threshold sweep 2026-06-03: pt=0.70 → 65 trades, Sharpe +0.04, PnL -2.86%.
        "prob_threshold": 0.70,
        "fixed_sl_pct": 0.025,
        "fixed_tp_pct": 0.035,
        "use_sma_filter": True,
        "risk": 0.015,
        "timeframe": "30m",
        "horizon": 24,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 5.0, "slippage_atr_mult": 0.10},
    },
    "DOGE/USDT": {
        # Disk sweep 2026-06-02 (60d, calibrated): pt=0.55 tp=0.045 sl=0.025
        # → 32 trades, WR 90.6%, PnL +22.22%, PF 44.21.
        "prob_threshold": 0.55,
        "fixed_sl_pct": 0.025,
        "fixed_tp_pct": 0.045,
        "use_sma_filter": True,
        "risk": 0.010,
        "timeframe": "15m",
        "horizon": 16,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 5.0, "slippage_atr_mult": 0.10},
    },
    # ── Phase D survivors (2026-06-02): inline 70/30 OOS validated ──
    "NEAR/USDT": {
        # Threshold sweep 2026-06-03: pt=0.65 → 54 trades, Sharpe +0.03, PnL -1.68%.
        "prob_threshold": 0.65,
        "fixed_sl_pct": 0.030,
        "fixed_tp_pct": 0.050,
        "use_sma_filter": True,
        "risk": 0.010,
        "timeframe": "15m",
        "horizon": 20,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 8.0, "slippage_atr_mult": 0.15},
    },
    # ATOM/USDT REMOVED 2026-06-03 — threshold sweep 0.30-0.95 all negative.
    # Best Sharpe -0.02 at pt=0.55 (PnL -4.22%). Model unprofitable across all settings.
    # If you want to re-enable: retrain with --vibe or --mtf and re-sweep.
    "LINK/USDT": {
        # Symbol scan 2026-06-02 (54d OOS): pt=0.55 tp=0.050 sl=0.030
        # → 31 trades, WR 48.4%, PnL +1.00%, PF 1.55. MARGINAL — included for
        # diversification at low risk. Skip from .env WATCHLIST until Phase F
        # confirms profitability on the most recent 30d.
        "prob_threshold": 0.55,
        "fixed_sl_pct": 0.030,
        "fixed_tp_pct": 0.050,
        "use_sma_filter": True,
        "risk": 0.008,
        "timeframe": "15m",
        "horizon": 20,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 5.0, "slippage_atr_mult": 0.10},
    },
    # JTO/USDT DISABLED 2026-06-06 — retrain AUC 0.5339 below min 0.55, model rejected.
    # No predictive signal found across any threshold. Remove from WATCHLIST.
    "JTO/USDT": {
        "prob_threshold": 0.99,
        "fixed_sl_pct": 0.025,
        "fixed_tp_pct": 0.040,
        "use_sma_filter": True,
        "risk": 0.001,
        "timeframe": "15m",
        "horizon": 20,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 8.0, "slippage_atr_mult": 0.15},
    },
    # INJ/USDT WATCH-ONLY 2026-06-06 — retrain AUC 0.5642, barely above random.
    # 60d backtest pt=0.20 shows +151% but model is partially in-sample (just retrained).
    # Threshold=0.20 matches calibrated range. Risk minimal until 30d live OOS confirms.
    # NOT in active WATCHLIST — add after validated performance.
    "INJ/USDT": {
        "prob_threshold": 0.20,
        "fixed_sl_pct": 0.025,
        "fixed_tp_pct": 0.040,
        "use_sma_filter": True,
        "risk": 0.005,
        "timeframe": "15m",
        "horizon": 20,
        "skip_regime": True,
        "exec_costs": {"spread_bps": 8.0, "slippage_atr_mult": 0.15},
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


def load_short_booster(symbol: str):
    """Load SHORT model for symbol. Returns None if no model file exists."""
    if symbol in _short_booster_cache:
        return _short_booster_cache[symbol]
    path = short_model_json_path_for_symbol(symbol)
    if path is None:
        return None
    try:
        if _is_ensemble_meta(path):
            from strategy.ensemble import EnsemblePredictor
            m = EnsemblePredictor.load(path)
        else:
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


def _is_ensemble_meta(path: Path) -> bool:
    try:
        import json
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("type") == "EnsemblePredictor"
    except Exception:
        return False


def load_booster_from_disk(path: Path | None = None):
    """Load model from *path* or default ``_XGB_MODEL_PATH`` (cached per resolved path).

    Returns either an XGBClassifier or an EnsemblePredictor depending on the JSON.
    """
    p = (path or _XGB_MODEL_PATH).resolve()
    key = str(p)
    if key in _booster_cache:
        return _booster_cache[key]
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if _is_ensemble_meta(p):
        from strategy.ensemble import EnsemblePredictor
        model = EnsemblePredictor.load(p)
    else:
        model = XGBClassifier()
        model.load_model(str(p))
    _booster_cache[key] = model
    return model

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
        self._short_warn_logged: set[str] = set()
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

    def _finalize_features(
        self,
        features: list[float],
        booster: Any,
        vibe_features: list[float] | None = None,
    ) -> "pd.DataFrame":
        """Build the model input frame from raw features for any booster.

        Appends VIBE features, pads/truncates to the booster's expected feature
        count, and labels columns with the booster's own feature names (falling
        back to ``FINAL_FEATURE_ORDER``). Shared by the LONG and SHORT paths so
        both absorb models of any feature count (e.g. 30 quant+vibe or 36 MTF)
        without a shape/column mismatch. Features the runtime does not compute
        (e.g. higher-timeframe columns on an MTF model) are zero-padded.
        """
        feats = list(features)
        if _VIBE_ENABLED:
            if vibe_features is not None and len(vibe_features) == len(VIBE_FEATURE_COLS):
                feats.extend(vibe_features)
            else:
                feats.extend(VIBE_FEATURE_NEUTRAL)

        try:
            model_features = int(booster.n_features_in_)
        except AttributeError:
            try:
                model_features = int(booster.get_booster().num_features())
            except Exception:
                model_features = len(_FEATURE_COLS)

        # Auto-Padding: V2 model (16) with only 12 input features
        if model_features == 16 and len(feats) == 12:
            feats.extend(VIBE_FEATURE_NEUTRAL)

        # Generic length safety net
        if len(feats) < model_features:
            feats.extend([0.0] * (model_features - len(feats)))
        elif len(feats) > model_features:
            feats = feats[:model_features]

        cols = self._model_feature_names(booster, model_features)
        return pd.DataFrame([feats], columns=cols)

    @staticmethod
    def _model_feature_names(booster: Any, model_features: int) -> list[str]:
        """Return exactly ``model_features`` column names for the booster.

        Prefers the booster's stored feature names (sklearn ``feature_names_in_``
        or the underlying booster's ``feature_names``) so the DataFrame matches
        what the model was trained on; otherwise falls back to
        ``FINAL_FEATURE_ORDER`` and synthetic ``f{i}`` names for any overflow.
        """
        names: list[str] | None = None
        raw = getattr(booster, "feature_names_in_", None)
        if raw is not None:
            names = [str(n) for n in raw]
        else:
            try:
                bn = booster.get_booster().feature_names
                if bn:
                    names = [str(n) for n in bn]
            except Exception:
                names = None
        if not names:
            names = list(FINAL_FEATURE_ORDER)
        if len(names) < model_features:
            names = names + [f"f{i}" for i in range(len(names), model_features)]
        return names[:model_features]

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

        # ------------------------------------------------------------------
        # Resolve booster (VIBE append + padding handled by _finalize_features)
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

        X = self._finalize_features(features, booster, vibe_features)
        features = list(X.iloc[0])  # keep downstream debug logging consistent
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
            X = self._finalize_features(features, short_model)
            proba = float(short_model.predict_proba(X)[0, 1])
        except Exception as exc:  # noqa: BLE001
            # Warn once per symbol to avoid every-cycle Telegram spam.
            if symbol not in self._short_warn_logged:
                self._short_warn_logged.add(symbol)
                logger.warning(
                    "SHORT predict_proba failed %s: %s - SHORT model may need retraining "
                    "with the current feature set.",
                    symbol,
                    exc,
                )
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
