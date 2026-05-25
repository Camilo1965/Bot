"""
strategy.quant_features
~~~~~~~~~~~~~~~~~~~~~~~

Pure quantitative OHLCV features for XGB sweep + live inference.

v2 feature set (24 quant features + 4 VIBE = 28 total):
  RSI, MACD, ATR, Bollinger, volume delta, log returns (1/5/10/20),
  ADX, StochRSI, Williams %R, OBV, CMF,
  rolling skewness/kurtosis/sharpe,
  temporal features (hour, day-of-week),
  close_vs_sma200_1h, vol_rel.

Labeling: triple-barrier (stop-loss / take-profit / time-exit) in addition
to the classic forward-return threshold.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ── v2 Feature columns (24 quant) ────────────────────────────────────────────
QUANT_FEATURE_COLS: list[str] = [
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "atr",
    "bb_pct_b",
    "bb_width",
    "vol_delta_norm",
    "log_ret_1",
    "log_ret_5",
    "log_ret_10",
    "log_ret_20",
    "close_vs_sma200_1h",
    "vol_rel",
    "adx",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "williams_r",
    "obv",
    "cmf",
    "ret_skew_20",
    "ret_kurt_20",
    "roll_sharpe_20",
    "hour_sin",
    "hour_cos",
    "dow",
]

# Fase 4: VIBE-Trading features injected at inference time (optional)
VIBE_FEATURE_COLS: list[str] = [
    "vibe_pattern_score",
    "vibe_factor_ic",
    "vibe_journal_health",
    "vibe_backtest_sharpe",
]

# Orden final unificado para entrenamiento e inferencia (24 quant + 4 VIBE)
FINAL_FEATURE_ORDER: list[str] = QUANT_FEATURE_COLS + VIBE_FEATURE_COLS

# ── Parameters ───────────────────────────────────────────────────────────────
_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_ATR_PERIOD = 14
_BB_PERIOD = 20
_BB_STD = 2.0
_VOL_DELTA_WIN = 20
_VOL_REL_WIN = 20
_SMA200_1H_PERIODS = 200
_ADX_PERIOD = 14
_STOCH_RSI_PERIOD = 14
_WILLIAMS_R_PERIOD = 14
_OBV_INIT = 0.0
_CMF_PERIOD = 20
_ROLL_WIN = 20

# ~200 horas en velas 15m (4 por hora) para alinear subsample [::4] con SMA200 en 1h
MIN_OHLC_ROWS = max(60, (_SMA200_1H_PERIODS - 1) * 4 + 1)

# Friction model (aligned with execution.paper_executor._TAKER_FEE_RATE ≈ 0.0004)
DEFAULT_TAKER_FEE_RATE: float = 0.0004
DEFAULT_LABEL_ROUND_TRIP: float = 2.0 * DEFAULT_TAKER_FEE_RATE + 0.0005


# ── Internal helpers ─────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = _ATR_PERIOD) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f = close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=_MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
    hist = line - signal
    return line, signal, hist


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = _ADX_PERIOD) -> pd.Series:
    prev_c = close.shift(1)
    prev_h = high.shift(1)
    prev_l = low.shift(1)

    up_move = high - prev_h
    down_move = prev_l - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    smoothed_plus_dm = plus_dm.ewm(com=period - 1, min_periods=period).mean()
    smoothed_minus_dm = minus_dm.ewm(com=period - 1, min_periods=period).mean()

    safe_atr = atr.replace(0.0, np.nan)
    plus_di = 100 * smoothed_plus_dm / safe_atr
    minus_di = 100 * smoothed_minus_dm / safe_atr

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = (100 * di_diff / di_sum.replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx.fillna(25.0)


def _stoch_rsi(close: pd.Series, period: int = _STOCH_RSI_PERIOD) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, period)
    rsi_min = rsi.rolling(period, min_periods=period).min()
    rsi_max = rsi.rolling(period, min_periods=period).max()
    denom = (rsi_max - rsi_min).replace(0, np.nan)
    stoch = ((rsi - rsi_min) / denom).clip(0.0, 1.0)
    k = stoch.ewm(span=3, adjust=False).mean().fillna(0.5)
    d = k.ewm(span=3, adjust=False).mean().fillna(0.5)
    return k, d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = _WILLIAMS_R_PERIOD) -> pd.Series:
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    wr = -100 * (highest_high - close) / denom
    return wr.fillna(-50.0)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = [0.0]
    c_vals = close.to_numpy()
    v_vals = volume.fillna(0).to_numpy()
    for i in range(1, len(c_vals)):
        if c_vals[i] > c_vals[i - 1]:
            obv.append(obv[-1] + v_vals[i])
        elif c_vals[i] < c_vals[i - 1]:
            obv.append(obv[-1] - v_vals[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = _CMF_PERIOD) -> pd.Series:
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    cmf = mfv.rolling(period, min_periods=period).sum() / volume.rolling(period, min_periods=period).sum()
    return cmf.fillna(0.0)


def _close_vs_sma200_1h_series(
    close: pd.Series,
    timestamp: pd.Series | None,
    base_timeframe_min: int = 15,
) -> pd.Series:
    """Relative distance close/SMA200(1h)-1; 0 when insufficient history."""
    c = close.astype(float)
    if timestamp is not None and len(timestamp):
        idx = pd.DatetimeIndex(pd.to_datetime(timestamp, utc=True))
        sc = pd.Series(c.values, index=idx)
        h_last = sc.resample("1h", label="right", closed="right").last().dropna()
        sma_h = h_last.rolling(_SMA200_1H_PERIODS, min_periods=_SMA200_1H_PERIODS).mean()
        aligned = sma_h.reindex(sc.index, method="ffill")
        raw = c.values / aligned.values - 1.0
        return pd.Series(raw, index=c.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    arr = c.values
    n = len(arr)
    out = np.zeros(n, dtype=float)
    step = max(1, 60 // base_timeframe_min)
    hourly = pd.Series(arr[::step], dtype=float)
    if len(hourly) < _SMA200_1H_PERIODS:
        return pd.Series(out, index=c.index)
    sma200 = hourly.rolling(_SMA200_1H_PERIODS, min_periods=_SMA200_1H_PERIODS).mean()
    for i in range(n):
        hi = i // step
        if hi < _SMA200_1H_PERIODS - 1:
            continue
        sm = float(sma200.iloc[hi])
        if sm > 0:
            out[i] = float(arr[i]) / sm - 1.0
    return pd.Series(out, index=c.index)


# ── Main feature builder ─────────────────────────────────────────────────────

def add_quant_features(
    ohlcv: pd.DataFrame,
    *,
    volume_col: str = "volume",
    base_timeframe_min: int = 15,
) -> pd.DataFrame:
    """Append feature columns to OHLCV frame (expects open, high, low, close)."""
    df = ohlcv.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    ts_col = df["timestamp"] if "timestamp" in df.columns else None

    # Classic
    df["rsi"] = _rsi(close, _RSI_PERIOD)
    m_line, m_sig, m_hist = _macd(close)
    df["macd_line"] = m_line
    df["macd_signal"] = m_sig
    df["macd_hist"] = m_hist
    df["atr"] = _atr(high, low, close, _ATR_PERIOD)

    mid = close.rolling(_BB_PERIOD, min_periods=_BB_PERIOD).mean()
    std = close.rolling(_BB_PERIOD, min_periods=_BB_PERIOD).std()
    upper = mid + _BB_STD * std
    lower = mid - _BB_STD * std
    df["bb_width"] = ((upper - lower) / mid.replace(0, np.nan)).fillna(0.0)
    rng = (upper - lower).replace(0, np.nan)
    df["bb_pct_b"] = ((close - lower) / rng).clip(0.0, 1.0).fillna(0.5)

    vol = df[volume_col].astype(float) if volume_col in df.columns else pd.Series(0.0, index=df.index)
    v_chg = vol.diff()
    v_ma = vol.rolling(_VOL_DELTA_WIN, min_periods=5).mean().replace(0, np.nan)
    df["vol_delta_norm"] = (v_chg / v_ma).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["log_ret_1"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["log_ret_5"] = np.log(close / close.shift(5)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["log_ret_10"] = np.log(close / close.shift(10)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["log_ret_20"] = np.log(close / close.shift(20)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["close_vs_sma200_1h"] = _close_vs_sma200_1h_series(close, ts_col, base_timeframe_min=base_timeframe_min)
    v_ma_rel = vol.rolling(_VOL_REL_WIN, min_periods=5).mean().replace(0, np.nan)
    df["vol_rel"] = (vol / v_ma_rel).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    # New v2 features
    df["adx"] = _adx(high, low, close, _ADX_PERIOD)
    stoch_k, stoch_d = _stoch_rsi(close, _STOCH_RSI_PERIOD)
    df["stoch_rsi_k"] = stoch_k
    df["stoch_rsi_d"] = stoch_d
    df["williams_r"] = _williams_r(high, low, close, _WILLIAMS_R_PERIOD)
    df["obv"] = _obv(close, vol)
    df["cmf"] = _cmf(high, low, close, vol, _CMF_PERIOD)

    returns = df["log_ret_1"]
    df["ret_skew_20"] = returns.rolling(_ROLL_WIN, min_periods=5).skew().fillna(0.0)
    df["ret_kurt_20"] = returns.rolling(_ROLL_WIN, min_periods=5).kurt().fillna(0.0)
    roll_mean = returns.rolling(_ROLL_WIN, min_periods=5).mean()
    roll_std = returns.rolling(_ROLL_WIN, min_periods=5).std().replace(0, np.nan)
    df["roll_sharpe_20"] = (roll_mean / roll_std * np.sqrt(_ROLL_WIN)).fillna(0.0)

    # Temporal features (if timestamp available)
    if ts_col is not None and len(ts_col):
        ts = pd.to_datetime(ts_col, utc=True)
        hour = ts.dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dow"] = ts.dt.dayofweek.astype(float) / 6.0  # normalize Mon=0, Sun=1
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0
        df["dow"] = 0.0

    return df


# ── Live inference vector ────────────────────────────────────────────────────

def compute_quant_vector_from_lists(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None,
    base_timeframe_min: int = 15,
    *,
    now: datetime | None = None,
) -> list[float] | None:
    """Latest feature vector aligned with QUANT_FEATURE_COLS (for live inference)."""
    if len(closes) < MIN_OHLC_ROWS or len(highs) < MIN_OHLC_ROWS or len(lows) < MIN_OHLC_ROWS:
        return None
    n = min(len(closes), len(highs), len(lows))
    if volumes is None or len(volumes) < n:
        vol = [0.0] * n
    else:
        vol = volumes[-n:]
    _now = now if now is not None else datetime.now(tz=timezone.utc)
    step = timedelta(minutes=base_timeframe_min)
    timestamps = [_now - step * (n - 1 - i) for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes[-n:],
            "high": highs[-n:],
            "low": lows[-n:],
            "close": closes[-n:],
            "volume": vol,
        }
    )
    feat = add_quant_features(df, base_timeframe_min=base_timeframe_min)
    row = feat.iloc[-1]
    out = []
    for c in QUANT_FEATURE_COLS:
        v = float(row.get(c, 0.0))
        if not np.isfinite(v):
            v = 0.0
        out.append(v)
    return out


# ── HTF filter ───────────────────────────────────────────────────────────────

def htf_sma200_1h_allows_long(closes: list[float], base_timeframe_min: int = 15) -> bool:
    """Long-only HTF gate: último close por encima de SMA200 en marco 1h."""
    if len(closes) < MIN_OHLC_ROWS:
        return False
    s = _close_vs_sma200_1h_series(pd.Series(closes, dtype=float), None, base_timeframe_min=base_timeframe_min)
    return float(s.iloc[-1]) > 0.0


# ── Labeling ─────────────────────────────────────────────────────────────────

def forward_return_label(
    close: pd.Series,
    horizon: int,
    round_trip_cost: float,
) -> pd.Series:
    """Binary label: forward simple return over *horizon* bars exceeds friction."""
    fwd = close.shift(-horizon) / close - 1.0
    return (fwd > round_trip_cost).astype(int)


def triple_barrier_label(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int,
    sl_mult: float = 1.5,
    tp_mult: float = 2.5,
    volatility: pd.Series | None = None,
    *,
    fixed_sl_pct: float | None = None,
    fixed_tp_pct: float | None = None,
) -> pd.Series:
    """Triple-barrier label: 1 if price hits take-profit before stop-loss within *horizon*.

    Priority: fixed_sl_pct/fixed_tp_pct > ATR-based (volatility) > hard-coded defaults.
    Pass fixed_sl_pct/fixed_tp_pct to align training barriers with live SYMBOL_CONFIG.
    """
    if fixed_sl_pct is not None and fixed_tp_pct is not None:
        sl_pct = fixed_sl_pct
        tp_pct = fixed_tp_pct
        volatility = None  # force scalar path below
    elif volatility is None:
        # Default: 2% SL, 4% TP (approx for crypto 15m)
        sl_pct = 0.02
        tp_pct = 0.04
    else:
        sl_pct = volatility * sl_mult / close
        tp_pct = volatility * tp_mult / close

    label = pd.Series(0, index=close.index, dtype=int)
    for i in range(len(close) - horizon):
        entry = close.iloc[i]
        if volatility is not None:
            sl = entry * (1.0 - sl_pct.iloc[i])
            tp = entry * (1.0 + tp_pct.iloc[i])
        else:
            sl = entry * (1.0 - sl_pct)
            tp = entry * (1.0 + tp_pct)

        future_high = high.iloc[i + 1 : i + 1 + horizon]
        future_low = low.iloc[i + 1 : i + 1 + horizon]

        if future_low.empty:
            continue

        # Check which barrier is hit first
        sl_hit = future_low[future_low <= sl]
        tp_hit = future_high[future_high >= tp]

        if tp_hit.empty and sl_hit.empty:
            # Time exit: label based on final return
            final_ret = close.iloc[i + horizon] / entry - 1.0
            label.iloc[i] = 1 if final_ret > 0 else 0
        elif tp_hit.empty:
            label.iloc[i] = 0
        elif sl_hit.empty:
            label.iloc[i] = 1
        else:
            # Both hit: compare indices
            label.iloc[i] = 1 if tp_hit.index[0] < sl_hit.index[0] else 0

    return label
