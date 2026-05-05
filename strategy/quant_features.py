"""
Pure quantitative OHLCV features for XGB sweep + live inference.

RSI, MACD, ATR, Bollinger Bands, normalized volume delta, log returns.
No sentiment / NLP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Must match training column order for XGBoost
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
]

_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_ATR_PERIOD = 14
_BB_PERIOD = 20
_BB_STD = 2.0
_VOL_DELTA_WIN = 20

# Warm-up: MACD slow + signal smoothing
MIN_OHLC_ROWS = 60

# Friction model (aligned with execution.paper_executor._TAKER_FEE_RATE ≈ 0.0004)
DEFAULT_TAKER_FEE_RATE: float = 0.0004
DEFAULT_LABEL_ROUND_TRIP: float = 2.0 * DEFAULT_TAKER_FEE_RATE + 0.0005


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


def add_quant_features(
    ohlcv: pd.DataFrame,
    *,
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Append feature columns to OHLCV frame (expects open, high, low, close)."""
    df = ohlcv.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

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

    return df


def compute_quant_vector_from_lists(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float] | None,
) -> list[float] | None:
    """Latest feature vector aligned with QUANT_FEATURE_COLS (for live inference)."""
    if len(closes) < MIN_OHLC_ROWS or len(highs) < MIN_OHLC_ROWS or len(lows) < MIN_OHLC_ROWS:
        return None
    n = min(len(closes), len(highs), len(lows))
    if volumes is None or len(volumes) < n:
        vol = [0.0] * n
    else:
        vol = volumes[-n:]
    df = pd.DataFrame(
        {
            "open": closes[-n:],  # unused but keeps shape
            "high": highs[-n:],
            "low": lows[-n:],
            "close": closes[-n:],
            "volume": vol,
        }
    )
    feat = add_quant_features(df)
    row = feat.iloc[-1]
    out = []
    for c in QUANT_FEATURE_COLS:
        v = float(row.get(c, 0.0))
        if not np.isfinite(v):
            v = 0.0
        out.append(v)
    return out


def forward_return_label(
    close: pd.Series,
    horizon: int,
    round_trip_cost: float,
) -> pd.Series:
    """Binary label: forward simple return over *horizon* bars exceeds friction."""
    fwd = close.shift(-horizon) / close - 1.0
    return (fwd > round_trip_cost).astype(int)
