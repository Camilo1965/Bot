"""data/multi_tf_fetcher.py — fetch multiple timeframes for MTF feature injection."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_TF_TO_MIN: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "8h": 480, "1d": 1440,
}

_CANDLES_PER_DAY: dict[str, int] = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24, "2h": 12, "4h": 6, "8h": 3, "1d": 1,
}


def fetch_multi_tf(
    symbol: str,
    tfs: list[str],
    days: int = 60,
    max_age_s: int = 300,
) -> dict[str, pd.DataFrame]:
    """Return {tf: ohlcv_df} for each tf in *tfs*, using DuckDB cache."""
    from data.ohlcv_cache import get_cache

    cache = get_cache()
    result: dict[str, pd.DataFrame] = {}
    for tf in tfs:
        cpd = _CANDLES_PER_DAY.get(tf, 24)
        limit = days * cpd
        df = cache.fetch(symbol, tf, limit, max_age_s=max_age_s)
        if df is not None and len(df) >= 5:
            result[tf] = df
    return result


def resample_to_higher_tf(base_df: pd.DataFrame, target_tf: str) -> pd.DataFrame | None:
    """Resample a base OHLCV DataFrame (with 'timestamp' col) to *target_tf*.

    Returns resampled DataFrame indexed by UTC timestamp, or None on failure.
    """
    if "timestamp" not in base_df.columns:
        return None
    ts = pd.to_datetime(base_df["timestamp"], utc=True, errors="coerce")
    tmp = base_df.copy()
    tmp.index = ts
    rule = _tf_to_pandas_rule(target_tf)
    agg: dict[str, str] = {}
    for col, how in [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last"), ("volume", "sum")]:
        if col in tmp.columns:
            agg[col] = how
    if not agg:
        return None
    resampled = tmp.resample(rule).agg(agg).dropna(subset=["close"])
    if len(resampled) < 5:
        return None
    resampled.index.name = "timestamp"
    return resampled


def _tf_to_pandas_rule(tf: str) -> str:
    """Convert exchange TF string to pandas resample rule."""
    mapping = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D",
    }
    return mapping.get(tf, tf)
