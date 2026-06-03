"""data/funding_rates.py — Binance Futures funding rate fetcher + feature engineering.

Funding rate > 0 → longs paying shorts (market overbought, short signal).
Funding rate < 0 → shorts paying longs (market oversold, long signal).

Features computed:
  funding_rate      — raw 8h rate (fraction, e.g. 0.0001 = 0.01%)
  funding_rate_ma3  — 3-period (24h) EMA of funding rate
  funding_extreme   — |rate| > 0.0003 (0.03%) → crowd positioning extreme
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("funding_rates")

_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "funding"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_EXTREME_THRESHOLD = 0.0003  # 0.03% per 8h ≈ annualized ~13%


def fetch_funding_rates(symbol: str, limit: int = 1000) -> pd.DataFrame | None:
    """Fetch historical funding rates from Binance Futures via CCXT.

    Returns DataFrame with columns: timestamp (UTC datetime), funding_rate.
    """
    try:
        import ccxt
        exchange = ccxt.binanceusdm({"options": {"defaultType": "future"}})
        # symbol format: BTC/USDT -> BTC/USDT:USDT for perpetual
        perp = symbol.replace("/USDT", "/USDT:USDT")
        rows = exchange.fetch_funding_rate_history(perp, limit=limit)
        if not rows:
            return None
        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                "funding_rate": float(r["fundingRate"]),
            }
            for r in rows
        ])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("[%s] funding rate fetch failed: %s", symbol, exc)
        return None


def add_funding_features(ohlcv_df: pd.DataFrame, symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Merge funding rate features into ohlcv_df (in-place copy).

    If funding data unavailable, columns are filled with 0 (neutral).
    Columns added: funding_rate, funding_rate_ma3, funding_extreme.
    """
    df = ohlcv_df.copy()
    ts_col = "timestamp"

    neutral = {
        "funding_rate": 0.0,
        "funding_rate_ma3": 0.0,
        "funding_extreme": 0.0,
    }

    if ts_col not in df.columns:
        for col, val in neutral.items():
            df[col] = val
        return df

    funding = fetch_funding_rates(symbol, limit=limit)
    if funding is None or len(funding) == 0:
        logger.warning("[%s] no funding data — filling neutral", symbol)
        for col, val in neutral.items():
            df[col] = val
        return df

    # Compute features on funding series
    funding["funding_rate_ma3"] = funding["funding_rate"].ewm(span=3, adjust=False).mean()
    funding["funding_extreme"] = (funding["funding_rate"].abs() > _EXTREME_THRESHOLD).astype(float)

    # Merge: forward-fill funding rate onto OHLCV timestamps
    df_ts = pd.to_datetime(df[ts_col], utc=True)
    fund_ts = funding["timestamp"].values
    fund_rate = funding["funding_rate"].values
    fund_ma3 = funding["funding_rate_ma3"].values
    fund_ext = funding["funding_extreme"].values

    # For each bar, find the last funding rate before or at that timestamp
    idxs = np.searchsorted(fund_ts, df_ts.values, side="right") - 1
    valid = idxs >= 0

    df["funding_rate"] = 0.0
    df["funding_rate_ma3"] = 0.0
    df["funding_extreme"] = 0.0
    df.loc[valid, "funding_rate"] = fund_rate[idxs[valid]]
    df.loc[valid, "funding_rate_ma3"] = fund_ma3[idxs[valid]]
    df.loc[valid, "funding_extreme"] = fund_ext[idxs[valid]]

    logger.info("[%s] funding features merged (%d bars, %.1f%% coverage)",
                symbol, len(df), valid.mean() * 100)
    return df


# ── Feature column names for use in QUANT_FEATURE_COLS extension ─────────────
FUNDING_FEATURE_COLS: list[str] = [
    "funding_rate",
    "funding_rate_ma3",
    "funding_extreme",
]
