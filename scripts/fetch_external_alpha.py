#!/usr/bin/env python3
"""
scripts/fetch_external_alpha.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Descarga datos de alpha externa de Binance Futures para ETH/USDT:
- Funding rate (cada 8h)
- Open Interest (cada 5m o 1h)
- Long/Short Account Ratio (cada 5m)

Guarda en CSV para análisis y entrenamiento.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt

BASE_URL = "https://fapi.binance.com"
SYMBOL = "ETHUSDT"
MAX_REQUESTS_PER_MINUTE = 1100  # conservative


def fetch_funding_rate(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch funding rate history. Each entry is every 8 hours."""
    all_data = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "startTime": current,
            "endTime": min(current + 30 * 24 * 3600 * 1000, end_ms),  # 30 days per request
            "limit": 1000,
        }
        r = requests.get(f"{BASE_URL}/fapi/v1/fundingRate", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1]["fundingTime"] + 1
        time.sleep(0.1)
    
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "fundingRate"]]


def fetch_open_interest(start_ms: int, end_ms: int, period: str = "5m") -> pd.DataFrame:
    """Fetch open interest history. period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d"""
    all_data = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "period": period,
            "startTime": current,
            "endTime": min(current + 7 * 24 * 3600 * 1000, end_ms),  # 7 days per request
            "limit": 500,
        }
        r = requests.get(f"{BASE_URL}/futures/data/openInterestHist", params=params, timeout=30)
        if r.status_code != 200:
            print(f"  OI fetch failed: {r.status_code} {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1]["timestamp"] + 1
        time.sleep(0.2)
    
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]


def fetch_long_short_ratio(start_ms: int, end_ms: int, period: str = "5m") -> pd.DataFrame:
    """Fetch long/short account ratio."""
    all_data = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "period": period,
            "startTime": current,
            "endTime": min(current + 7 * 24 * 3600 * 1000, end_ms),
            "limit": 500,
        }
        r = requests.get(f"{BASE_URL}/futures/data/globalLongShortAccountRatio", params=params, timeout=30)
        if r.status_code != 200:
            print(f"  LS ratio fetch failed: {r.status_code} {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1]["timestamp"] + 1
        time.sleep(0.2)
    
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["longAccount"] = df["longAccount"].astype(float)
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df["shortAccount"] = df["shortAccount"].astype(float)
    return df[["timestamp", "longAccount", "longShortRatio", "shortAccount"]]


def fetch_taker_volume(start_ms: int, end_ms: int, period: str = "5m") -> pd.DataFrame:
    """Fetch taker buy/sell volume."""
    all_data = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "period": period,
            "startTime": current,
            "endTime": min(current + 7 * 24 * 3600 * 1000, end_ms),
            "limit": 500,
        }
        r = requests.get(f"{BASE_URL}/futures/data/takerlongshortRatio", params=params, timeout=30)
        if r.status_code != 200:
            print(f"  Taker vol fetch failed: {r.status_code} {r.text[:200]}")
            break
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1]["timestamp"] + 1
        time.sleep(0.2)
    
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["buyRatio"] = df["buyRatio"].astype(float)
    df["sellRatio"] = df["sellRatio"].astype(float)
    return df[["timestamp", "buyRatio", "sellRatio"]]


def align_to_ohlcv(alpha_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill alpha data to match OHLCV 15m timestamps (no lookahead)."""
    ohlcv_df = ohlcv_df.copy()
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
    
    alpha_df = alpha_df.sort_values("timestamp").set_index("timestamp")
    ohlcv_df = ohlcv_df.sort_values("timestamp").set_index("timestamp")
    
    # Reindex alpha to OHLCV timestamps using backward fill (each 15m bar sees latest known alpha)
    aligned = alpha_df.reindex(ohlcv_df.index, method="ffill")
    return aligned.reset_index()


def main():
    # Fetch 100k 15m candles to determine date range
    print("Fetching OHLCV to determine date range...")
    ohlcv = fetch_ohlcv_ccxt("ETH/USDT", timeframe="15m", limit=100_000)
    start_dt = pd.to_datetime(ohlcv["timestamp"]).min()
    end_dt = pd.to_datetime(ohlcv["timestamp"]).max()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    print(f"Date range: {start_dt} to {end_dt}")
    
    # Fetch alpha data
    print("\nFetching funding rates...")
    funding = fetch_funding_rate(start_ms, end_ms)
    print(f"  Got {len(funding)} funding records")
    
    print("\nFetching open interest (5m)...")
    oi = fetch_open_interest(start_ms, end_ms, period="5m")
    print(f"  Got {len(oi)} OI records")
    
    print("\nFetching long/short ratio (5m)...")
    ls = fetch_long_short_ratio(start_ms, end_ms, period="5m")
    print(f"  Got {len(ls)} LS ratio records")
    
    print("\nFetching taker volume ratio (5m)...")
    taker = fetch_taker_volume(start_ms, end_ms, period="5m")
    print(f"  Got {len(taker)} taker records")
    
    # Align to OHLCV
    print("\nAligning alpha data to OHLCV 15m...")
    funding_aligned = align_to_ohlcv(funding, ohlcv)
    oi_aligned = align_to_ohlcv(oi, ohlcv)
    ls_aligned = align_to_ohlcv(ls, ohlcv)
    taker_aligned = align_to_ohlcv(taker, ohlcv)
    
    # Merge
    merged = ohlcv.copy()
    merged = merged.merge(funding_aligned, on="timestamp", how="left")
    merged = merged.merge(oi_aligned, on="timestamp", how="left")
    merged = merged.merge(ls_aligned, on="timestamp", how="left")
    merged = merged.merge(taker_aligned, on="timestamp", how="left")
    
    # Forward fill missing values (alpha data may have gaps)
    for col in ["fundingRate", "sumOpenInterest", "sumOpenInterestValue", "longAccount", "longShortRatio", "buyRatio", "sellRatio"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    
    # Save
    out_path = Path("data/alpha_external_eth.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"Total rows: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
