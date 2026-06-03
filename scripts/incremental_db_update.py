#!/usr/bin/env python3
"""scripts/incremental_db_update.py — Upsert latest N candles into TimescaleDB.

Run hourly via scheduler to keep the DB up to date.

Usage:
    python scripts/incremental_db_update.py
    python scripts/incremental_db_update.py --hours 2 --symbols BTC/USDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

logger = logging.getLogger("incremental_db_update")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
    "NEAR/USDT", "ATOM/USDT", "LINK/USDT", "XRP/USDT",
]
_DEFAULT_TF = "15m"
_CANDLES_PER_HOUR = {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "1h": 1}


def _build_dsn() -> str:
    return (
        os.environ.get("DATABASE_URL")
        or f"postgresql://{os.environ.get('DB_USER','clawdbot')}:"
           f"{os.environ.get('DB_PASSWORD','clawdbot_secret')}@"
           f"{os.environ.get('DB_HOST','localhost')}:"
           f"{os.environ.get('DB_PORT','5432')}/"
           f"{os.environ.get('DB_NAME','clawdbot')}"
    )


async def update_one(symbol: str, tf: str, hours: int) -> int:
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed")
        return 0

    limit = hours * _CANDLES_PER_HOUR.get(tf, 4) + 10  # small buffer

    try:
        from data.ohlcv_cache import get_cache
        df = get_cache().fetch(symbol, tf, limit, max_age_s=0)
    except Exception as exc:
        logger.error("[%s] Fetch failed: %s", symbol, exc)
        return 0

    if df is None or len(df) == 0:
        return 0

    rows = []
    for _, row in df.iterrows():
        try:
            import pandas as pd
            ts = pd.to_datetime(row["timestamp"], utc=True)
            rows.append((
                symbol, ts,
                float(row.get("open", 0) or 0),
                float(row.get("high", 0) or 0),
                float(row.get("low", 0) or 0),
                float(row.get("close", 0) or 0),
                float(row.get("volume", 0) or 0),
            ))
        except Exception:
            continue

    if not rows:
        return 0

    dsn = _build_dsn()
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:
        logger.error("DB connection failed: %s", exc)
        return 0

    try:
        await conn.executemany(
            """INSERT INTO ohlcv (symbol, ts, open, high, low, close, vol)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (symbol, ts) DO UPDATE SET
                 open = EXCLUDED.open, high = EXCLUDED.high,
                 low = EXCLUDED.low, close = EXCLUDED.close, vol = EXCLUDED.vol""",
            rows,
        )
        logger.info("[%s] Upserted %d rows", symbol, len(rows))
        return len(rows)
    finally:
        await conn.close()


async def main_async(symbols: list[str], tf: str, hours: int) -> None:
    tasks = [update_one(sym, tf, hours) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total = sum(r for r in results if isinstance(r, int))
    logger.info("Incremental update done: %d rows upserted", total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--tf", type=str, default=_DEFAULT_TF)
    parser.add_argument("--hours", type=int, default=2, help="Look back this many hours")
    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else _DEFAULT_SYMBOLS
    )
    asyncio.run(main_async(symbols, args.tf, args.hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
