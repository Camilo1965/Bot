#!/usr/bin/env python3
"""scripts/backfill_db.py — Backfill TimescaleDB ohlcv table from CCXT.

Usage:
    python scripts/backfill_db.py --symbols BTC/USDT,ETH/USDT --days 365
    python scripts/backfill_db.py --all --days 365
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

logger = logging.getLogger("backfill_db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT",
    "NEAR/USDT", "ATOM/USDT", "LINK/USDT", "XRP/USDT",
]
_DEFAULT_TF = "15m"
_CANDLES_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24}
_BATCH_SIZE = 500


def _build_dsn() -> str:
    return (
        os.environ.get("DATABASE_URL")
        or f"postgresql://{os.environ.get('DB_USER','clawdbot')}:"
           f"{os.environ.get('DB_PASSWORD','clawdbot_secret')}@"
           f"{os.environ.get('DB_HOST','localhost')}:"
           f"{os.environ.get('DB_PORT','5432')}/"
           f"{os.environ.get('DB_NAME','clawdbot')}"
    )


async def backfill_one(symbol: str, tf: str, days: int) -> int:
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed — pip install asyncpg")
        return 0

    limit = days * _CANDLES_PER_DAY.get(tf, 96)
    logger.info("[%s] Fetching %d candles at %s ...", symbol, limit, tf)

    try:
        from data.ohlcv_cache import get_cache
        df = get_cache().fetch(symbol, tf, limit, max_age_s=0)  # force fresh
    except Exception as exc:
        logger.error("[%s] Fetch failed: %s", symbol, exc)
        return 0

    if df is None or len(df) == 0:
        logger.warning("[%s] No data returned", symbol)
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
        logger.warning("[%s] No valid rows to insert", symbol)
        return 0

    dsn = _build_dsn()
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:
        logger.error("DB connection failed: %s", exc)
        return 0

    try:
        inserted = 0
        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i:i + _BATCH_SIZE]
            await conn.executemany(
                """INSERT INTO ohlcv (symbol, ts, open, high, low, close, vol)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)
                   ON CONFLICT (symbol, ts) DO NOTHING""",
                batch,
            )
            inserted += len(batch)
        logger.info("[%s] Inserted %d rows", symbol, inserted)
        return inserted
    finally:
        await conn.close()


async def main_async(symbols: list[str], tf: str, days: int) -> int:
    total = 0
    for sym in symbols:
        n = await backfill_one(sym, tf, days)
        total += n
    logger.info("Backfill complete: %d rows total across %d symbols", total, len(symbols))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tf", type=str, default=_DEFAULT_TF)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    if args.all:
        symbols = _DEFAULT_SYMBOLS
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        parser.error("Pass --all or --symbols")
        return 1

    return asyncio.run(main_async(symbols, args.tf, args.days))


if __name__ == "__main__":
    raise SystemExit(main())
