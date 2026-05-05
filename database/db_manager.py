"""
database.db_manager
~~~~~~~~~~~~~~~~~~~

TimescaleDB persistence manager for market data, trades, and ML telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

# Column name matches gui/db_reader queries (`timestamp` in table; aggregation aliases AS bucket).
_CREATE_MARKET_DATA = """
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);
"""

_CREATE_HYPERTABLE_MARKET = (
    "SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);"
)

# Older bot builds used `bucket`; rename once so INSERT matches deployed DBs.
_MIGRATE_MARKET_BUCKET_TO_TS = """
DO $$
BEGIN
  IF to_regclass('public.market_data') IS NOT NULL THEN
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'bucket')
       AND NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'timestamp') THEN
      ALTER TABLE market_data RENAME COLUMN bucket TO timestamp;
    END IF;
  END IF;
END $$;
"""

_CREATE_ML_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS ml_predictions (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    side TEXT NOT NULL
);
"""

_CREATE_HYPERTABLE_PRED = "SELECT create_hypertable('ml_predictions', 'timestamp', if_not_exists => TRUE);"

_CREATE_HTF_TREND = """
CREATE TABLE IF NOT EXISTS htf_trend (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    trend_status TEXT NOT NULL
);
"""

_CREATE_HYPERTABLE_HTF = "SELECT create_hypertable('htf_trend', 'timestamp', if_not_exists => TRUE);"

_INSERT_MARKET = "INSERT INTO market_data (timestamp, symbol, open, high, low, close, volume) VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT DO NOTHING;"
_INSERT_PRED = "INSERT INTO ml_predictions (timestamp, symbol, confidence, side) VALUES ($1, $2, $3, $4);"
_UPSERT_HTF = "INSERT INTO htf_trend (timestamp, symbol, timeframe, trend_status) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING;"

class DatabaseManager:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        dsn = os.environ.get("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/postgres")
        self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        logger.info("Connected to TimescaleDB.")

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("Disconnected from TimescaleDB.")

    async def initialize(self) -> None:
        if not self._pool: return
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(_MIGRATE_MARKET_BUCKET_TO_TS)
            except Exception:  # noqa: BLE001
                pass
            await conn.execute(_CREATE_MARKET_DATA)
            try: await conn.execute(_CREATE_HYPERTABLE_MARKET)
            except Exception: pass
            await conn.execute(_CREATE_ML_PREDICTIONS)
            try: await conn.execute(_CREATE_HYPERTABLE_PRED)
            except Exception: pass
            await conn.execute(_CREATE_HTF_TREND)
            try: await conn.execute(_CREATE_HYPERTABLE_HTF)
            except Exception: pass
        logger.info("Database schema initialised.")

    @staticmethod
    def _to_timestamptz(ts: datetime | int | float) -> datetime:
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
        t = float(ts)
        if t > 1e12:  # ms since epoch (MT5 kline queue)
            t /= 1000.0
        return datetime.fromtimestamp(t, tz=timezone.utc)

    async def insert_market_data(
        self,
        bucket: datetime | int | float,
        symbol: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        if not self._pool: return
        b = self._to_timestamptz(bucket)
        async with self._pool.acquire() as conn:
            await conn.execute(_INSERT_MARKET, b, symbol, open, high, low, close, volume)

    async def fetch_market_data(self, symbol: str, limit: int = 1000) -> list[float]:
        if not self._pool: return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT close FROM market_data WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2",
                symbol,
                limit,
            )
            return [float(r["close"]) for r in reversed(rows)]

    async def insert_ml_prediction(self, symbol: str, confidence: float, side: str) -> None:
        if not self._pool: return
        ts = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(_INSERT_PRED, ts, symbol, confidence, side)

    async def upsert_htf_trend(self, symbol: str, timeframe: str, trend_status: str) -> None:
        if not self._pool: return
        ts = datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(_UPSERT_HTF, ts, symbol, timeframe, trend_status)

db = DatabaseManager()
async def init_db() -> None: await db.connect(); await db.initialize()
async def close_db() -> None: await db.disconnect()
