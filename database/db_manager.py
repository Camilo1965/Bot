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
from asyncpg.exceptions import UniqueViolationError

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

# Legacy tick-style table (gui/db_reader) may only have best_bid/best_ask; add OHLC for bot kline writes.
_MIGRATE_ADD_MARKET_OHLC = """
DO $$
BEGIN
  IF to_regclass('public.market_data') IS NULL THEN RETURN; END IF;
  IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'open') THEN
    ALTER TABLE market_data ADD COLUMN open DOUBLE PRECISION;
  END IF;
  IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'high') THEN
    ALTER TABLE market_data ADD COLUMN high DOUBLE PRECISION;
  END IF;
  IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'low') THEN
    ALTER TABLE market_data ADD COLUMN low DOUBLE PRECISION;
  END IF;
  IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'close') THEN
    ALTER TABLE market_data ADD COLUMN close DOUBLE PRECISION;
  END IF;
  IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'market_data' AND column_name = 'volume') THEN
    ALTER TABLE market_data ADD COLUMN volume DOUBLE PRECISION;
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

_INSERT_PRED = "INSERT INTO ml_predictions (timestamp, symbol, confidence, side) VALUES ($1, $2, $3, $4);"
_UPSERT_HTF = "INSERT INTO htf_trend (timestamp, symbol, timeframe, trend_status) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING;"

class DatabaseManager:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._market_cols: frozenset[str] | None = None

    async def _refresh_market_columns(self, conn: asyncpg.Connection) -> None:
        rows = await conn.fetch(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'market_data'
            """
        )
        self._market_cols = frozenset(str(r["column_name"]) for r in rows)

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
            try:
                await conn.execute(_MIGRATE_ADD_MARKET_OHLC)
            except Exception:  # noqa: BLE001
                logger.warning("market_data OHLC migration skipped: check Timescale/locks.")
            try: await conn.execute(_CREATE_HYPERTABLE_MARKET)
            except Exception: pass
            await conn.execute(_CREATE_ML_PREDICTIONS)
            try: await conn.execute(_CREATE_HYPERTABLE_PRED)
            except Exception: pass
            await conn.execute(_CREATE_HTF_TREND)
            try: await conn.execute(_CREATE_HYPERTABLE_HTF)
            except Exception: pass
            await self._refresh_market_columns(conn)
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
        """Insert kline row; fills ``best_bid``/``best_ask`` when present (legacy tick NOT NULL)."""
        if not self._pool:
            return
        b = self._to_timestamptz(bucket)
        row: dict[str, Any] = {
            "timestamp": b,
            "symbol": symbol,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            # Quote bracket for the bar — satisfies tick-schema NOT NULL without inventing spread:
            "best_bid": float(low),
            "best_ask": float(high),
        }
        order = (
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "best_bid",
            "best_ask",
        )
        async with self._pool.acquire() as conn:
            if not self._market_cols:
                await self._refresh_market_columns(conn)
            cols = self._market_cols or frozenset()
            keys = [k for k in order if k in cols]
            if "timestamp" not in keys or "symbol" not in keys:
                logger.warning("market_data missing timestamp/symbol — skip insert")
                return
            vals = [row[k] for k in keys]
            placeholders = ", ".join(f"${i + 1}" for i in range(len(keys)))
            stmt = f"INSERT INTO market_data ({', '.join(keys)}) VALUES ({placeholders})"
            try:
                await conn.execute(stmt, *vals)
            except UniqueViolationError:
                pass

    async def fetch_market_data(self, symbol: str, limit: int = 1000) -> list[float]:
        if not self._pool: return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT close FROM market_data WHERE symbol = $1 AND close IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT $2",
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
