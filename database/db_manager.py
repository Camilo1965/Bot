"""
database.db_manager
~~~~~~~~~~~~~~~~~~~~

Async database manager for TimescaleDB (PostgreSQL) using asyncpg.

Creates two hypertables on first connection:

* **market_data** – L2 order-book top-of-book snapshots
  (timestamp, symbol, best_bid, bid_volume, best_ask, ask_volume)

* **news_sentiment** – scored news headlines
  (timestamp, headline, sentiment_score, source)

Credentials are read from environment variables (populated via .env):

    DB_USER      – PostgreSQL username        (default: clawdbot)
    DB_PASSWORD  – PostgreSQL password        (default: clawdbot_secret)
    DB_HOST      – PostgreSQL host            (default: localhost)
    DB_PORT      – PostgreSQL port            (default: 5432)
    DB_NAME      – PostgreSQL database name   (default: clawdbot)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_MARKET_DATA = """
CREATE TABLE IF NOT EXISTS market_data (
    timestamp   TIMESTAMPTZ        NOT NULL,
    symbol      TEXT               NOT NULL,
    best_bid    DOUBLE PRECISION   NOT NULL,
    bid_volume  DOUBLE PRECISION   NOT NULL,
    best_ask    DOUBLE PRECISION   NOT NULL,
    ask_volume  DOUBLE PRECISION   NOT NULL
);
"""

_CREATE_NEWS_SENTIMENT = """
CREATE TABLE IF NOT EXISTS news_sentiment (
    timestamp       TIMESTAMPTZ      NOT NULL,
    headline        TEXT             NOT NULL,
    sentiment_score DOUBLE PRECISION NOT NULL,
    source          TEXT             NOT NULL
);
"""

_CREATE_HYPERTABLE_MARKET = """
SELECT create_hypertable(
    'market_data', 'timestamp',
    if_not_exists => TRUE
);
"""

_CREATE_HYPERTABLE_SENTIMENT = """
SELECT create_hypertable(
    'news_sentiment', 'timestamp',
    if_not_exists => TRUE
);
"""

_INSERT_MARKET_TICK = """
INSERT INTO market_data
    (timestamp, symbol, best_bid, bid_volume, best_ask, ask_volume)
VALUES ($1, $2, $3, $4, $5, $6);
"""

_INSERT_SENTIMENT = """
INSERT INTO news_sentiment
    (timestamp, headline, sentiment_score, source)
VALUES ($1, $2, $3, $4);
"""

_FETCH_MARKET_DATA = """
SELECT timestamp, best_bid, best_ask
FROM market_data
WHERE symbol = $1
ORDER BY timestamp ASC
LIMIT $2;
"""

_CREATE_TRADES_HISTORY = """
CREATE TABLE IF NOT EXISTS trades_history (
    id            SERIAL             PRIMARY KEY,
    symbol        TEXT               NOT NULL,
    entry_price   DOUBLE PRECISION   NOT NULL,
    position_size DOUBLE PRECISION   NOT NULL,
    entry_time    TIMESTAMPTZ        NOT NULL,
    exit_price    DOUBLE PRECISION,
    exit_time     TIMESTAMPTZ,
    pnl           DOUBLE PRECISION,
    status        TEXT               NOT NULL DEFAULT 'open'
);
"""

_INSERT_OPEN_TRADE = """
INSERT INTO trades_history
    (symbol, entry_price, position_size, entry_time, status)
VALUES ($1, $2, $3, $4, 'open')
RETURNING id;
"""

_CLOSE_TRADE = """
UPDATE trades_history
SET exit_price = $2,
    exit_time  = $3,
    pnl        = $4,
    status     = 'closed'
WHERE id = $1;
"""

_FETCH_TOTAL_PNL = """
SELECT COALESCE(SUM(pnl), 0.0) AS total_pnl
FROM trades_history
WHERE status = 'closed';
"""


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Manages an asyncpg connection pool and provides insert helpers."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection pool and initialise the schema."""
        dsn = self._build_dsn()
        logger.info("Connecting to TimescaleDB at %s", self._redacted_dsn())
        self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
        await self._initialise_schema()
        logger.info("DatabaseManager ready.")

    async def close(self) -> None:
        """Close the connection pool gracefully."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("DatabaseManager connection pool closed.")

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------

    async def insert_market_tick(
        self,
        symbol: str,
        best_bid: float,
        bid_volume: float,
        best_ask: float,
        ask_volume: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Insert one order-book top-of-book snapshot into *market_data*.

        Parameters
        ----------
        symbol:     Trading pair, e.g. ``"BTC/USDT"``.
        best_bid:   Best bid price.
        bid_volume: Volume at best bid.
        best_ask:   Best ask price.
        ask_volume: Volume at best ask.
        timestamp:  Event time (UTC).  Defaults to *now* when ``None``.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        ts = timestamp or datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                _INSERT_MARKET_TICK,
                ts,
                symbol,
                best_bid,
                bid_volume,
                best_ask,
                ask_volume,
            )

    async def insert_sentiment(
        self,
        headline: str,
        sentiment_score: float,
        source: str,
        timestamp: datetime | None = None,
    ) -> None:
        """Insert one scored headline into *news_sentiment*.

        Parameters
        ----------
        headline:        Raw headline text.
        sentiment_score: VADER compound score in ``[-1, +1]``.
        source:          Origin of the headline (e.g. feed URL or name).
        timestamp:       Event time (UTC).  Defaults to *now* when ``None``.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        ts = timestamp or datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                _INSERT_SENTIMENT,
                ts,
                headline,
                sentiment_score,
                source,
            )

    async def fetch_market_data(
        self,
        symbol: str,
        limit: int = 500,
    ) -> list[float]:
        """Fetch the most recent *limit* mid-prices for *symbol*.

        Mid-price is computed as ``(best_bid + best_ask) / 2``.  Results are
        returned in **chronological** (ascending timestamp) order.

        Parameters
        ----------
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        limit:  Maximum number of rows to retrieve.

        Returns
        -------
        A list of mid-price floats, oldest first.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_MARKET_DATA, symbol, limit)
        # Rows come back in ascending timestamp order; return as-is
        return [(float(r["best_bid"]) + float(r["best_ask"])) / 2.0 for r in rows]

    async def insert_open_trade(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        entry_time: datetime | None = None,
    ) -> int:
        """Insert a new open trade into *trades_history* and return its id.

        Parameters
        ----------
        symbol:        Trading pair, e.g. ``"BTC/USDT"``.
        entry_price:   Price at which the trade was entered.
        position_size: Size of the position in quote currency.
        entry_time:    Trade entry time (UTC).  Defaults to *now*.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        ts = entry_time or datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_INSERT_OPEN_TRADE, symbol, entry_price, position_size, ts)
        return int(row["id"])  # type: ignore[index]

    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: datetime | None = None,
        pnl: float = 0.0,
    ) -> None:
        """Update a trade record in *trades_history* to mark it as closed.

        Parameters
        ----------
        trade_id:   Row id of the trade to close.
        exit_price: Price at which the trade was exited.
        exit_time:  Trade exit time (UTC).  Defaults to *now*.
        pnl:        Realised profit / loss in quote currency.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        ts = exit_time or datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(_CLOSE_TRADE, trade_id, exit_price, ts, pnl)

    async def fetch_total_pnl(self) -> float:
        """Return the sum of all realised PnL from closed trades.

        Returns ``0.0`` when no closed trades exist yet.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_FETCH_TOTAL_PNL)
        return float(row["total_pnl"])  # type: ignore[index]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _initialise_schema(self) -> None:
        """Create tables and hypertables if they do not exist yet."""
        assert self._pool is not None  # noqa: S101 – guaranteed by connect()
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_MARKET_DATA)
            await conn.execute(_CREATE_NEWS_SENTIMENT)
            await conn.execute(_CREATE_TRADES_HISTORY)
            await conn.execute(_CREATE_HYPERTABLE_MARKET)
            await conn.execute(_CREATE_HYPERTABLE_SENTIMENT)
        logger.info("TimescaleDB schema initialised.")

    @staticmethod
    def _build_dsn() -> str:
        """Construct the asyncpg DSN from environment variables."""
        user = os.environ.get("DB_USER", "clawdbot")
        password = os.environ.get("DB_PASSWORD", "clawdbot_secret")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME", "clawdbot")
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    @staticmethod
    def _redacted_dsn() -> str:
        """Return a DSN with the password replaced by ``***`` for logging."""
        user = os.environ.get("DB_USER", "clawdbot")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        name = os.environ.get("DB_NAME", "clawdbot")
        return f"postgresql://{user}:***@{host}:{port}/{name}"


# ---------------------------------------------------------------------------
# Module-level convenience instance (imported by main.py)
# ---------------------------------------------------------------------------

db: DatabaseManager = DatabaseManager()


async def init_db() -> None:
    """Open the global :data:`db` connection pool."""
    await db.connect()


async def close_db() -> None:
    """Close the global :data:`db` connection pool."""
    await db.close()
