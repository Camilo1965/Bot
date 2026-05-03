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
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

import asyncio
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
ORDER BY timestamp DESC
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

_ALTER_TRADES_HISTORY_METRICS = """
ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS pnl_net DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS commission DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS swap DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS fee DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS exit_reason TEXT,
ADD COLUMN IF NOT EXISTS mt5_ticket BIGINT,
ADD COLUMN IF NOT EXISTS sl_price DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS tp_price DOUBLE PRECISION;
"""

_CREATE_COMMANDS = """
CREATE TABLE IF NOT EXISTS commands (
    id         SERIAL       PRIMARY KEY,
    command    TEXT         NOT NULL,
    issued_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    consumed   BOOLEAN      NOT NULL DEFAULT FALSE
);
"""

_FETCH_PENDING_COMMANDS = """
SELECT id, command FROM commands
WHERE consumed = FALSE
ORDER BY issued_at ASC;
"""

_MARK_COMMAND_CONSUMED = """
UPDATE commands SET consumed = TRUE WHERE id = $1;
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
    pnl_net    = $5,
    commission = $6,
    swap       = $7,
    fee        = $8,
    exit_reason = $9,
    status     = 'closed'
WHERE id = $1;
"""

_FETCH_TOTAL_PNL = """
SELECT COALESCE(SUM(pnl), 0.0) AS total_pnl
FROM trades_history
WHERE status = 'closed';
"""

_FETCH_DAILY_STATS = """
SELECT 
    COUNT(*) as total_trades,
    COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
    COALESCE(SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
    COALESCE(SUM(pnl), 0.0) as daily_pnl
FROM trades_history
WHERE status = 'closed' AND exit_time >= CURRENT_DATE;
"""

_FETCH_PERIOD_SUMMARY = """
SELECT
    COUNT(*) AS total_trades,
    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0.0) > 0 THEN 1 ELSE 0 END), 0) AS wins,
    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0.0) <= 0 THEN 1 ELSE 0 END), 0) AS losses,
    COALESCE(SUM(COALESCE(pnl_net, pnl, 0.0)), 0.0) AS pnl_total,
    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0.0) > 0 THEN COALESCE(pnl_net, pnl, 0.0) ELSE 0 END), 0.0) AS gross_profit,
    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0.0) < 0 THEN ABS(COALESCE(pnl_net, pnl, 0.0)) ELSE 0 END), 0.0) AS gross_loss
FROM trades_history
WHERE status = 'closed'
  AND exit_time >= $1
  AND exit_time < $2;
"""

_FETCH_BEST_WORST_TRADE = """
SELECT id, symbol, COALESCE(pnl_net, pnl, 0.0) AS pnl, exit_time
FROM trades_history
WHERE status = 'closed'
  AND exit_time >= $1
  AND exit_time < $2
ORDER BY pnl DESC
LIMIT 1;
"""

_FETCH_WORST_TRADE = """
SELECT id, symbol, COALESCE(pnl_net, pnl, 0.0) AS pnl, exit_time
FROM trades_history
WHERE status = 'closed'
  AND exit_time >= $1
  AND exit_time < $2
ORDER BY pnl ASC
LIMIT 1;
"""

_FETCH_SYMBOL_PERFORMANCE = """
SELECT
    symbol,
    COUNT(*) AS total_trades,
    COALESCE(SUM(CASE WHEN COALESCE(pnl_net, pnl, 0.0) > 0 THEN 1 ELSE 0 END), 0) AS wins,
    COALESCE(SUM(COALESCE(pnl_net, pnl, 0.0)), 0.0) AS pnl_total
FROM trades_history
WHERE status = 'closed'
  AND exit_time >= $1
  AND exit_time < $2
GROUP BY symbol
ORDER BY pnl_total DESC;
"""

_FETCH_DAILY_PNL_SERIES = """
SELECT
    DATE_TRUNC('day', exit_time) AS day_utc,
    COALESCE(SUM(COALESCE(pnl_net, pnl, 0.0)), 0.0) AS pnl_total
FROM trades_history
WHERE status = 'closed'
  AND exit_time >= $1
  AND exit_time < $2
GROUP BY 1
ORDER BY 1 ASC;
"""

_FETCH_RECENT_CLOSED_TRADES = """
SELECT
    id,
    symbol,
    entry_price,
    exit_price,
    pnl,
    pnl_net,
    commission,
    swap,
    fee,
    exit_time
FROM trades_history
WHERE status = 'closed'
ORDER BY exit_time DESC
LIMIT $1;
"""

_CREATE_HTF_TREND_STATUS = """
CREATE TABLE IF NOT EXISTS htf_trend_status (
    symbol      TEXT         NOT NULL,
    timeframe   TEXT         NOT NULL,
    trend       TEXT         NOT NULL DEFAULT 'neutral',
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe)
);
"""

_UPSERT_HTF_TREND = """
INSERT INTO htf_trend_status (symbol, timeframe, trend, updated_at)
VALUES ($1, $2, $3, NOW())
ON CONFLICT (symbol, timeframe) DO UPDATE
SET trend = EXCLUDED.trend, updated_at = NOW();
"""

_FETCH_HTF_TRENDS = """
SELECT symbol, timeframe, trend
FROM htf_trend_status;
"""

_CREATE_HEALTH_EVENTS = """
CREATE TABLE IF NOT EXISTS health_events (
    timestamp    TIMESTAMPTZ      NOT NULL,
    level        TEXT             NOT NULL,
    component    TEXT             NOT NULL,
    event_code   TEXT             NOT NULL,
    message      TEXT             NOT NULL,
    payload_json TEXT             NOT NULL DEFAULT '{}'
);
"""

_INSERT_HEALTH_EVENT = """
INSERT INTO health_events
    (timestamp, level, component, event_code, message, payload_json)
VALUES ($1, $2, $3, $4, $5, $6);
"""


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Manages an asyncpg connection pool and provides insert helpers."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        # Background batching for market data
        self._market_tick_buffer: list[tuple] = []
        self._batch_size_threshold: int = 100
        self._flush_interval: float = 1.0  # seconds
        self._last_flush_time: float = time.monotonic()
        self._batcher_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the connection pool and initialise the schema."""
        dsn = self._build_dsn()
        logger.info("Connecting to TimescaleDB at %s", self._redacted_dsn())
        self._pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=20)
        await self._initialise_schema()
        # Start background batcher
        self._batcher_task = asyncio.create_task(self._background_batcher())
        logger.info("DatabaseManager ready.")

    async def close(self) -> None:
        """Close the connection pool gracefully."""
        if self._batcher_task:
            self._batcher_task.cancel()
            try:
                await self._batcher_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush_market_ticks()
        
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("DatabaseManager connection pool closed.")

    async def _background_batcher(self) -> None:
        """Background task that flushes the market tick buffer periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_market_ticks()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[DB] Market batcher background error: %s", exc)

    async def _flush_market_ticks(self) -> None:
        """Execute a bulk insert of all buffered market ticks."""
        if not self._market_tick_buffer or self._pool is None:
            return
        
        async with self._pool.acquire() as conn:
            batch = list(self._market_tick_buffer)
            self._market_tick_buffer.clear()
            try:
                # executemany is much faster for batch inserts
                await conn.executemany(_INSERT_MARKET_TICK, batch)
                self._last_flush_time = time.monotonic()
            except Exception as exc:
                logger.error("[DB] Bulk insert market ticks failed: %s. Dropping %d records.", exc, len(batch))

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
        """Buffer one order-book top-of-book snapshot into the batch queue.
        
        This method is now NON-BLOCKING for the database write.
        """
        ts = timestamp or datetime.now(tz=timezone.utc)
        self._market_tick_buffer.append((
            ts, symbol, best_bid, bid_volume, best_ask, ask_volume
        ))
        
        # Immediate background flush if buffer is full
        if len(self._market_tick_buffer) >= self._batch_size_threshold:
            asyncio.create_task(self._flush_market_ticks())

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
        # Query returns newest-first (DESC) for efficient LIMIT; reverse so callers
        # receive chronological order (oldest -> newest).
        rows_chrono = list(reversed(rows))
        return [(float(r["best_bid"]) + float(r["best_ask"])) / 2.0 for r in rows_chrono]

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
        pnl_net: float | None = None,
        commission: float = 0.0,
        swap: float = 0.0,
        fee: float = 0.0,
        exit_reason: str = "",
    ) -> None:
        """Update a trade record in *trades_history* to mark it as closed.

        Parameters
        ----------
        trade_id:    Row id of the trade to close.
        exit_price:  Price at which the trade was exited.
        exit_time:   Trade exit time (UTC).  Defaults to *now*.
        pnl:         Realised profit / loss in quote currency.
        exit_reason: Exit reason code (e.g. SL_BASE, TRAILING_STOP, SMART_EXIT_ML).
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        ts = exit_time or datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                _CLOSE_TRADE,
                trade_id,
                exit_price,
                ts,
                pnl,
                pnl if pnl_net is None else pnl_net,
                commission,
                swap,
                fee,
                exit_reason,
            )

    async def upsert_htf_trend(
        self,
        symbol: str,
        timeframe: str,
        trend: str,
    ) -> None:
        """Insert or update the HTF trend status for a symbol/timeframe pair.

        Parameters
        ----------
        symbol:    Trading pair, e.g. ``"BTC/USDT"``.
        timeframe: Candle timeframe, e.g. ``"4h"``, ``"1h"``, ``"15m"``.
        trend:     Trend direction: ``"bullish"``, ``"bearish"``, or ``"neutral"``.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            await conn.execute(_UPSERT_HTF_TREND, symbol, timeframe, trend)

    async def fetch_htf_trends(self) -> dict[str, dict[str, str]]:
        """Return all HTF trend statuses keyed by ``symbol → timeframe → trend``.

        Returns an empty dict when no entries exist yet.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_HTF_TRENDS)
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            sym = row["symbol"]
            tf = row["timeframe"]
            trend = row["trend"]
            result.setdefault(sym, {})[tf] = trend
        return result

    async def fetch_total_pnl(self) -> float:
        """Return the sum of all realised PnL from closed trades.

        Returns ``0.0`` when no closed trades exist yet.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_FETCH_TOTAL_PNL)
        return float(row["total_pnl"])  # type: ignore[index]

    async def fetch_daily_stats(self) -> dict[str, Any]:
        """Return today's trading statistics (UTC date)."""
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_FETCH_DAILY_STATS)
        if row is None:
            return {"total_trades": 0, "wins": 0, "losses": 0, "daily_pnl": 0.0}
        return {
            "total_trades": int(row["total_trades"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "daily_pnl": float(row["daily_pnl"]),
        }

    @staticmethod
    def _tz_now(tz_name: str) -> datetime:
        """Return timezone-aware now() for the provided IANA timezone."""
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")
        return datetime.now(tz=tz)

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """Normalize a timezone-aware datetime to UTC."""
        return dt.astimezone(timezone.utc)

    def _period_bounds(
        self,
        period: str,
        tz_name: str,
    ) -> tuple[datetime, datetime]:
        """Compute [start, end) bounds for day/week/month in UTC."""
        local_now = self._tz_now(tz_name)
        if period == "day":
            start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_local = start_local + timedelta(days=1)
        elif period == "week":
            start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_local = start_local - timedelta(days=start_local.weekday())
            end_local = start_local + timedelta(days=7)
        elif period == "month":
            start_local = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start_local.month == 12:
                end_local = start_local.replace(year=start_local.year + 1, month=1, day=1)
            else:
                end_local = start_local.replace(month=start_local.month + 1, day=1)
        else:
            raise ValueError(f"Unsupported period: {period}")
        return self._to_utc(start_local), self._to_utc(end_local)

    async def fetch_period_summary(self, period: str, tz_name: str = "America/Bogota") -> dict[str, Any]:
        """Return aggregate summary for the current day/week/month in timezone."""
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        start_utc, end_utc = self._period_bounds(period, tz_name)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_FETCH_PERIOD_SUMMARY, start_utc, end_utc)
            best = await conn.fetchrow(_FETCH_BEST_WORST_TRADE, start_utc, end_utc)
            worst = await conn.fetchrow(_FETCH_WORST_TRADE, start_utc, end_utc)
        total_trades = int(row["total_trades"]) if row else 0
        wins = int(row["wins"]) if row else 0
        losses = int(row["losses"]) if row else 0
        pnl_total = float(row["pnl_total"]) if row else 0.0
        gross_profit = float(row["gross_profit"]) if row else 0.0
        gross_loss = float(row["gross_loss"]) if row else 0.0
        winrate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        return {
            "period": period,
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "pnl_total": pnl_total,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "winrate": winrate,
            "profit_factor": profit_factor,
            "best_trade": dict(best) if best else None,
            "worst_trade": dict(worst) if worst else None,
        }

    async def fetch_symbol_performance(
        self,
        period: str,
        tz_name: str = "America/Bogota",
    ) -> list[dict[str, Any]]:
        """Return PnL and winrate breakdown by symbol for period."""
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        start_utc, end_utc = self._period_bounds(period, tz_name)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_SYMBOL_PERFORMANCE, start_utc, end_utc)
        result: list[dict[str, Any]] = []
        for row in rows:
            total_trades = int(row["total_trades"])
            wins = int(row["wins"])
            result.append(
                {
                    "symbol": str(row["symbol"]),
                    "total_trades": total_trades,
                    "wins": wins,
                    "winrate": (wins / total_trades * 100.0) if total_trades > 0 else 0.0,
                    "pnl_total": float(row["pnl_total"]),
                }
            )
        return result

    async def fetch_daily_pnl_series(
        self,
        days: int = 30,
        tz_name: str = "America/Bogota",
    ) -> list[dict[str, Any]]:
        """Return daily aggregated PnL for the last N days."""
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        if days <= 0:
            return []
        local_now = self._tz_now(tz_name)
        start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_local = start_local - timedelta(days=(days - 1))
        end_local = start_local + timedelta(days=days)
        start_utc = self._to_utc(start_local)
        end_utc = self._to_utc(end_local)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_DAILY_PNL_SERIES, start_utc, end_utc)
        return [{"day_utc": r["day_utc"].isoformat(), "pnl_total": float(r["pnl_total"])} for r in rows]

    async def fetch_recent_closed_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return most recent closed trades for dashboard and Telegram history."""
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        safe_limit = max(1, min(limit, 100))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_RECENT_CLOSED_TRADES, safe_limit)
        return [dict(r) for r in rows]

    async def insert_health_event(
        self,
        level: str,
        component: str,
        event_code: str,
        message: str,
        payload_json: str = "{}",
        timestamp: datetime | None = None,
    ) -> None:
        """Insert one operational/health event row.

        Parameters
        ----------
        level:
            Severity (e.g. ``INFO``, ``WARNING``, ``ERROR``).
        component:
            Subsystem name (e.g. ``health_monitor``, ``reconciler``).
        event_code:
            Stable short code for alert type.
        message:
            Human-readable description.
        payload_json:
            JSON string with extra metadata.
        timestamp:
            Event time in UTC. Defaults to now.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        ts = timestamp or datetime.now(tz=timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                _INSERT_HEALTH_EVENT,
                ts,
                level,
                component,
                event_code,
                message,
                payload_json,
            )

    async def fetch_pending_commands(self) -> list[dict]:
        """Return all unread rows from the *commands* table.

        Each dict contains ``id`` (int) and ``command`` (str).  The caller is
        responsible for marking consumed rows via :meth:`mark_command_consumed`.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_FETCH_PENDING_COMMANDS)
        return [{"id": r["id"], "command": r["command"]} for r in rows]

    async def mark_command_consumed(self, command_id: int) -> None:
        """Mark a command row as consumed so it is not processed again.

        Parameters
        ----------
        command_id: Primary key of the row in the *commands* table.
        """
        if self._pool is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            await conn.execute(_MARK_COMMAND_CONSUMED, command_id)

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
            await conn.execute(_ALTER_TRADES_HISTORY_METRICS)
            await conn.execute(_CREATE_COMMANDS)
            await conn.execute(_CREATE_HTF_TREND_STATUS)
            await conn.execute(_CREATE_HEALTH_EVENTS)
            
            # Intentar crear hypertables (TimescaleDB), pero no fallar si no existe
            try:
                await conn.execute(_CREATE_HYPERTABLE_MARKET)
                await conn.execute(_CREATE_HYPERTABLE_SENTIMENT)
                logger.info("TimescaleDB hypertables created/verified.")
            except Exception as exc:
                if "does not exist" in str(exc):
                    logger.warning("TimescaleDB not detected – falling back to standard PostgreSQL tables.")
                else:
                    logger.error("Unexpected error during hypertable creation: %s", exc)
        logger.info("Database schema initialised.")

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
