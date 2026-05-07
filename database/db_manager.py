"""
database.db_manager
~~~~~~~~~~~~~~~~~~~

TimescaleDB persistence manager for market data, trades, and ML telemetry.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

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

_CREATE_TRADE_HISTORY = """
CREATE TABLE IF NOT EXISTS trades_history (
    id UUID PRIMARY KEY,
    timestamp_open TIMESTAMPTZ NOT NULL,
    timestamp_close TIMESTAMPTZ,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    lots DOUBLE PRECISION NOT NULL,
    pnl_usd DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    exit_reason TEXT,
    win_probability DOUBLE PRECISION,
    atr_at_entry DOUBLE PRECISION
);
"""

_CREATE_HYPERTABLE_HTF = "SELECT create_hypertable('htf_trend', 'timestamp', if_not_exists => TRUE);"
_CREATE_HYPERTABLE_TRADES = "SELECT create_hypertable('trades_history', 'timestamp_open', if_not_exists => TRUE);"

_INSERT_TRADE = """
INSERT INTO trades_history (
    id, timestamp_open, symbol, timeframe, side, entry_price, lots, win_probability, atr_at_entry
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);
"""

_UPDATE_TRADE_EXIT = """
UPDATE trades_history SET
    timestamp_close = $2,
    exit_price = $3,
    pnl_usd = $4,
    pnl_pct = $5,
    exit_reason = $6
WHERE id = $1;
"""

_INSERT_PRED = "INSERT INTO ml_predictions (timestamp, symbol, confidence, side) VALUES ($1, $2, $3, $4);"
_UPSERT_HTF = "INSERT INTO htf_trend (timestamp, symbol, timeframe, trend_status) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING;"

_EMPTY_PERIOD_SUMMARY: dict[str, Any] = {
    "pnl_total": 0.0,
    "total_trades": 0,
    "wins": 0,
    "losses": 0,
    "winrate": 0.0,
    "profit_factor": 0.0,
    "best_trade": None,
    "worst_trade": None,
}

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
            
            await conn.execute(_CREATE_TRADE_HISTORY)
            try: await conn.execute(_CREATE_HYPERTABLE_TRADES)
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
        v_bar = max(float(volume), 0.0)
        # Split bar volume across bid/ask so NOT NULL tick columns stay consistent (~total preserved).
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
            "bid_volume": v_bar * 0.5,
            "ask_volume": v_bar * 0.5,
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
            "bid_volume",
            "ask_volume",
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

    async def insert_trade_open(
        self,
        trade_id: Any,
        timestamp_open: datetime,
        symbol: str,
        timeframe: str,
        side: str,
        entry_price: float,
        lots: float,
        win_probability: float | None = None,
        atr: float | None = None,
    ) -> None:
        if not self._pool: return
        async with self._pool.acquire() as conn:
            await conn.execute(
                _INSERT_TRADE,
                trade_id, timestamp_open, symbol, timeframe, side,
                entry_price, lots, win_probability, atr
            )

    async def update_trade_exit(
        self,
        trade_id: Any,
        timestamp_close: datetime,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        if not self._pool: return
        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPDATE_TRADE_EXIT,
                trade_id, timestamp_close, exit_price, pnl_usd, pnl_pct, exit_reason
            )

    # ── Legacy Aliases for MT5Executor ───────────────────────────────────────
    async def insert_open_trade(self, symbol: str, entry_price: float, position_size: float, entry_time: datetime) -> Any:
        trade_id = uuid.uuid4()
        await self.insert_trade_open(
            trade_id=trade_id,
            timestamp_open=entry_time,
            symbol=symbol,
            timeframe="15m", # default
            side="LONG",
            entry_price=entry_price,
            lots=position_size / entry_price if entry_price > 0 else 0,
        )
        return trade_id

    async def close_trade(self, trade_id: Any, exit_price: float, exit_time: datetime, pnl: float, **kwargs: Any) -> None:
        pnl_pct = kwargs.get("pnl_pct", 0.0)
        reason = kwargs.get("exit_reason", "unknown")
        await self.update_trade_exit(
            trade_id=trade_id,
            timestamp_close=exit_time,
            exit_price=exit_price,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )

    # ── CEO / Telegram: closed-trade analytics (expects ``trades_history``) ─────────

    @staticmethod
    async def _trade_column_triplet(conn: asyncpg.Connection) -> tuple[str, str, str, str] | None:
        rows = await conn.fetch(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'trades_history'
            """
        )
        if not rows:
            return None
        names = {str(r["column_name"]) for r in rows}
        pnl_c = next((c for c in ("pnl", "net_pnl", "pnl_usdt", "gross_pnl") if c in names), None)
        exit_c = next((c for c in ("exit_time", "closed_at", "close_time") if c in names), None)
        sym_c = "symbol" if "symbol" in names else None
        if not pnl_c or not exit_c or not sym_c:
            return None
        st_c = "status" if "status" in names else ""
        return (pnl_c, exit_c, sym_c, st_c)

    @staticmethod
    def _period_start_utc(period: str, tz_name: str) -> datetime:
        tz = ZoneInfo(tz_name)
        now_local = datetime.now(tz=tz)
        if period == "day":
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "month":
            start_local = now_local - timedelta(days=30)
        else:
            start_local = now_local - timedelta(days=7)
        return start_local.astimezone(timezone.utc)

    async def fetch_period_summary(self, period: str, *, tz_name: str = "UTC") -> dict[str, Any]:
        """Aggregate closed trades for CEO dashboard / Telegram reports."""
        if not self._pool:
            return dict(_EMPTY_PERIOD_SUMMARY)
        end_utc = datetime.now(tz=timezone.utc)
        start_utc = self._period_start_utc(period, tz_name)
        try:
            async with self._pool.acquire() as conn:
                meta = await self._trade_column_triplet(conn)
                if meta is None:
                    return dict(_EMPTY_PERIOD_SUMMARY)
                pnl_c, exit_c, sym_c, st_c = meta
                st_sql = f'"{st_c}" = \'closed\'' if st_c else "TRUE"
                q_agg = f"""
                    SELECT
                      COALESCE(SUM("{pnl_c}"), 0.0) AS pnl_total,
                      COUNT(*)::bigint AS total_trades,
                      COALESCE(SUM(CASE WHEN "{pnl_c}" > 0 THEN 1 ELSE 0 END), 0)::bigint AS wins,
                      COALESCE(SUM(CASE WHEN "{pnl_c}" < 0 THEN 1 ELSE 0 END), 0)::bigint AS losses,
                      COALESCE(SUM(CASE WHEN "{pnl_c}" > 0 THEN "{pnl_c}" ELSE 0 END), 0.0) AS gross_win,
                      COALESCE(SUM(CASE WHEN "{pnl_c}" < 0 THEN "{pnl_c}" ELSE 0 END), 0.0) AS gross_loss
                    FROM trades_history
                    WHERE {st_sql}
                      AND "{exit_c}" >= $1 AND "{exit_c}" <= $2
                """
                row = await conn.fetchrow(q_agg, start_utc, end_utc)
                if row is None:
                    return dict(_EMPTY_PERIOD_SUMMARY)
                total = int(row["total_trades"] or 0)
                wins = int(row["wins"] or 0)
                losses = int(row["losses"] or 0)
                winrate = (100.0 * wins / total) if total > 0 else 0.0
                gw = float(row["gross_win"] or 0.0)
                gl = float(row["gross_loss"] or 0.0)
                if gl < 0 and gw > 0:
                    pf = gw / abs(gl)
                elif total > 0 and losses == 0 and wins > 0:
                    pf = float("inf")
                else:
                    pf = 0.0
                if pf == float("inf"):
                    pf = 999.99

                q_best = f"""
                    SELECT "{sym_c}" AS symbol, "{pnl_c}" AS pnl FROM trades_history
                    WHERE {st_sql} AND "{exit_c}" >= $1 AND "{exit_c}" <= $2
                    ORDER BY "{pnl_c}" DESC NULLS LAST LIMIT 1
                """
                q_worst = f"""
                    SELECT "{sym_c}" AS symbol, "{pnl_c}" AS pnl FROM trades_history
                    WHERE {st_sql} AND "{exit_c}" >= $1 AND "{exit_c}" <= $2
                    ORDER BY "{pnl_c}" ASC NULLS LAST LIMIT 1
                """
                br = await conn.fetchrow(q_best, start_utc, end_utc)
                wr = await conn.fetchrow(q_worst, start_utc, end_utc)
                best_trade = (
                    {"symbol": str(br["symbol"]), "pnl": float(br["pnl"])}
                    if br and br["symbol"] is not None
                    else None
                )
                worst_trade = (
                    {"symbol": str(wr["symbol"]), "pnl": float(wr["pnl"])}
                    if wr and wr["symbol"] is not None
                    else None
                )

                return {
                    "pnl_total": float(row["pnl_total"] or 0.0),
                    "total_trades": total,
                    "wins": wins,
                    "losses": losses,
                    "winrate": winrate,
                    "profit_factor": float(pf),
                    "best_trade": best_trade,
                    "worst_trade": worst_trade,
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_period_summary failed: %s", exc)
            return dict(_EMPTY_PERIOD_SUMMARY)

    async def fetch_symbol_performance(self, period: str, *, tz_name: str = "UTC") -> list[dict[str, Any]]:
        """Per-symbol realised PnL (rolling window matches CEO ``month`` = 30d)."""
        if not self._pool:
            return []
        end_utc = datetime.now(tz=timezone.utc)
        start_utc = self._period_start_utc("month" if period == "month" else "week", tz_name)
        try:
            async with self._pool.acquire() as conn:
                meta = await self._trade_column_triplet(conn)
                if meta is None:
                    return []
                pnl_c, exit_c, sym_c, st_c = meta
                st_sql = f'"{st_c}" = \'closed\'' if st_c else "TRUE"
                q = f"""
                    SELECT "{sym_c}" AS symbol, COALESCE(SUM("{pnl_c}"), 0.0) AS pnl_total
                    FROM trades_history
                    WHERE {st_sql}
                      AND "{exit_c}" >= $1 AND "{exit_c}" <= $2
                    GROUP BY "{sym_c}"
                    ORDER BY pnl_total DESC
                """
                rows = await conn.fetch(q, start_utc, end_utc)
                return [{"symbol": str(r["symbol"]), "pnl_total": float(r["pnl_total"])} for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_symbol_performance failed: %s", exc)
            return []

    async def fetch_recent_closed_trades(self, limit: int = 250) -> list[dict[str, Any]]:
        """Latest closed trades for CEO table."""
        if not self._pool:
            return []
        lim = max(1, min(int(limit), 500))
        try:
            async with self._pool.acquire() as conn:
                meta = await self._trade_column_triplet(conn)
                if meta is None:
                    return []
                pnl_c, exit_c, sym_c, st_c = meta
                st_sql = f'"{st_c}" = \'closed\'' if st_c else "TRUE"
                q = f"""
                    SELECT "{sym_c}" AS symbol, "{exit_c}" AS exit_time,
                           "{pnl_c}" AS pnl, "{pnl_c}" AS pnl_net
                    FROM trades_history
                    WHERE {st_sql}
                    ORDER BY "{exit_c}" DESC NULLS LAST
                    LIMIT $1
                """
                rows = await conn.fetch(q, lim)
                out: list[dict[str, Any]] = []
                for r in rows:
                    out.append(
                        {
                            "symbol": r["symbol"],
                            "exit_time": r["exit_time"],
                            "pnl": r["pnl"],
                            "pnl_net": r["pnl_net"],
                        }
                    )
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_recent_closed_trades failed: %s", exc)
            return []

    async def fetch_daily_pnl_series(self, *, days: int, tz_name: str = "UTC") -> list[dict[str, Any]]:
        """Calendar-ish daily totals for Telegram history report."""
        if not self._pool:
            return []
        n = max(1, min(int(days), 366))
        try:
            async with self._pool.acquire() as conn:
                meta = await self._trade_column_triplet(conn)
                if meta is None:
                    return []
                pnl_c, exit_c, _, st_c = meta
                st_sql = f'"{st_c}" = \'closed\'' if st_c else "TRUE"
                q = f"""
                    SELECT date_trunc('day', "{exit_c}" AT TIME ZONE 'UTC')::date AS day_utc,
                           COALESCE(SUM("{pnl_c}"), 0.0) AS pnl_total
                    FROM trades_history
                    WHERE {st_sql}
                      AND "{exit_c}" >= (CURRENT_TIMESTAMP AT TIME ZONE 'UTC' - $1::interval)
                    GROUP BY 1
                    ORDER BY 1 ASC
                """
                rows = await conn.fetch(q, timedelta(days=n))
                return [
                    {
                        "day_utc": r["day_utc"].isoformat() if r["day_utc"] is not None else "",
                        "pnl_total": float(r["pnl_total"] or 0.0),
                    }
                    for r in rows
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_daily_pnl_series failed: %s", exc)
            return []

db = DatabaseManager()
async def init_db() -> None: await db.connect(); await db.initialize()
async def close_db() -> None: await db.disconnect()
