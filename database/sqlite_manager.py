"""
database.sqlite_manager
~~~~~~~~~~~~~~~~~~~~~~~

SQLite/aiosqlite mirror of ``database.db_manager.DatabaseManager``.
Zero-dependency replacement for TimescaleDB on small VPS without Docker.

Schema is identical except:
  * UUID/TIMESTAMPTZ → TEXT (ISO-8601)
  * No hypertable; plain indices instead
  * Single file at ``data/clawdbot.sqlite`` (configurable via SQLITE_DB_PATH)

Public API matches DatabaseManager exactly so callers don't change.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiosqlite

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO / "data" / "clawdbot.sqlite"


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS market_data (
        timestamp TEXT NOT NULL,
        symbol    TEXT NOT NULL,
        open      REAL NOT NULL,
        high      REAL NOT NULL,
        low       REAL NOT NULL,
        close     REAL NOT NULL,
        volume    REAL NOT NULL,
        PRIMARY KEY (symbol, timestamp)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ml_predictions (
        timestamp  TEXT NOT NULL,
        symbol     TEXT NOT NULL,
        confidence REAL NOT NULL,
        side       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS htf_trend (
        timestamp    TEXT NOT NULL,
        symbol       TEXT NOT NULL,
        timeframe    TEXT NOT NULL,
        trend_status TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades_history (
        id              TEXT PRIMARY KEY,
        timestamp_open  TEXT NOT NULL,
        timestamp_close TEXT,
        symbol          TEXT NOT NULL,
        timeframe       TEXT NOT NULL,
        side            TEXT NOT NULL,
        entry_price     REAL NOT NULL,
        exit_price      REAL,
        lots            REAL NOT NULL,
        pnl_usd         REAL,
        pnl_pct         REAL,
        exit_reason     TEXT,
        win_probability REAL,
        atr_at_entry    REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_market_symbol_ts ON market_data(symbol, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades_history(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_trades_open ON trades_history(timestamp_open)",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades_history(timestamp_close)",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades_history(exit_reason)",
    "CREATE INDEX IF NOT EXISTS idx_pred_symbol_ts ON ml_predictions(symbol, timestamp DESC)",
]

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


def _ts(value: datetime | int | float) -> str:
    if isinstance(value, datetime):
        v = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).isoformat()
    t = float(value)
    if t > 1e12:
        t /= 1000.0
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class SQLiteManager:
    """Mirror of DatabaseManager for SQLite. Single connection, serialized via asyncio.Lock."""

    def __init__(self) -> None:
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        path = Path(os.environ.get("SQLITE_DB_PATH", str(_DEFAULT_DB)))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(path))
        self._conn.row_factory = aiosqlite.Row
        # WAL = concurrent readers + single writer; safer for dashboard polling.
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.commit()
        logger.info("Connected to SQLite at %s", path)

    async def disconnect(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Disconnected from SQLite.")

    async def initialize(self) -> None:
        if not self._conn:
            return
        async with self._lock:
            for stmt in _SCHEMA:
                await self._conn.execute(stmt)
            await self._conn.commit()
        logger.info("SQLite schema initialised.")

    # ── INSERT/UPSERT ──────────────────────────────────────────────────────
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
        if not self._conn:
            return
        ts = _ts(bucket)
        async with self._lock:
            await self._conn.execute(
                "INSERT OR IGNORE INTO market_data (timestamp, symbol, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, symbol, float(open), float(high), float(low), float(close), float(volume)),
            )
            await self._conn.commit()

    async def fetch_market_data(self, symbol: str, limit: int = 1000) -> list[float]:
        if not self._conn:
            return []
        async with self._lock:
            cur = await self._conn.execute(
                "SELECT close FROM market_data WHERE symbol = ? AND close IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?",
                (symbol, int(limit)),
            )
            rows = await cur.fetchall()
        return [float(r["close"]) for r in reversed(rows)]

    async def fetch_market_data_ohlcv(self, symbol: str, limit: int = 5000) -> list[dict]:
        if not self._conn:
            return []
        async with self._lock:
            cur = await self._conn.execute(
                "SELECT timestamp, open, high, low, close, volume FROM market_data "
                "WHERE symbol = ? AND close IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
                (symbol, int(limit)),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def insert_ml_prediction(self, symbol: str, confidence: float, side: str) -> None:
        if not self._conn:
            return
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO ml_predictions (timestamp, symbol, confidence, side) VALUES (?, ?, ?, ?)",
                (_now_iso(), symbol, float(confidence), side),
            )
            await self._conn.commit()

    async def upsert_htf_trend(self, symbol: str, timeframe: str, trend_status: str) -> None:
        if not self._conn:
            return
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO htf_trend (timestamp, symbol, timeframe, trend_status) VALUES (?, ?, ?, ?)",
                (_now_iso(), symbol, timeframe, trend_status),
            )
            await self._conn.commit()

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
        if not self._conn:
            return
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO trades_history (id, timestamp_open, symbol, timeframe, side, entry_price, lots, "
                "win_probability, atr_at_entry) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(trade_id),
                    _ts(timestamp_open),
                    symbol,
                    timeframe,
                    side,
                    float(entry_price),
                    float(lots),
                    None if win_probability is None else float(win_probability),
                    None if atr is None else float(atr),
                ),
            )
            await self._conn.commit()

    async def update_trade_exit(
        self,
        trade_id: Any,
        timestamp_close: datetime,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        if not self._conn:
            return
        async with self._lock:
            await self._conn.execute(
                "UPDATE trades_history SET timestamp_close = ?, exit_price = ?, pnl_usd = ?, pnl_pct = ?, "
                "exit_reason = ? WHERE id = ?",
                (_ts(timestamp_close), float(exit_price), float(pnl_usd), float(pnl_pct), exit_reason, str(trade_id)),
            )
            await self._conn.commit()

    # ── Legacy aliases ────────────────────────────────────────────────────
    async def insert_open_trade(self, symbol: str, entry_price: float, position_size: float, entry_time: datetime) -> Any:
        trade_id = str(uuid.uuid4())
        try:
            from strategy.ml_predictor import get_symbol_config
            tf = get_symbol_config(symbol).get("timeframe", "15m")
        except Exception:
            tf = "15m"
        await self.insert_trade_open(
            trade_id=trade_id,
            timestamp_open=entry_time,
            symbol=symbol,
            timeframe=tf,
            side="LONG",
            entry_price=entry_price,
            lots=position_size / entry_price if entry_price > 0 else 0.0,
        )
        return trade_id

    async def close_trade(self, trade_id: Any, exit_price: float, exit_time: datetime, pnl: float, **kwargs: Any) -> None:
        await self.update_trade_exit(
            trade_id=trade_id,
            timestamp_close=exit_time,
            exit_price=exit_price,
            pnl_usd=pnl,
            pnl_pct=kwargs.get("pnl_pct", 0.0),
            exit_reason=kwargs.get("exit_reason", "unknown"),
        )

    # ── Analytics ──────────────────────────────────────────────────────────
    @staticmethod
    def _period_start_iso(period: str, tz_name: str) -> str:
        tz = ZoneInfo(tz_name)
        now_local = datetime.now(tz=tz)
        if period == "day":
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "month":
            start_local = now_local - timedelta(days=30)
        else:
            start_local = now_local - timedelta(days=7)
        return start_local.astimezone(timezone.utc).isoformat()

    async def fetch_period_summary(self, period: str, *, tz_name: str = "UTC") -> dict[str, Any]:
        if not self._conn:
            return dict(_EMPTY_PERIOD_SUMMARY)
        end_iso = _now_iso()
        start_iso = self._period_start_iso(period, tz_name)
        try:
            async with self._lock:
                cur = await self._conn.execute(
                    """
                    SELECT
                      COALESCE(SUM(pnl_usd), 0.0) AS pnl_total,
                      COUNT(*) AS total_trades,
                      COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), 0) AS wins,
                      COALESCE(SUM(CASE WHEN pnl_usd < 0 THEN 1 ELSE 0 END), 0) AS losses,
                      COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN pnl_usd ELSE 0 END), 0.0) AS gross_win,
                      COALESCE(SUM(CASE WHEN pnl_usd < 0 THEN pnl_usd ELSE 0 END), 0.0) AS gross_loss
                    FROM trades_history
                    WHERE timestamp_close IS NOT NULL
                      AND timestamp_close >= ? AND timestamp_close <= ?
                    """,
                    (start_iso, end_iso),
                )
                row = await cur.fetchone()
                if row is None:
                    return dict(_EMPTY_PERIOD_SUMMARY)
                total = int(row["total_trades"] or 0)
                wins = int(row["wins"] or 0)
                losses = int(row["losses"] or 0)
                gw = float(row["gross_win"] or 0.0)
                gl = float(row["gross_loss"] or 0.0)
                winrate = (100.0 * wins / total) if total > 0 else 0.0
                if gl < 0 and gw > 0:
                    pf = gw / abs(gl)
                elif total > 0 and losses == 0 and wins > 0:
                    pf = 999.99
                else:
                    pf = 0.0

                cur = await self._conn.execute(
                    "SELECT symbol, pnl_usd FROM trades_history WHERE timestamp_close IS NOT NULL "
                    "AND timestamp_close >= ? AND timestamp_close <= ? ORDER BY pnl_usd DESC LIMIT 1",
                    (start_iso, end_iso),
                )
                br = await cur.fetchone()
                cur = await self._conn.execute(
                    "SELECT symbol, pnl_usd FROM trades_history WHERE timestamp_close IS NOT NULL "
                    "AND timestamp_close >= ? AND timestamp_close <= ? ORDER BY pnl_usd ASC LIMIT 1",
                    (start_iso, end_iso),
                )
                wr = await cur.fetchone()
                best = {"symbol": br["symbol"], "pnl": float(br["pnl_usd"])} if br and br["symbol"] else None
                worst = {"symbol": wr["symbol"], "pnl": float(wr["pnl_usd"])} if wr and wr["symbol"] else None

                return {
                    "pnl_total": float(row["pnl_total"] or 0.0),
                    "total_trades": total,
                    "wins": wins,
                    "losses": losses,
                    "winrate": winrate,
                    "profit_factor": float(pf),
                    "best_trade": best,
                    "worst_trade": worst,
                }
        except Exception as exc:
            logger.warning("fetch_period_summary failed: %s", exc)
            return dict(_EMPTY_PERIOD_SUMMARY)

    async def fetch_symbol_performance(self, period: str, *, tz_name: str = "UTC") -> list[dict[str, Any]]:
        if not self._conn:
            return []
        end_iso = _now_iso()
        start_iso = self._period_start_iso("month" if period == "month" else "week", tz_name)
        try:
            async with self._lock:
                cur = await self._conn.execute(
                    "SELECT symbol, COALESCE(SUM(pnl_usd), 0.0) AS pnl_total FROM trades_history "
                    "WHERE timestamp_close IS NOT NULL AND timestamp_close >= ? AND timestamp_close <= ? "
                    "GROUP BY symbol ORDER BY pnl_total DESC",
                    (start_iso, end_iso),
                )
                rows = await cur.fetchall()
            return [{"symbol": r["symbol"], "pnl_total": float(r["pnl_total"])} for r in rows]
        except Exception as exc:
            logger.warning("fetch_symbol_performance failed: %s", exc)
            return []

    async def fetch_recent_closed_trades(self, limit: int = 250) -> list[dict[str, Any]]:
        if not self._conn:
            return []
        lim = max(1, min(int(limit), 500))
        try:
            async with self._lock:
                cur = await self._conn.execute(
                    "SELECT symbol, timestamp_close AS exit_time, pnl_usd AS pnl, pnl_usd AS pnl_net "
                    "FROM trades_history WHERE timestamp_close IS NOT NULL "
                    "ORDER BY timestamp_close DESC LIMIT ?",
                    (lim,),
                )
                rows = await cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("fetch_recent_closed_trades failed: %s", exc)
            return []

    async def fetch_daily_pnl_series(self, *, days: int, tz_name: str = "UTC") -> list[dict[str, Any]]:
        if not self._conn:
            return []
        n = max(1, min(int(days), 366))
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=n)).isoformat()
        try:
            async with self._lock:
                cur = await self._conn.execute(
                    "SELECT substr(timestamp_close, 1, 10) AS day_utc, COALESCE(SUM(pnl_usd), 0.0) AS pnl_total "
                    "FROM trades_history WHERE timestamp_close IS NOT NULL AND timestamp_close >= ? "
                    "GROUP BY day_utc ORDER BY day_utc ASC",
                    (cutoff,),
                )
                rows = await cur.fetchall()
            return [{"day_utc": r["day_utc"], "pnl_total": float(r["pnl_total"] or 0.0)} for r in rows]
        except Exception as exc:
            logger.warning("fetch_daily_pnl_series failed: %s", exc)
            return []


_sqlite_db: SQLiteManager | None = None


def get_sqlite_db() -> SQLiteManager:
    global _sqlite_db
    if _sqlite_db is None:
        _sqlite_db = SQLiteManager()
    return _sqlite_db
