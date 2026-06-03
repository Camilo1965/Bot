"""
data.ohlcv_cache
~~~~~~~~~~~~~~~~

DuckDB-backed OHLCV cache with TTL. Wraps `fetch_ohlcv_ccxt` and returns
cached data when fresh enough. Each (symbol, timeframe) is checked against
MAX(ts); if stale or insufficient, the live fetch runs and results are upserted.

Usage::

    from data.ohlcv_cache import OHLCVCache
    cache = OHLCVCache()
    df = cache.fetch("BTC/USDT", "15m", limit=5760)  # 60d at 15m

TTL defaults:
    max_age_s=300 — refetch if most recent candle > 5 min old

The cache lives at data/cache/ohlcv.duckdb and is safe to delete to force
a full refetch.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _REPO / "data" / "cache" / "ohlcv.duckdb"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol    VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    ts        TIMESTAMP NOT NULL,
    open      DOUBLE,
    high      DOUBLE,
    low       DOUBLE,
    close     DOUBLE,
    volume    DOUBLE,
    PRIMARY KEY (symbol, timeframe, ts)
)
"""


class OHLCVCache:
    def __init__(self, db_path: Path | str = _DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._available = True
        try:
            with self._conn() as con:
                con.execute(_CREATE_TABLE)
            logger.debug("OHLCVCache ready at %s", self._db_path)
        except Exception as exc:
            logger.warning("OHLCVCache init failed (DB locked?): %s — cache disabled", exc)
            self._available = False

    def _conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._db_path))

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        max_age_s: float = 300.0,
    ) -> pd.DataFrame:
        """Return cached OHLCV df if fresh + sufficient, else refetch and upsert.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        Falls back to direct CCXT fetch if DB is locked or unavailable.
        """
        if not self._available:
            logger.debug("[cache SKIP] DB unavailable, fetching %s %s from CCXT", symbol, timeframe)
            df = self._live_fetch(symbol, timeframe, limit)
            return df if df is not None else pd.DataFrame()

        with self._lock:
            if self._is_fresh(symbol, timeframe, limit, max_age_s):
                df = self._read(symbol, timeframe, limit)
                if df is not None and len(df) >= limit:
                    logger.debug("[cache HIT] %s %s (n=%d)", symbol, timeframe, len(df))
                    return df

        logger.debug("[cache MISS] %s %s — fetching from CCXT", symbol, timeframe)
        df = self._live_fetch(symbol, timeframe, limit)
        if df is not None and len(df) > 0:
            with self._lock:
                self._upsert(symbol, timeframe, df)
        return df if df is not None else pd.DataFrame()

    def _is_fresh(self, symbol: str, timeframe: str, limit: int, max_age_s: float) -> bool:
        try:
            with self._conn() as con:
                row = con.execute(
                    "SELECT MAX(ts), COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
                    [symbol, timeframe],
                ).fetchone()
            if row is None or row[0] is None:
                return False
            max_ts, count = row
            if count < limit:
                return False
            now = datetime.now(timezone.utc)
            try:
                last_dt = pd.Timestamp(max_ts).tz_localize("UTC") if pd.Timestamp(max_ts).tzinfo is None else pd.Timestamp(max_ts)
            except Exception:
                return False
            age_s = (now - last_dt.to_pydatetime()).total_seconds()
            return age_s <= max_age_s
        except Exception as exc:
            logger.debug("OHLCVCache._is_fresh error: %s", exc)
            return False

    def _read(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        try:
            with self._conn() as con:
                df = con.execute(
                    "SELECT ts AS timestamp, open, high, low, close, volume FROM ohlcv "
                    "WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?",
                    [symbol, timeframe, limit],
                ).df()
            if df is None or df.empty:
                return None
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        except Exception as exc:
            logger.warning("OHLCVCache._read error: %s", exc)
            return None

    def _upsert(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        try:
            rows = []
            for _, row in df.iterrows():
                ts = row.get("timestamp")
                if ts is None:
                    continue
                ts_dt = pd.Timestamp(ts)
                if ts_dt.tzinfo is not None:
                    ts_dt = ts_dt.tz_localize(None)
                rows.append((
                    symbol, timeframe,
                    ts_dt.isoformat(),
                    float(row.get("open", 0) or 0),
                    float(row.get("high", 0) or 0),
                    float(row.get("low", 0) or 0),
                    float(row.get("close", 0) or 0),
                    float(row.get("volume", 0) or 0),
                ))
            if not rows:
                return
            with self._conn() as con:
                con.executemany(
                    "INSERT OR REPLACE INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            logger.debug("[cache UPSERT] %s %s %d rows", symbol, timeframe, len(rows))
        except Exception as exc:
            logger.warning("OHLCVCache._upsert error: %s", exc)

    @staticmethod
    def _live_fetch(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        try:
            from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
            return fetch_ohlcv_ccxt(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:
            logger.error("OHLCVCache live fetch failed for %s %s: %s", symbol, timeframe, exc)
            return None

    def invalidate(self, symbol: str | None = None, timeframe: str | None = None) -> None:
        """Remove cached entries. Pass symbol+timeframe to target, or None to clear all."""
        with self._lock:
            try:
                with self._conn() as con:
                    if symbol and timeframe:
                        con.execute("DELETE FROM ohlcv WHERE symbol=? AND timeframe=?", [symbol, timeframe])
                    elif symbol:
                        con.execute("DELETE FROM ohlcv WHERE symbol=?", [symbol])
                    else:
                        con.execute("DELETE FROM ohlcv")
            except Exception as exc:
                logger.warning("OHLCVCache.invalidate error: %s", exc)


# Module-level singleton (lazy-init)
_cache: OHLCVCache | None = None
_cache_lock = threading.Lock()


def get_cache(db_path: Path | str = _DEFAULT_DB) -> OHLCVCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                try:
                    _cache = OHLCVCache(db_path)
                except Exception as exc:
                    logger.warning("get_cache: could not init OHLCVCache: %s", exc)
                    _cache = OHLCVCache.__new__(OHLCVCache)
                    _cache._db_path = Path(db_path)
                    _cache._lock = threading.Lock()
                    _cache._available = False
    return _cache
