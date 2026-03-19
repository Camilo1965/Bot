"""
gui.db_reader
~~~~~~~~~~~~~

QThread worker that polls TimescaleDB every 2 seconds and emits structured
data back to the main GUI thread via Qt signals.

Fetched each tick:
* Latest OHLCV candles (15-minute buckets, last 12 h) for the selected symbol.
* Total realised PnL from closed trades.
* Active (open) trade records.
* The most recent sentiment score from the news_sentiment table.

The worker uses a plain synchronous ``asyncpg`` connection created inside its
own thread-local event loop so it does not interfere with the trading engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

_OHLCV_QUERY = """
SELECT
    time_bucket('15 minutes', timestamp)              AS bucket,
    first((best_bid + best_ask) / 2.0, timestamp)    AS open,
    MAX((best_bid + best_ask) / 2.0)                 AS high,
    MIN((best_bid + best_ask) / 2.0)                 AS low,
    last((best_bid + best_ask) / 2.0, timestamp)     AS close,
    COUNT(*)                                          AS volume
FROM market_data
WHERE symbol = $1
  AND timestamp > NOW() - INTERVAL '12 hours'
GROUP BY bucket
ORDER BY bucket ASC;
"""

_TOTAL_PNL_QUERY = """
SELECT COALESCE(SUM(pnl), 0.0) AS total_pnl
FROM trades_history
WHERE status = 'closed';
"""

_ACTIVE_TRADES_QUERY = """
SELECT id, symbol, entry_price, position_size, entry_time
FROM trades_history
WHERE status = 'open'
ORDER BY entry_time DESC;
"""

_LATEST_SENTIMENT_QUERY = """
SELECT sentiment_score, timestamp
FROM news_sentiment
ORDER BY timestamp DESC
LIMIT 1;
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_dsn() -> str:
    """Construct the asyncpg DSN from environment variables (same as db_manager)."""
    user = os.environ.get("DB_USER", "clawdbot")
    password = os.environ.get("DB_PASSWORD", "clawdbot_secret")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "clawdbot")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class DBReaderThread(QThread):
    """Polls TimescaleDB every 2 seconds and emits data signals.

    Signals
    -------
    data_ready : emitted on each successful poll with a dict containing keys:
        ``ohlcv``      – list of dicts {bucket, open, high, low, close, volume}
        ``total_pnl``  – float, total realised PnL of closed trades
        ``active_trades`` – list of dicts describing open positions
        ``sentiment``  – float | None, latest sentiment score
    error      : emitted when a database error occurs (str message).
    log_message: emitted when the worker has a status message for the log panel.
    """

    data_ready: pyqtSignal = pyqtSignal(dict)
    error: pyqtSignal = pyqtSignal(str)
    log_message: pyqtSignal = pyqtSignal(str)

    POLL_INTERVAL_S: float = 2.0

    def __init__(self, symbol: str = "BTC/USDT", parent: Any = None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_symbol(self, symbol: str) -> None:
        """Change the symbol to query (takes effect on the next poll)."""
        self.symbol = symbol

    def stop(self) -> None:
        """Request the worker to exit cleanly."""
        self._running = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: D102
        self._running = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._poll_loop())
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Async poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        import asyncpg  # imported here to keep the top-level import clean

        conn: asyncpg.Connection | None = None
        dsn = _build_dsn()

        while self._running:
            # (Re)connect if needed
            if conn is None or conn.is_closed():
                try:
                    conn = await asyncpg.connect(dsn=dsn)
                    self.log_message.emit("DB reader connected to TimescaleDB.")
                except Exception as exc:  # noqa: BLE001
                    host = os.environ.get("DB_HOST", "localhost")
                    port = os.environ.get("DB_PORT", "5432")
                    err = f"DB connection error ({host}:{port}): {exc}"
                    logger.warning(err)
                    self.error.emit(err)
                    await asyncio.sleep(self.POLL_INTERVAL_S)
                    continue

            try:
                payload = await self._fetch_all(conn)
                self.data_ready.emit(payload)
            except Exception as exc:  # noqa: BLE001
                err = f"DB query error: {exc}"
                logger.warning(err)
                self.error.emit(err)
                # Force reconnect on next iteration
                try:
                    await conn.close()
                except Exception:  # noqa: BLE001
                    pass
                conn = None

            await asyncio.sleep(self.POLL_INTERVAL_S)

        if conn is not None and not conn.is_closed():
            await conn.close()

    async def _fetch_all(self, conn: Any) -> dict[str, Any]:
        """Run all queries in parallel and assemble the payload dict."""
        ohlcv_rows, pnl_row, trade_rows, sentiment_row = await asyncio.gather(
            conn.fetch(_OHLCV_QUERY, self.symbol),
            conn.fetchrow(_TOTAL_PNL_QUERY),
            conn.fetch(_ACTIVE_TRADES_QUERY),
            conn.fetchrow(_LATEST_SENTIMENT_QUERY),
        )

        ohlcv = [
            {
                "bucket": row["bucket"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
            for row in ohlcv_rows
        ]

        total_pnl = float(pnl_row["total_pnl"]) if pnl_row else 0.0

        active_trades = [
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "entry_price": float(row["entry_price"]),
                "position_size": float(row["position_size"]),
                "entry_time": row["entry_time"],
            }
            for row in trade_rows
        ]

        sentiment: float | None = None
        if sentiment_row:
            sentiment = float(sentiment_row["sentiment_score"])

        ts = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        self.log_message.emit(
            f"[{ts}] Fetched {len(ohlcv)} candles | PnL={total_pnl:+.2f} | "
            f"Open trades={len(active_trades)} | Sentiment={sentiment}"
        )

        return {
            "ohlcv": ohlcv,
            "total_pnl": total_pnl,
            "active_trades": active_trades,
            "sentiment": sentiment,
        }
