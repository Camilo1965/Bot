"""
DuckDB helpers for reading trade_journal.csv.
Runs blocking DuckDB queries in a threadpool executor.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_JOURNAL = _REPO / "data" / "trade_journal.csv"


def _query_trades_sync(days: int, symbol: str | None) -> list[dict]:
    if not _JOURNAL.is_file():
        return []
    try:
        import duckdb
        sym_filter = f"AND symbol = '{symbol}'" if symbol else ""
        sql = f"""
            SELECT *
            FROM read_csv_auto('{_JOURNAL.as_posix()}')
            WHERE exit_time >= now() - INTERVAL '{days} days'
            {sym_filter}
            ORDER BY exit_time DESC
        """
        rows = duckdb.execute(sql).fetchdf()
        return rows.to_dict(orient="records")
    except Exception:
        return []


def _query_equity_snapshots_sync(days: int) -> list[dict]:
    if not _JOURNAL.is_file():
        return []
    try:
        import duckdb
        sql = f"""
            SELECT
                strftime(exit_time::TIMESTAMP, '%Y-%m-%dT%H:00:00') AS ts,
                SUM(pnl_usd) AS pnl_usd,
                SUM(SUM(pnl_usd)) OVER (ORDER BY strftime(exit_time::TIMESTAMP, '%Y-%m-%dT%H:00:00')) AS cumulative_pnl
            FROM read_csv_auto('{_JOURNAL.as_posix()}')
            WHERE exit_time >= now() - INTERVAL '{days} days'
            GROUP BY 1
            ORDER BY 1
        """
        rows = duckdb.execute(sql).fetchdf()
        return rows.to_dict(orient="records")
    except Exception:
        return []


async def query_trades(days: int = 30, symbol: str | None = None) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _query_trades_sync, days, symbol)


async def query_equity_snapshots(days: int = 30) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _query_equity_snapshots_sync, days)
