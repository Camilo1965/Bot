"""
GET /api/signals/recent           — last N rows from shadow_run_signals.csv
GET /api/signals/by-symbol/{sym}  — filtered rows
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_SIGNALS_CSV = _REPO / "logs" / "shadow_run_signals.csv"


def _read_signals_sync(limit: int, symbol: Optional[str]) -> list[dict]:
    if not _SIGNALS_CSV.is_file():
        return []
    try:
        rows: list[dict] = []
        with open(_SIGNALS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if symbol and row.get("symbol", "").upper() != symbol.upper():
                    continue
                rows.append(dict(row))
        # Return the last `limit` rows (CSV is chronological)
        return rows[-limit:][::-1]
    except Exception:
        return []


@router.get("/api/signals/recent")
async def get_recent_signals(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _read_signals_sync, limit, None)


@router.get("/api/signals/by-symbol/{sym}")
async def get_signals_by_symbol(
    sym: str,
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _read_signals_sync, limit, sym)
