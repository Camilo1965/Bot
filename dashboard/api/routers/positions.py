"""
GET  /api/positions/open              — open positions from Redis
GET  /api/positions/history           — closed trades from trade_journal.csv
POST /api/positions/{sym}/close       — write close signal to Redis for the bot to pick up
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Query
from dashboard.api.db.redis_client import get_redis
from dashboard.api.db.duckdb_client import query_trades

router = APIRouter()


@router.get("/api/positions/open")
async def get_open_positions() -> list[dict]:
    try:
        redis = await get_redis()
        keys = await redis.keys("clawdbot:positions:*")
        if not keys:
            return []
        raw_values = await redis.mget(*keys)
        positions = []
        for raw in raw_values:
            if raw:
                try:
                    positions.append(json.loads(raw))
                except Exception:
                    pass
        return positions
    except Exception:
        return []


@router.post("/api/positions/{sym}/close")
async def manual_close_position(sym: str) -> dict:
    """Write a manual-close flag to Redis so the bot closes the position on next tick."""
    symbol = sym.replace("_", "/").upper()
    try:
        redis = await get_redis()
        # Bot polls clawdbot:close_signals:{symbol} and closes on next loop iteration
        await redis.set(f"clawdbot:close_signal:{symbol}", "1", ex=300)
        return {"ok": True, "symbol": symbol, "msg": "Close signal queued (TTL 5min)"}
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "error": str(exc)}


@router.get("/api/positions/history")
async def get_position_history(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = Query(None),
) -> list[dict]:
    rows = await query_trades(days=90, symbol=symbol)
    return rows[:limit]
