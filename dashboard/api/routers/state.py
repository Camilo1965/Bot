"""
GET /api/state
Returns current bot state from Redis key clawdbot:state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter
from dashboard.api.db.redis_client import get_redis

router = APIRouter()

_STUB = {
    "balance": 0.0,
    "equity": 0.0,
    "open_positions_count": 0,
    "daily_pnl_pct": 0.0,
    "kill_switch_active": False,
    "kill_switch_reason": None,
    "last_updated": None,
}


@router.get("/api/state")
async def get_state() -> dict:
    try:
        redis = await get_redis()
        raw = await redis.get("clawdbot:state")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {**_STUB, "last_updated": datetime.now(timezone.utc).isoformat()}
