"""
GET /api/state           — current bot state (balance/equity/open_pos/kill switch)
GET /api/equity?days=30  — equity curve from trade journal (daily cumulative PnL)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Query
from dashboard.api.db.redis_client import get_redis

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent

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


@router.get("/api/equity")
async def get_equity(days: int = Query(default=30, ge=1, le=365)) -> list[dict]:
    """Cumulative equity series from closed trades. Starts at INITIAL_BALANCE."""
    try:
        sys.path.insert(0, str(_REPO))
        from database.db_manager import db
        rows = await db.fetch_daily_pnl_series(days=int(days))
    except Exception:
        rows = []
    if not rows:
        return []
    import os
    base = float(os.environ.get("INITIAL_BALANCE", "10000") or "10000")
    out: list[dict] = []
    running = base
    for r in rows:
        running += float(r.get("pnl_total") or 0.0)
        out.append({"ts": r.get("day_utc", ""), "equity": running})
    return out
