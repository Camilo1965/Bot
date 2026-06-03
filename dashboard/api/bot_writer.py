"""
bot_writer.py
~~~~~~~~~~~~~
Called from the bot's main loop to persist state to Redis.
All functions are non-blocking — errors are swallowed so the bot loop never
fails due to a Redis problem.

Redis key schema
----------------
clawdbot:state                      → JSON blob (current bot state)
clawdbot:positions:{trade_id}       → JSON position, expires 7d
clawdbot:signals:{symbol}           → JSON latest signal, expires 1h
clawdbot:events (pub/sub channel)   → position change events
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import redis.asyncio as aioredis

_redis: aioredis.Redis | None = None

_7D = 7 * 24 * 3600
_1H = 3600


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
    return _redis


async def write_state(
    balance: float,
    equity: float,
    daily_pnl_pct: float,
    open_positions: dict,
) -> None:
    try:
        r = await _get_redis()
        payload = {
            "balance": balance,
            "equity": equity,
            "daily_pnl_pct": daily_pnl_pct,
            "open_positions_count": len(open_positions),
            "kill_switch_active": False,
            "kill_switch_reason": None,
            "last_updated": _utcnow(),
        }
        # Try to enrich with kill switch state without hard-coupling
        try:
            import sys
            import os as _os
            _root = _os.path.join(_os.path.dirname(__file__), "..", "..", "..")
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from risk.kill_switch import get_kill_switch
            ks = get_kill_switch()
            payload["kill_switch_active"] = ks.is_active()
            payload["kill_switch_reason"] = ks.get_reason()
        except Exception:
            pass
        await r.set("clawdbot:state", json.dumps(payload))
    except Exception:
        pass


async def write_position_opened(
    trade_id: str,
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    size: float,
) -> None:
    try:
        r = await _get_redis()
        payload = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "size": size,
            "status": "open",
            "opened_at": _utcnow(),
        }
        key = f"clawdbot:positions:{trade_id}"
        await r.set(key, json.dumps(payload), ex=_7D)
        await r.publish(
            "clawdbot:events",
            json.dumps({"event": "position_opened", "trade_id": trade_id, "symbol": symbol, "ts": _utcnow()}),
        )
    except Exception:
        pass


async def write_position_closed(
    trade_id: str,
    pnl_usd: float,
    reason: str,
) -> None:
    try:
        r = await _get_redis()
        key = f"clawdbot:positions:{trade_id}"
        raw = await r.get(key)
        payload: dict = json.loads(raw) if raw else {"trade_id": trade_id}
        payload.update({
            "status": "closed",
            "pnl_usd": pnl_usd,
            "close_reason": reason,
            "closed_at": _utcnow(),
        })
        # Keep for 7d after close for audit purposes
        await r.set(key, json.dumps(payload), ex=_7D)
        await r.publish(
            "clawdbot:events",
            json.dumps({
                "event": "position_closed",
                "trade_id": trade_id,
                "pnl_usd": pnl_usd,
                "reason": reason,
                "ts": _utcnow(),
            }),
        )
    except Exception:
        pass


async def publish_signal(
    symbol: str,
    raw_prob: float,
    cal_prob: float,
    threshold: float,
    decision: str,
) -> None:
    try:
        r = await _get_redis()
        payload = {
            "symbol": symbol,
            "raw_prob": raw_prob,
            "cal_prob": cal_prob,
            "threshold": threshold,
            "decision": decision,
            "ts": _utcnow(),
        }
        await r.set(f"clawdbot:signals:{symbol}", json.dumps(payload), ex=_1H)
    except Exception:
        pass
