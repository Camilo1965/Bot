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

from dashboard.api.db.redis_client import get_redis as _get_redis

_7D = 7 * 24 * 3600
_1H = 3600


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


async def write_state(
    balance: float,
    equity: float,
    daily_pnl_pct: float,
    open_positions: dict,
    *,
    execution_mode: str | None = None,
    session_id: str | None = None,
    session_started_at: str | None = None,
    watchlist: list[str] | None = None,
    total_pnl_session: float | None = None,
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
        if execution_mode is not None:
            payload["execution_mode"] = execution_mode
        if session_id is not None:
            payload["session_id"] = session_id
        if session_started_at is not None:
            payload["session_started_at"] = session_started_at
        if watchlist is not None:
            payload["watchlist"] = list(watchlist)
        if total_pnl_session is not None:
            payload["total_pnl_session"] = float(total_pnl_session)
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
    *,
    ml_confidence: float | None = None,
    sl_pct: float | None = None,
    activation_pct: float | None = None,
    atr_at_entry: float | None = None,
    timeframe: str | None = None,
    entry_time: str | None = None,
    threshold: float | None = None,
) -> None:
    try:
        r = await _get_redis()
        payload = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "sl_price": sl,
            "tp_price": tp,
            "size_usd": size,
            "current_price": entry,
            "pnl_usd": 0.0,
            "pnl_pct": 0.0,
            "status": "open",
            "opened_at": entry_time or _utcnow(),
        }
        if ml_confidence is not None:
            payload["ml_confidence"] = float(ml_confidence)
        if sl_pct is not None:
            payload["sl_pct"] = float(sl_pct)
        if activation_pct is not None:
            payload["activation_pct"] = float(activation_pct)
        if atr_at_entry is not None:
            payload["atr_at_entry"] = float(atr_at_entry)
        if timeframe is not None:
            payload["timeframe"] = str(timeframe)
        if threshold is not None:
            payload["threshold"] = float(threshold)
        key = f"clawdbot:positions:{trade_id}"
        await r.set(key, json.dumps(payload), ex=_7D)
        await r.publish(
            "clawdbot:events",
            json.dumps({"event": "position_opened", "trade_id": trade_id, "symbol": symbol, "ts": _utcnow()}),
        )
    except Exception:
        pass


async def update_position_live(
    trade_id: str,
    current_price: float,
    pnl_usd: float,
    pnl_pct: float,
    *,
    peak_price: float | None = None,
    current_sl: float | None = None,
    trailing_active: bool | None = None,
) -> None:
    """Per-tick mark-to-market update. Touches only the live fields, preserves opened metadata."""
    try:
        r = await _get_redis()
        key = f"clawdbot:positions:{trade_id}"
        raw = await r.get(key)
        if not raw:
            return
        try:
            payload: dict = json.loads(raw)
        except Exception:
            return
        if payload.get("status") != "open":
            return
        payload["current_price"] = float(current_price)
        payload["pnl_usd"] = float(pnl_usd)
        payload["pnl_pct"] = float(pnl_pct)
        if peak_price is not None:
            payload["peak_price"] = float(peak_price)
        if current_sl is not None:
            payload["current_sl"] = float(current_sl)
        if trailing_active is not None:
            payload["trailing_active"] = bool(trailing_active)
        payload["mark_updated_at"] = _utcnow()
        await r.set(key, json.dumps(payload), ex=_7D)
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
    *,
    regime_prob: float | None = None,
    htf_pass: bool | None = None,
    reason_skip: str | None = None,
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
        if regime_prob is not None:
            payload["regime_prob"] = round(float(regime_prob), 4)
        if htf_pass is not None:
            payload["htf_pass"] = bool(htf_pass)
        if reason_skip is not None:
            payload["reason_skip"] = str(reason_skip)
        await r.set(f"clawdbot:signals:{symbol}", json.dumps(payload), ex=_1H)
    except Exception:
        pass
