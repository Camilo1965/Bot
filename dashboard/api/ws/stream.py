"""
WebSocket handler for /ws/stream.

- Broadcasts equity.tick from clawdbot:state every second.
- Forwards position change events from Redis pub/sub channel clawdbot:events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect
from dashboard.api.db.redis_client import get_redis

logger = logging.getLogger(__name__)

_connections: Set[WebSocket] = set()
_last_state_snapshot: dict | None = None
_last_signals: dict[str, dict] = {}


async def _broadcast(message: dict) -> None:
    dead: list[WebSocket] = []
    payload = json.dumps(message)
    for ws in list(_connections):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections.discard(ws)


async def _equity_tick_loop() -> None:
    global _last_state_snapshot
    while True:
        await asyncio.sleep(1)
        if not _connections:
            continue
        try:
            redis = await get_redis()
            raw = await redis.get("clawdbot:state")
            if raw:
                state = json.loads(raw)
                _last_state_snapshot = state
                await _broadcast({
                    "type": "equity.tick",
                    "equity": state.get("equity", 0.0),
                    "balance": state.get("balance", 0.0),
                    "daily_pnl_pct": state.get("daily_pnl_pct", 0.0),
                    "open_positions_count": state.get("open_positions_count", 0),
                    "kill_switch_active": state.get("kill_switch_active", False),
                    "ts": state.get("last_updated"),
                })
        except Exception:
            pass


async def _positions_poll_loop() -> None:
    """Polls Redis position keys every 2s and broadcasts as positions.snapshot."""
    while True:
        await asyncio.sleep(2)
        if not _connections:
            continue
        try:
            redis = await get_redis()
            keys = await redis.keys("clawdbot:positions:*")
            if not keys:
                await _broadcast({"type": "positions.snapshot", "positions": []})
                continue
            values = await redis.mget(*keys)
            positions = []
            for raw in values:
                if not raw:
                    continue
                try:
                    p = json.loads(raw)
                    if p.get("status") == "open":
                        positions.append(p)
                except Exception:
                    pass
            await _broadcast({"type": "positions.snapshot", "positions": positions})
        except Exception:
            pass


async def _signals_poll_loop() -> None:
    """Polls Redis signal keys every 3s; broadcasts diffs only."""
    global _last_signals
    while True:
        await asyncio.sleep(3)
        if not _connections:
            continue
        try:
            redis = await get_redis()
            keys = await redis.keys("clawdbot:signals:*")
            if not keys:
                continue
            values = await redis.mget(*keys)
            for k, raw in zip(keys, values):
                if not raw:
                    continue
                try:
                    sig = json.loads(raw)
                    key_str = k.decode() if isinstance(k, bytes) else k
                    if _last_signals.get(key_str) != sig:
                        _last_signals[key_str] = sig
                        await _broadcast({"type": "signal.update", **sig})
                except Exception:
                    pass
        except Exception:
            pass


async def _pubsub_relay_loop() -> None:
    while True:
        try:
            redis = await get_redis()
            pubsub = redis.pubsub()
            await pubsub.subscribe("clawdbot:events")
            async for message in pubsub.listen():
                if message.get("type") == "message":
                    try:
                        data = json.loads(message["data"])
                        # Map raw events to typed events for the frontend
                        ev_name = data.get("event", "")
                        if ev_name == "position_opened":
                            await _broadcast({"type": "position.opened", **data})
                        elif ev_name == "position_closed":
                            await _broadcast({"type": "position.closed", **data})
                        elif ev_name == "alert":
                            await _broadcast({"type": "alert.new", **data})
                        elif ev_name == "kill_switch":
                            await _broadcast({"type": "kill_switch.changed", **data})
                        else:
                            await _broadcast({"type": "event", **data})
                    except Exception:
                        pass
        except Exception:
            await asyncio.sleep(2)


async def handle_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    # Send hello + immediate snapshot so new clients don't wait for first tick
    try:
        await websocket.send_text(json.dumps({"type": "hello", "connected_clients": len(_connections)}))
        if _last_state_snapshot:
            await websocket.send_text(json.dumps({"type": "equity.tick", **_last_state_snapshot, "ts": _last_state_snapshot.get("last_updated")}))
    except Exception:
        pass
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Keepalive ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _connections.discard(websocket)


async def start_background_tasks() -> None:
    asyncio.create_task(_equity_tick_loop())
    asyncio.create_task(_positions_poll_loop())
    asyncio.create_task(_signals_poll_loop())
    asyncio.create_task(_pubsub_relay_loop())
