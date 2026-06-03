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
    while True:
        await asyncio.sleep(1)
        if not _connections:
            continue
        try:
            redis = await get_redis()
            raw = await redis.get("clawdbot:state")
            if raw:
                state = json.loads(raw)
                await _broadcast({
                    "type": "equity.tick",
                    "equity": state.get("equity", 0.0),
                    "balance": state.get("balance", 0.0),
                    "daily_pnl_pct": state.get("daily_pnl_pct", 0.0),
                    "open_positions_count": state.get("open_positions_count", 0),
                    "ts": state.get("last_updated"),
                })
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
                        await _broadcast({"type": "event", **data})
                    except Exception:
                        pass
        except Exception:
            await asyncio.sleep(2)


async def handle_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    try:
        while True:
            # Keep connection open; client can send pings
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _connections.discard(websocket)


async def start_background_tasks() -> None:
    asyncio.create_task(_equity_tick_loop())
    asyncio.create_task(_pubsub_relay_loop())
