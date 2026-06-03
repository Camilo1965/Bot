"""
GET  /api/risk/state
GET  /api/risk/killswitch/state
GET  /api/risk/killswitch/history
POST /api/risk/killswitch/trigger
POST /api/risk/killswitch/reset
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add bot root to sys.path so risk.kill_switch can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_KS_LOG = _REPO / "logs" / "kill_switch_events.log"


def _get_ks():
    try:
        from risk.kill_switch import get_kill_switch
        return get_kill_switch()
    except Exception:
        return None


def _build_risk_state() -> dict:
    ks = _get_ks()
    if ks is None:
        return {
            "kill_switch_active": False,
            "reason": None,
            "since": None,
            "until": None,
            "demoted_symbols": {},
            "consecutive_losses": 0,
            "daily_pnl_pct": 0.0,
            "drawdown_7d_pct": 0.0,
        }
    status = ks.status()
    return {
        "kill_switch_active": ks.is_active(),
        "reason": status.get("reason"),
        "since": status.get("since"),
        "until": status.get("until"),
        "demoted_symbols": status.get("demoted_symbols", {}),
        "consecutive_losses": 0,
        "daily_pnl_pct": 0.0,
        "drawdown_7d_pct": 0.0,
    }


def _read_ks_history_sync() -> list[dict]:
    if not _KS_LOG.is_file():
        return []
    events = []
    try:
        with open(_KS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return events


class TriggerBody(BaseModel):
    reason: str
    duration_hours: float = 24.0


@router.get("/api/risk/state")
async def get_risk_state() -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_risk_state)


@router.get("/api/risk/killswitch/state")
async def get_killswitch_state() -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_risk_state)


@router.get("/api/risk/killswitch/history")
async def get_killswitch_history() -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _read_ks_history_sync)


@router.post("/api/risk/killswitch/trigger")
async def trigger_killswitch(body: TriggerBody) -> dict:
    def _do():
        ks = _get_ks()
        if ks is None:
            raise RuntimeError("kill_switch module unavailable")
        ks.activate(body.reason, body.duration_hours)
        return {"ok": True, "reason": body.reason, "duration_hours": body.duration_hours}

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _do)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/risk/killswitch/reset")
async def reset_killswitch() -> dict:
    def _do():
        ks = _get_ks()
        if ks is None:
            raise RuntimeError("kill_switch module unavailable")
        ks.deactivate()
        return {"ok": True}

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _do)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
