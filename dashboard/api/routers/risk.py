"""
GET  /api/risk/state
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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes")


def _build_risk_state() -> dict:
    ks = _get_ks()
    if ks is None:
        active = False
        reason = None
        since = None
        until = None
        demoted = {}
    else:
        status = ks.status()
        active = ks.is_active()
        reason = status.get("reason")
        since = status.get("since")
        until = status.get("until")
        demoted = status.get("demoted_symbols", {})

    hour_start = _env_int("TRADE_HOUR_START_UTC", 8)
    hour_end = _env_int("TRADE_HOUR_END_UTC", 22)

    return {
        "kill_switch_active": active,
        "reason": reason,
        "since": since,
        "until": until,
        "demoted_symbols": demoted,
        "consecutive_losses": 0,
        "daily_pnl_pct": 0.0,
        "drawdown_7d_pct": 0.0,
        # Thresholds — from env, shown in UI gauges
        "limits": {
            "max_daily_loss_pct": _env_float("MAX_DAILY_LOSS_PCT", 5.0),
            "max_consecutive_losses": _env_int("MAX_CONSECUTIVE_LOSSES", 3),
            "max_drawdown_7d_pct": _env_float("MAX_DRAWDOWN_7D_PCT", 10.0),
            "max_positions": _env_int("MAX_POSITIONS", 2),
            "risk_per_trade_pct": _env_float("RISK_PER_TRADE", 0.02) * 100,
        },
        # Active filter config — shown in filter status panel
        "filters": {
            "trade_hour_filter": _env_bool("TRADE_HOUR_FILTER", True),
            "trade_hour_start_utc": hour_start,
            "trade_hour_end_utc": hour_end,
            "max_spread_pct": _env_float("MAX_SPREAD_PCT", 0.005) * 100,
            "regime_filter_enabled": _env_bool("REGIME_FILTER_ENABLED", True),
            "regime_trending_threshold": _env_float("REGIME_TRENDING_THRESHOLD", 0.65),
            "sma200_filter": True,
            "atr_sl_mult": _env_float("ATR_SL_MULT", 2.0),
            "atr_tp_mult": _env_float("ATR_TP_MULT", 2.5),
            "base_sl_pct": _env_float("BASE_SL", 0.015) * 100,
        },
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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _build_risk_state)


@router.get("/api/risk/killswitch/state")
async def get_killswitch_state() -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _build_risk_state)


@router.get("/api/risk/killswitch/history")
async def get_killswitch_history() -> list[dict]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _read_ks_history_sync)


@router.post("/api/risk/killswitch/trigger")
async def trigger_killswitch(body: TriggerBody) -> dict:
    def _do():
        ks = _get_ks()
        if ks is None:
            raise RuntimeError("kill_switch module unavailable")
        ks.activate(body.reason, body.duration_hours)
        return {"ok": True, "reason": body.reason, "duration_hours": body.duration_hours}

    loop = asyncio.get_running_loop()
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

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _do)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
