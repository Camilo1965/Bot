"""
GET  /api/alerts           — merged kill_switch_events + drift_events, sorted by ts
POST /api/alerts/ack       — mark alert IDs as read in Redis
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel
from dashboard.api.db.redis_client import get_redis

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_KS_LOG = _REPO / "logs" / "kill_switch_events.log"
_DRIFT_LOG = _REPO / "logs" / "drift_events.log"
_ACKED_KEY = "clawdbot:alerts:acked"


def _derive_alert(obj: dict, source: str) -> dict:
    """Inject UI-facing severity/message defaults derived from event shape."""
    event = str(obj.get("event") or obj.get("type") or "")
    severity = obj.get("severity")
    message = obj.get("message")

    if not severity:
        if source == "kill_switch":
            if event == "activated":
                severity = "critical"
            elif event == "symbol_demoted":
                severity = "warn"
            else:
                severity = "info"
        elif source == "drift":
            severity = "warn"
        else:
            severity = "info"

    if not message:
        if source == "kill_switch":
            if event == "activated":
                reason = obj.get("reason", "")
                message = f"Kill switch activated{(': ' + reason) if reason else ''}"
            elif event == "deactivated":
                message = "Kill switch deactivated"
            elif event == "symbol_demoted":
                sym = obj.get("symbol", "?")
                thr = obj.get("threshold")
                message = f"Symbol demoted: {sym}" + (f" (threshold={thr})" if thr is not None else "")
            else:
                message = f"kill_switch event: {event or 'unknown'}"
        elif source == "drift":
            if event == "drift_detected":
                sym = obj.get("symbol", "?")
                ks = obj.get("ks_stat")
                pv = obj.get("p_value")
                ks_str = f", KS={float(ks):.3f}" if isinstance(ks, (int, float)) else ""
                pv_str = f", p={float(pv):.4f}" if isinstance(pv, (int, float)) else ""
                message = f"Drift detected: {sym}{ks_str}{pv_str}"
            else:
                message = f"drift event: {event or 'unknown'}"
        else:
            message = event or "alert"

    obj["severity"] = severity
    obj["message"] = message
    obj.setdefault("source", source)
    if "id" not in obj:
        obj["id"] = f"{source}:{obj.get('ts', '')}:{event}"
    return obj


def _read_jsonl_sync(path: Path, source: str) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    events.append(_derive_alert(obj, source))
                except Exception:
                    pass
    except Exception:
        pass
    return events


def _read_all_alerts_sync() -> list[dict]:
    ks_events = _read_jsonl_sync(_KS_LOG, "kill_switch")
    drift_events = _read_jsonl_sync(_DRIFT_LOG, "drift")
    merged = ks_events + drift_events
    merged.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return merged


class AckBody(BaseModel):
    ids: list[str]


@router.get("/api/alerts")
async def get_alerts(status: Optional[str] = Query("unread")) -> list[dict]:
    loop = asyncio.get_running_loop()
    alerts = await loop.run_in_executor(None, _read_all_alerts_sync)

    if status == "unread":
        try:
            redis = await get_redis()
            acked = await redis.smembers(_ACKED_KEY)
            acked_set = {a.decode() if isinstance(a, bytes) else a for a in acked}
            alerts = [a for a in alerts if a.get("id") not in acked_set]
        except Exception:
            pass

    return alerts


@router.post("/api/alerts/ack")
async def ack_alerts(body: AckBody) -> dict:
    if not body.ids:
        return {"ok": True, "acked": 0}
    try:
        redis = await get_redis()
        await redis.sadd(_ACKED_KEY, *body.ids)
        return {"ok": True, "acked": len(body.ids)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
