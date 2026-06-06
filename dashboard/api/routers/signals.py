"""
GET /api/signals/recent           — last N signals (CSV if exists, else Redis snapshot)
GET /api/signals/by-symbol/{sym}  — filtered rows
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from dashboard.api.db.redis_client import get_redis

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_SIGNALS_CSV = _REPO / "logs" / "shadow_run_signals.csv"


def _read_signals_sync(limit: int, symbol: Optional[str]) -> list[dict]:
    if not _SIGNALS_CSV.is_file():
        return []
    target = _norm_sym(symbol) if symbol else None
    try:
        rows: list[dict] = []
        with open(_SIGNALS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if target and _norm_sym(row.get("symbol", "")) != target:
                    continue
                rows.append({
                    "timestamp": row.get("timestamp", ""),
                    "symbol": row.get("symbol", ""),
                    "cal_prob": _to_float(row.get("cal_prob")),
                    "raw_prob": _to_float(row.get("raw_prob")),
                    "threshold": _to_float(row.get("threshold")),
                    "ml_decision": row.get("ml_decision", ""),
                    "regime_prob": _to_float(row.get("regime_prob")) if row.get("regime_prob") else None,
                    "htf_pass": row.get("htf_pass", "").lower() not in ("false", "0", ""),
                    "reason_skip": row.get("reason_skip", ""),
                    "executed": row.get("executed", "False"),
                })
        return rows[-limit:][::-1]
    except Exception:
        return []


def _norm_sym(s: str) -> str:
    return s.replace("/", "").replace("_", "").upper()


def _to_float(v: Optional[str]) -> float:
    try:
        return float(v) if v is not None and v != "" else 0.0
    except (ValueError, TypeError):
        return 0.0


async def _read_signals_redis(limit: int, symbol: Optional[str]) -> list[dict]:
    try:
        redis = await get_redis()
        keys = await redis.keys("clawdbot:signals:*")
        if not keys:
            return []
        values = await redis.mget(*keys)
        target = _norm_sym(symbol) if symbol else None
        rows: list[dict] = []
        for raw in values:
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            sym = str(obj.get("symbol", ""))
            if target and _norm_sym(sym) != target:
                continue
            rows.append({
                "timestamp": obj.get("ts", ""),
                "symbol": sym,
                "cal_prob": obj.get("cal_prob", obj.get("raw_prob", 0.0)),
                "raw_prob": obj.get("raw_prob", 0.0),
                "threshold": obj.get("threshold", 0.0),
                "ml_decision": obj.get("decision", ""),
                "regime_prob": obj.get("regime_prob"),
                "htf_pass": obj.get("htf_pass"),
                "reason_skip": obj.get("reason_skip", ""),
                "executed": "False",
            })
        rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return rows[:limit]
    except Exception:
        return []


@router.get("/api/signals/recent")
async def get_recent_signals(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    # Prefer Redis (live, updated every ~15s) — fall back to CSV for historical rows
    redis_rows = await _read_signals_redis(limit, None)
    if redis_rows:
        # Merge with CSV history so the table shows more than just one row per symbol
        loop = asyncio.get_running_loop()
        csv_rows = await loop.run_in_executor(None, _read_signals_sync, limit, None)
        seen: set[tuple] = set()
        merged: list[dict] = []
        for r in redis_rows + csv_rows:
            key = (r.get("symbol", ""), r.get("timestamp", ""))
            if key not in seen:
                seen.add(key)
                merged.append(r)
        merged.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return merged[:limit]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _read_signals_sync, limit, None)


@router.get("/api/signals/by-symbol/{sym}")
async def get_signals_by_symbol(
    sym: str,
    limit: int = Query(200, ge=1, le=1000),
) -> list[dict]:
    loop = asyncio.get_running_loop()
    csv_rows = await loop.run_in_executor(None, _read_signals_sync, limit, sym)
    if csv_rows:
        return csv_rows
    return await _read_signals_redis(limit, sym)
