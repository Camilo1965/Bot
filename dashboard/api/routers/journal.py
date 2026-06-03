"""
GET /api/journal/trades          — recent closed trades (sourced from DB)
GET /api/journal/trades/csv      — direct stream of the paper_executor CSV journal (if present)
"""

from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_JOURNAL_CSV = _REPO / "logs" / "paper_journal.csv"


@router.get("/api/journal/trades")
async def list_trades(limit: int = Query(default=200, ge=1, le=1000)) -> list[dict]:
    """Return recent closed trades. Tries DB first; falls back to CSV journal."""
    # DB path
    try:
        sys.path.insert(0, str(_REPO))
        from database.db_manager import db
        rows = await db.fetch_recent_closed_trades(limit=int(limit))
        out: list[dict] = []
        for r in rows:
            out.append({
                "id": str(r.get("symbol", "")) + "_" + str(r.get("exit_time", "")),
                "timestamp_close": str(r.get("exit_time", "")),
                "symbol": r.get("symbol", ""),
                "side": "long",
                "pnl_usd": float(r.get("pnl") or 0.0),
                "pnl_pct": 0.0,
                "close_reason": "",
                "ml_confidence": 0.0,
            })
        if out:
            return out
    except Exception:
        pass

    # CSV fallback
    if not _JOURNAL_CSV.is_file():
        return []
    try:
        with _JOURNAL_CSV.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)[-int(limit):]
        out2: list[dict] = []
        for r in rows:
            out2.append({
                "id": r.get("trade_id", ""),
                "timestamp_close": r.get("timestamp_close", ""),
                "symbol": r.get("symbol", ""),
                "side": r.get("side", "long").lower(),
                "pnl_usd": float(r.get("pnl_usd") or 0.0),
                "pnl_pct": float(r.get("pnl_pct") or 0.0),
                "close_reason": r.get("close_reason", ""),
                "ml_confidence": float(r.get("ml_confidence") or 0.0),
            })
        return list(reversed(out2))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})


@router.get("/api/journal/trades/csv")
async def export_trades_csv(limit: int = Query(default=2000, ge=1, le=10000)) -> PlainTextResponse:
    rows = await list_trades(limit=limit)
    if isinstance(rows, JSONResponse):
        return PlainTextResponse(content="error", status_code=500)
    if not rows:
        return PlainTextResponse(content="", media_type="text/csv")
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return PlainTextResponse(content=buf.getvalue(), media_type="text/csv")
