"""
GET  /api/backtest/list           — list report files in logs/
POST /api/backtest/run            — launch backtest subprocess, stream SSE progress
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_LOGS_DIR = _REPO / "logs"


@router.get("/api/backtest/list")
async def list_reports() -> list[dict]:
    if not _LOGS_DIR.is_dir():
        return []
    files = []
    for f in sorted(_LOGS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.suffix == ".json" and "backtest" in f.name:
            files.append({
                "filename": f.name,
                "created_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                "size_bytes": f.stat().st_size,
                "path": str(f),
            })
    return files[:20]


async def _stream_subprocess(cmd: list[str]) -> asyncio.StreamReader | None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_REPO),
    )
    return proc


@router.post("/api/backtest/run")
async def run_backtest(body: dict) -> StreamingResponse:
    """Launch backtest and stream stdout as SSE events."""
    symbols = body.get("symbols", "BTC/USDT")
    days = int(body.get("days", 30))
    mode = body.get("mode", "disk")
    pt_override = body.get("prob_threshold_override")

    script = "backtest_disk_loaded.py" if mode == "disk" else "backtest_full_bot.py"
    cmd = [sys.executable, str(_REPO / "scripts" / script)]
    if mode == "disk":
        sym_list = [s.strip() for s in str(symbols).split(",") if s.strip()]
        if len(sym_list) == 1:
            cmd += ["--symbol", sym_list[0]]
        else:
            cmd += ["--symbols", ",".join(sym_list)]
        cmd += ["--days", str(days)]
        if pt_override is not None:
            cmd += ["--prob-threshold", str(float(pt_override))]

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'cmd': ' '.join(cmd)})}\n\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(_REPO),
            )
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                yield f"data: {json.dumps({'type': 'log', 'line': text})}\n\n"
            await proc.wait()
            yield f"data: {json.dumps({'type': 'done', 'exit_code': proc.returncode})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'msg': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
