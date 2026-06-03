"""
GET  /api/symbols/config           — read SYMBOL_CONFIG as list
POST /api/symbols/{sym}/retrain    — trigger retrain for one symbol (SSE)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

_REPO = Path(__file__).resolve().parent.parent.parent.parent


def _get_symbol_config() -> list[dict]:
    sys.path.insert(0, str(_REPO))
    from scripts.backtest_full_bot import SYMBOL_CONFIG
    result = []
    for sym, cfg in SYMBOL_CONFIG.items():
        result.append({
            "symbol": sym,
            "timeframe": cfg.get("timeframe", "15m"),
            "prob_threshold": cfg.get("prob_threshold", 0.55),
            "fixed_sl_pct": cfg.get("fixed_sl_pct", 0.025),
            "fixed_tp_pct": cfg.get("fixed_tp_pct", 0.040),
            "risk": cfg.get("risk", 0.010),
            "enabled": cfg.get("prob_threshold", 1.0) < 0.95,
            "skip_regime": cfg.get("skip_regime", True),
            "max_position_pct": 35,
        })
    return result


@router.get("/api/symbols/config")
async def get_symbols_config() -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_symbol_config)


@router.post("/api/symbols/{sym}/retrain")
async def retrain_symbol(sym: str, body: dict | None = None) -> StreamingResponse:
    """Trigger retrain for a single symbol and stream progress as SSE."""
    days = int((body or {}).get("days", 180))
    vibe = bool((body or {}).get("vibe", True))
    side = str((body or {}).get("side", "both"))

    symbol = sym.replace("_", "/").upper()
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "retrain_model.py"),
        "--symbols", symbol,
        "--days", str(days),
        "--side", side,
    ]
    if vibe:
        cmd.append("--vibe")

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'symbol': symbol})}\n\n"
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
