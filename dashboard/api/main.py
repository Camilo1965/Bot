"""
ClawdBot Dashboard API — FastAPI backend (Sprint 1)

Design tokens (for HTML/CSS frontend):
  bg:       #0A0E14
  surface:  #10151D
  elevated: #1E2530
  border:   #374151
  text:     #F3F4F6
  sub:      #9CA3AF
  green:    #10B981
  red:      #EF4444
  blue:     #3B82F6
  yellow:   #F59E0B

Run:
  uvicorn dashboard.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import socket

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

load_dotenv()

from dashboard.api.routers import state, positions, signals, performance, risk, models, alerts
from dashboard.api.routers import backtest as backtest_router
from dashboard.api.routers import symbols as symbols_router
from dashboard.api.routers import journal as journal_router
from dashboard.api.ws.stream import handle_ws, start_background_tasks

_API_KEY = os.environ.get("DASHBOARD_API_KEY", "dev-secret")

app = FastAPI(title="ClawdBot Dashboard API", version="1.0.0")

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Skip auth for root health check and WebSocket
    path = request.url.path
    if path in ("/", "/ws/stream") or path.startswith("/docs") or path.startswith("/openapi"):
        return await call_next(request)

    # If no key is configured or it is the default dev key, skip check
    if _API_KEY == "dev-secret" or not _API_KEY:
        return await call_next(request)

    provided = request.headers.get("X-Api-Key", "")
    if provided != _API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)


@app.on_event("startup")
async def on_startup() -> None:
    await start_background_tasks()


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "clawdbot-api"}


@app.get("/api/system/info")
async def system_info() -> dict:
    lan_ip = _get_local_ip()
    api_port = int(os.environ.get("DASHBOARD_API_PORT", "8000"))
    web_port = int(os.environ.get("DASHBOARD_WEB_PORT", "3000"))
    return {
        "lan_ip": lan_ip,
        "hostname": socket.gethostname(),
        "dashboard_url": f"http://{lan_ip}:{web_port}",
        "api_url": f"http://{lan_ip}:{api_port}",
        "web_port": web_port,
        "api_port": api_port,
        "execution_mode": os.environ.get("EXECUTION_MODE", "paper"),
    }


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await handle_ws(websocket)


# Register routers
app.include_router(state.router)
app.include_router(positions.router)
app.include_router(signals.router)
app.include_router(performance.router)
app.include_router(risk.router)
app.include_router(models.router)
app.include_router(alerts.router)
app.include_router(backtest_router.router)
app.include_router(symbols_router.router)
app.include_router(journal_router.router)
