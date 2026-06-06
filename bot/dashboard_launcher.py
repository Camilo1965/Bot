"""
Auto-launch the FastAPI dashboard API (uvicorn) and Next.js web server
as managed subprocesses when the bot starts.

Configuration via .env:
  DASHBOARD_HOST=0.0.0.0          # bind address (0.0.0.0 = all interfaces)
  DASHBOARD_API_PORT=8000
  DASHBOARD_WEB_PORT=3000
  DASHBOARD_API_URL=http://localhost:8000   # URL the browser uses to reach the API
                                             # Set to http://<YOUR_IP>:8000 for remote access
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO    = Path(__file__).resolve().parent.parent
_WEB_DIR = _REPO / "dashboard" / "web"

_HOST     = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
_API_PORT = int(os.environ.get("DASHBOARD_API_PORT", "8000"))
_WEB_PORT = int(os.environ.get("DASHBOARD_WEB_PORT", "3000"))
_API_URL  = os.environ.get("DASHBOARD_API_URL", f"http://localhost:{_API_PORT}")


def _get_local_ip() -> str:
    """Return the machine's LAN IP (the one reachable from the local network)."""
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


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _write_env_local(api_url: str) -> None:
    """Write NEXT_PUBLIC_API_URL to dashboard/web/.env.local."""
    (_WEB_DIR / ".env.local").write_text(f"NEXT_PUBLIC_API_URL={api_url}\n", encoding="utf-8")


async def _start_api_inproc() -> asyncio.Task | None:
    """Run uvicorn in-process so it shares the bot's fakeredis instance.
    Returns the asyncio task; None if port already bound."""
    if _port_in_use(_API_PORT):
        logger.info("[DASH] API port %s already bound — skipping uvicorn launch", _API_PORT)
        return None
    import uvicorn
    from dashboard.api.main import app
    config = uvicorn.Config(
        app,
        host=_HOST,
        port=_API_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    logger.info("[DASH] Starting API server in-process → http://%s:%s", _HOST, _API_PORT)
    return asyncio.create_task(server.serve(), name="uvicorn_dashboard_api")


def _start_web(api_url: str) -> subprocess.Popen | None:
    if not (_WEB_DIR / "package.json").exists():
        logger.info("[DASH] dashboard/web/package.json not found — skipping web server")
        return None
    if _port_in_use(_WEB_PORT):
        logger.info("[DASH] Web port %s already bound — skipping Next.js launch", _WEB_PORT)
        return None
    _write_env_local(api_url)
    built = (_WEB_DIR / ".next" / "BUILD_ID").exists()
    script = "start" if built else "dev"
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    cmd = [npm, "run", script]
    logger.info("[DASH] Starting web server (%s) → http://%s:%s", script, _HOST, _WEB_PORT)
    return subprocess.Popen(cmd, cwd=str(_WEB_DIR))


def _print_access_banner(lan_ip: str, api_url: str) -> None:
    lines = [
        "",
        "  +------------------------------------------------------+",
        "  |        CLAWDBOT DASHBOARD - ACCESO REMOTO            |",
        "  +------------------------------------------------------+",
       f"  | [WEB] Red local:  http://{lan_ip}:{_WEB_PORT}",
       f"  | [WEB] Localhost:  http://localhost:{_WEB_PORT}",
       f"  | [API]             {api_url}",
        "  +------------------------------------------------------+",
       f"  | Internet: DASHBOARD_API_URL=http://<IP_PUBLICA>:{_API_PORT}",
        "  +------------------------------------------------------+",
        "",
    ]
    for line in lines:
        logger.info(line)


async def start_dashboard_servers() -> None:
    """Launch API + web servers, detect LAN IP, print access banner."""
    lan_ip = _get_local_ip()

    # If DASHBOARD_API_URL is still the default localhost, auto-set to LAN IP
    # so the Next.js frontend can talk to the API from other devices.
    api_url = _API_URL
    if "localhost" in api_url or "127.0.0.1" in api_url:
        api_url = f"http://{lan_ip}:{_API_PORT}"
        logger.info("[DASH] Auto-detected LAN IP %s — using as API URL", lan_ip)

    api_task = await _start_api_inproc()
    if api_task is not None:
        await asyncio.sleep(3)   # let uvicorn bind before Next.js starts

    web_proc = _start_web(api_url)

    _print_access_banner(lan_ip, api_url)

    while True:
        await asyncio.sleep(30)
        if api_task is not None and api_task.done():
            exc = api_task.exception() if not api_task.cancelled() else None
            logger.warning("[DASH] api server exited (exc=%s) — restarting", exc)
            api_task = await _start_api_inproc()
        if web_proc is not None and web_proc.poll() is not None:
            logger.warning("[DASH] web server exited (code %s) — restarting", web_proc.returncode)
            web_proc = _start_web(api_url)
