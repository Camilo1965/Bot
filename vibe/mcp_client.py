"""
vibe.mcp_client
~~~~~~~~~~~~~~~
Async MCP client for Vibe-Trading tools.

Launches ``vibe-trading-mcp`` as a subprocess and communicates
via JSON-RPC 2.0 over stdio.  All methods are no-ops if the
binary is not installed or the subprocess fails to start.

Uses subprocess.Popen + asyncio.to_thread() instead of
asyncio.create_subprocess_exec so it works on Windows with
SelectorEventLoop (the bot's required event loop for MT5).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("clawdbot.vibe")

_MCP_BINARY = "vibe-trading-mcp"
_REQUEST_TIMEOUT = 180


class VibeMCPClient:
    """Manages the MCP subprocess lifecycle and tool calls."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._available = False
        self._lock = asyncio.Lock()
        self._init_error: str | None = None
        self._stderr_lines: list[str] = []

    async def start(self) -> bool:
        """Start the MCP subprocess.  Returns True if successful."""
        if not shutil.which(_MCP_BINARY):
            logger.warning(
                "[VIBE] %s not found on PATH - Vibe-Trading tools disabled.",
                _MCP_BINARY,
            )
            return False

        env = dict(os.environ)

        bot_root = Path(__file__).resolve().parent.parent
        agent_env = bot_root / "agent" / ".env"
        if agent_env.is_file():
            logger.info("[VIBE] Found agent/.env at %s", agent_env)
            try:
                for line in agent_env.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, _, v = line.partition("=")
                        k, v = k.strip(), v.strip()
                        if k and k not in env:
                            env[k] = v
            except Exception as exc:
                logger.warning("[VIBE] Could not read agent/.env: %s", exc)
        else:
            logger.warning(
                "[VIBE] agent/.env not found at %s - "
                "Vibe-Trading will rely on process env vars only.",
                agent_env,
            )

        try:
            self._proc = subprocess.Popen(
                [_MCP_BINARY],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Drain stderr in background so it doesn't block
            asyncio.create_task(self._drain_stderr())

            # Give the subprocess a moment to start (1s, not blocking Ctrl+C)
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self._proc.kill()
                self._proc = None
                raise

            # Check if process is still alive
            if self._proc.poll() is not None:
                stderr_tail = "\n".join(self._stderr_lines[-20:])
                self._init_error = (
                    f"vibe-trading-mcp exited with code {self._proc.returncode}.\n"
                    f"stderr tail:\n{stderr_tail}"
                )
                logger.warning("[VIBE] %s", self._init_error)
                self._proc = None
                return False

            # Send initialize
            result = await self._call("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "clawdbot", "version": "1.0"},
            })
            if result:
                await self._notify("notifications/initialized", {})
                self._available = True
                logger.warning("[VIBE] ✅ MCP client started successfully.")
                return True

            stderr_tail = "\n".join(self._stderr_lines[-20:])
            self._init_error = (
                "MCP initialize returned no result.\n"
                f"stderr tail:\n{stderr_tail}"
            )
            logger.warning("[VIBE] %s", self._init_error)
            await self.stop()
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("[VIBE] MCP client start failed: %s - tools disabled.", exc)
            self._proc = None
            return False

    async def _drain_stderr(self) -> None:
        """Background task: read stderr from MCP subprocess and log it."""
        if not self._proc or not self._proc.stderr:
            return
        try:
            while True:
                raw = await asyncio.to_thread(self._proc.stderr.readline)
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 200:
                        self._stderr_lines = self._stderr_lines[-100:]
                    logger.debug("[VIBE MCP stderr] %s", line)
        except Exception:
            pass

    async def stop(self) -> None:
        """Gracefully shut down the MCP subprocess."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(lambda: self._proc.wait(timeout=5)),
                    timeout=6.0,
                )
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                self._proc.kill()
            self._proc = None
        self._available = False
        logger.info("[VIBE] MCP client stopped.")

    @property
    def available(self) -> bool:
        return self._available and self._proc is not None and self._proc.poll() is None

    @property
    def last_error(self) -> str | None:
        return self._init_error

    async def _call(self, method: str, params: dict[str, Any]) -> Any | None:
        """Send a JSON-RPC request and return the result."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            return None
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(request) + "\n"
        try:
            await asyncio.to_thread(self._proc.stdin.write, line.encode("utf-8"))
            await asyncio.to_thread(self._proc.stdin.flush)
        except Exception as exc:
            logger.warning("[VIBE] Failed to write to MCP stdin: %s", exc)
            return None

        # Read lines until we get a valid JSON-RPC response with our ID
        deadline = asyncio.get_event_loop().time() + _REQUEST_TIMEOUT
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning("[VIBE] Timeout waiting for MCP response (method=%s).", method)
                return None
            try:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(self._proc.stdout.readline),
                    timeout=min(remaining, 30),
                )
            except asyncio.TimeoutError:
                logger.warning("[VIBE] Timeout reading MCP response line.")
                return None
            if not raw:
                return None
            decoded = raw.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue
            try:
                response = json.loads(decoded)
            except json.JSONDecodeError:
                logger.debug("[VIBE] Non-JSON line from MCP: %s", decoded[:200])
                continue
            if "id" in response and response.get("id") == self._request_id:
                if "error" in response:
                    logger.warning("[VIBE] MCP error: %s", response["error"])
                    return None
                return response.get("result")
            # Not our response ID - skip it
            continue

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._proc or not self._proc.stdin:
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        line = json.dumps(notification) + "\n"
        try:
            await asyncio.to_thread(self._proc.stdin.write, line.encode("utf-8"))
            await asyncio.to_thread(self._proc.stdin.flush)
        except Exception:
            pass

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call an MCP tool by name.  Returns the tool result dict or None."""
        if not self.available:
            logger.debug("[VIBE] Client not available - skipping tool '%s'.", tool_name)
            return None
        async with self._lock:
            return await self._call("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })

    async def backtest(self, prompt: str) -> dict | None:
        return await self.call_tool("backtest", {"prompt": prompt})

    async def analyze_trade_journal(self, csv_path: str) -> dict | None:
        return await self.call_tool("analyze_trade_journal", {"file_path": csv_path})

    async def extract_shadow_strategy(self, csv_path: str) -> dict | None:
        return await self.call_tool("extract_shadow_strategy", {"file_path": csv_path})

    async def run_shadow_backtest(self, prompt: str) -> dict | None:
        return await self.call_tool("run_shadow_backtest", {"prompt": prompt})

    async def render_shadow_report(self, run_id: str) -> dict | None:
        return await self.call_tool("render_shadow_report", {"run_id": run_id})

    async def pattern_recognition(self, symbol: str, prompt: str) -> dict | None:
        return await self.call_tool("pattern", {"symbol": symbol, "prompt": prompt})

    async def factor_analysis(self, prompt: str) -> dict | None:
        return await self.call_tool("factor_analysis", {"prompt": prompt})

    async def get_market_data(self, symbol: str, start_date: str, end_date: str) -> dict | None:
        return await self.call_tool("get_market_data", {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        })

    async def list_skills(self) -> list | None:
        return await self.call_tool("list_skills", {})