"""
vibe.mcp_client
~~~~~~~~~~~~~~~
Async MCP client for Vibe-Trading tools.

Launches ``vibe-trading-mcp`` as a subprocess and communicates
via JSON-RPC 2.0 over stdio.  All methods are no-ops if the
binary is not installed or the subprocess fails to start.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

logger = logging.getLogger("clawdbot.vibe")

_MCP_BINARY = "vibe-trading-mcp"
_REQUEST_TIMEOUT = 120


class VibeMCPClient:
    """Manages the MCP subprocess lifecycle and tool calls."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._available = False
        self._lock = asyncio.Lock()
        self._reader_lock = asyncio.Lock()

    async def start(self) -> bool:
        """Start the MCP subprocess.  Returns True if successful."""
        if not shutil.which(_MCP_BINARY):
            logger.warning(
                "[VIBE] %s not found on PATH — Vibe-Trading tools disabled.",
                _MCP_BINARY,
            )
            return False
        try:
            import os
            env = dict(os.environ)
            self._proc = await asyncio.create_subprocess_exec(
                _MCP_BINARY,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            result = await self._call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "clawdbot", "version": "1.0"},
                },
            )
            if result:
                await self._notify("notifications/initialized", {})
                self._available = True
                logger.info("[VIBE] MCP client started successfully.")
                return True
            logger.warning("[VIBE] MCP initialize failed — tools disabled.")
            await self.stop()
            return False
        except Exception as exc:
            logger.warning("[VIBE] MCP client start failed: %s — tools disabled.", exc)
            self._proc = None
            return False

    async def stop(self) -> None:
        """Gracefully shut down the MCP subprocess."""
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
            self._proc = None
        self._available = False
        logger.info("[VIBE] MCP client stopped.")

    @property
    def available(self) -> bool:
        return self._available and self._proc is not None and self._proc.returncode is None

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
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()
        raw = await asyncio.wait_for(
            self._proc.stdout.readline(), timeout=_REQUEST_TIMEOUT
        )
        if not raw:
            return None
        response = json.loads(raw.decode("utf-8"))
        if "error" in response:
            logger.warning("[VIBE] MCP error: %s", response["error"])
            return None
        return response.get("result")

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
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call an MCP tool by name.  Returns the tool result dict or None."""
        if not self.available:
            logger.debug("[VIBE] Client not available — skipping tool '%s'.", tool_name)
            return None
        async with self._lock:
            try:
                return await self._call("tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                })
            except Exception as exc:
                logger.warning("[VIBE] Tool '%s' call failed: %s", tool_name, exc)
                return None

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