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
_REQUEST_TIMEOUT = 300
_MAX_CONSECUTIVE_FAILURES = 2
_HEALTH_CHECK_INTERVAL = 120
_HEALTH_CHECK_TIMEOUT = 90
_RESTART_BACKOFF_BASE = 5
_TOOL_CALL_COOLDOWN = 60


class VibeMCPClient:
    """Manages the MCP subprocess lifecycle and tool calls."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._available = False
        self._lock = asyncio.Lock()
        self._init_error: str | None = None
        self._stderr_lines: list[str] = []
        self._consecutive_failures = 0
        self._restart_backoff = 0
        self._health_check_task: asyncio.Task | None = None
        self._restarting = False
        self._busy = False
        self._last_tool_call_time = 0.0

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
            try:
                result = await self._call("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "clawdbot", "version": "1.0"},
                })
            except Exception as exc:
                self._init_error = f"Initialize exception: {exc}"
                logger.warning("[VIBE] MCP initialize exception: %s", exc)
                await self.stop()
                return False
            if result:
                await self._notify("notifications/initialized", {})
                self._available = True
                self._consecutive_failures = 0
                self._restart_backoff = 0
                self._health_check_task = asyncio.create_task(self._health_check_loop())
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

    async def _health_check_loop(self) -> None:
        """Background health check: restart MCP if process dies or becomes unresponsive."""
        while True:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
            if self._restarting:
                continue
            if self._busy:
                continue
            if not self._available or not self._proc:
                continue

            # Cooldown: skip check right after a tool call finishes
            elapsed = asyncio.get_event_loop().time() - self._last_tool_call_time
            if 0 < elapsed < _TOOL_CALL_COOLDOWN:
                continue

            # Only check if the subprocess is still alive — no stdin/stdout traffic
            if self._proc.poll() is not None:
                stderr = "\n".join(self._stderr_lines[-5:])
                logger.warning(
                    "[VIBE] MCP process died (exit=%s, stderr: %s) — triggering restart.",
                    self._proc.returncode,
                    stderr[:300],
                )
                await self._trigger_restart()
                continue

            # Verify stdout pipe is still open (process alive but pipe broken)
            if self._proc.stdout and self._proc.stdout.closed:
                logger.warning("[VIBE] MCP stdout pipe closed — triggering restart.")
                await self._trigger_restart()

    async def _trigger_restart(self) -> None:
        """Trigger MCP restart with backoff."""
        if self._restarting:
            return
        self._restarting = True
        self._available = False
        self._consecutive_failures = 0

        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(lambda: self._proc.wait(timeout=3)),
                    timeout=4.0,
                )
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                pass
            self._proc = None

        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await asyncio.wait_for(self._health_check_task, timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._health_check_task = None

        if self._restart_backoff == 0:
            self._restart_backoff = _RESTART_BACKOFF_BASE
        else:
            self._restart_backoff = min(self._restart_backoff * 2, 60)

        stderr = "\n".join(self._stderr_lines[-5:])
        logger.warning(
            "[VIBE] Restarting MCP in %ds... (stderr: %s)",
            self._restart_backoff,
            stderr[:300],
        )
        await asyncio.sleep(self._restart_backoff)

        success = await self.start()
        if success:
            self._restart_backoff = 0
            logger.warning("[VIBE] ✅ MCP restarted successfully.")
        else:
            logger.warning("[VIBE] MCP restart failed - will retry.")

        self._restarting = False

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
                    # Log errors/warnings at WARNING, rest at DEBUG
                    lower = line.lower()
                    if "error" in lower or "traceback" in lower or "exception" in lower:
                        logger.warning("[VIBE MCP stderr] %s", line)
                    else:
                        logger.debug("[VIBE MCP stderr] %s", line)
        except Exception:
            pass

    async def stop(self) -> None:
        """Gracefully shut down the MCP subprocess."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await asyncio.wait_for(self._health_check_task, timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._health_check_task = None

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
            except Exception as exc:
                logger.warning("[VIBE] Error reading MCP stdout: %s", exc)
                return None
            if not raw:
                logger.warning("[VIBE] MCP stdout EOF (method=%s) — subprocess likely died.", method)
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

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], max_retries: int = 2) -> dict[str, Any] | None:
        """Call an MCP tool with retry. Returns the tool result dict or None."""
        if not self.available:
            logger.debug("[VIBE] Client not available - skipping tool '%s'.", tool_name)
            return None

        self._busy = True
        try:
            last_error: str | None = None
            for attempt in range(max_retries + 1):
                async with self._lock:
                    result = await self._call("tools/call", {
                        "name": tool_name,
                        "arguments": arguments,
                    })
                if result is not None:
                    if attempt > 0:
                        logger.info("[VIBE] %s succeeded after %d retries.", tool_name, attempt)
                    return result
                last_error = f"Attempt {attempt + 1}/{max_retries + 1} failed"
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.warning("[VIBE] %s failed - retrying in %ds...", tool_name, backoff)
                    await asyncio.sleep(backoff)

            logger.warning("[VIBE] %s failed after %d attempts.", tool_name, max_retries + 1)
            return None
        finally:
            self._busy = False
            self._last_tool_call_time = asyncio.get_event_loop().time()

    async def backtest(self, run_dir: str) -> dict | None:
        return await self.call_tool("backtest", {"run_dir": run_dir})

    async def analyze_trade_journal(
        self,
        file_path: str,
        analysis_type: str = "full",
        filter_expr: str = "",
    ) -> dict | None:
        args: dict[str, Any] = {"file_path": file_path, "analysis_type": analysis_type}
        if filter_expr:
            args["filter_expr"] = filter_expr
        return await self.call_tool("analyze_trade_journal", args)

    async def extract_shadow_strategy(
        self,
        journal_path: str,
        min_support: int = 3,
        max_rules: int = 5,
    ) -> dict | None:
        return await self.call_tool("extract_shadow_strategy", {
            "journal_path": journal_path,
            "min_support": min_support,
            "max_rules": max_rules,
        })

    async def run_shadow_backtest(
        self,
        shadow_id: str,
        window_start: str = "",
        window_end: str = "",
        markets: list[str] | None = None,
        journal_path: str = "",
    ) -> dict | None:
        args: dict[str, Any] = {"shadow_id": shadow_id}
        if window_start:
            args["window_start"] = window_start
        if window_end:
            args["window_end"] = window_end
        if markets:
            args["markets"] = markets
        if journal_path:
            args["journal_path"] = journal_path
        return await self.call_tool("run_shadow_backtest", args)

    async def render_shadow_report(
        self,
        shadow_id: str,
        include_today_signals: bool = True,
        window_start: str = "",
        window_end: str = "",
        journal_path: str = "",
    ) -> dict | None:
        args: dict[str, Any] = {
            "shadow_id": shadow_id,
            "include_today_signals": include_today_signals,
        }
        if window_start:
            args["window_start"] = window_start
        if window_end:
            args["window_end"] = window_end
        if journal_path:
            args["journal_path"] = journal_path
        return await self.call_tool("render_shadow_report", args)

    async def pattern_recognition(self, run_dir: str) -> dict | None:
        return await self.call_tool("pattern_recognition", {"run_dir": run_dir})

    async def factor_analysis(
        self,
        codes: list[str],
        factor_name: str,
        start_date: str,
        end_date: str,
        source: str = "auto",
        top_n: int = 10,
        bottom_n: int = 10,
    ) -> dict | None:
        return await self.call_tool("factor_analysis", {
            "codes": codes,
            "factor_name": factor_name,
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
            "top_n": top_n,
            "bottom_n": bottom_n,
        })

    async def get_market_data(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        source: str = "auto",
        interval: str = "1D",
    ) -> dict | None:
        return await self.call_tool("get_market_data", {
            "codes": codes,
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
            "interval": interval,
        })

    async def list_skills(self) -> list | None:
        return await self.call_tool("list_skills", {})