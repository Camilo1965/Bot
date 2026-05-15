"""
vibe.mcp_client
~~~~~~~~~~~~~~~
Async MCP client for Vibe-Trading tools.

Launches ``vibe-trading-mcp`` as a subprocess and communicates
via JSON-RPC 2.0 over stdio.  All methods are no-ops if the
binary is not installed or the subprocess fails to start.

Uses subprocess.Popen + threading for I/O so it works on Windows
with SelectorEventLoop (the bot's required event loop for MT5).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("clawdbot.vibe")

_MCP_BINARY = "vibe-trading-mcp"
_REQUEST_TIMEOUT = 90
_MAX_CONSECUTIVE_FAILURES = 2
_HEALTH_CHECK_INTERVAL = 60
_HEALTH_CHECK_TIMEOUT = 90
_RESTART_BACKOFF_BASE = 5
_TOOL_CALL_COOLDOWN = 60

_TOOL_TIMEOUTS: dict[str, float] = {
    "pattern_recognition":   120.0,
    "backtest":              180.0,
    "run_shadow_backtest":   180.0,
    "factor_analysis":       120.0,
    "run_swarm":             300.0,
    "analyze_trade_journal":  60.0,
    "get_market_data":        30.0,
    "web_search":             30.0,
    "extract_shadow_strategy": 60.0,
    "render_shadow_report":   60.0,
    "list_skills":            15.0,
    "list_swarm_presets":     15.0,
    "get_run_result":         15.0,
    "list_runs":              15.0,
    "read_file":              10.0,
    "write_file":             10.0,
}


def _resolve_mcp_binary() -> str | None:
    path = shutil.which(_MCP_BINARY)
    if path:
        return path
    exe_dir = Path(sys.executable).parent
    candidates = [exe_dir / f"{_MCP_BINARY}.exe", exe_dir / _MCP_BINARY]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


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
        self._io_lock = threading.Lock()

    async def start(self) -> bool:
        binary_path = _resolve_mcp_binary()
        if not binary_path:
            logger.warning("[VIBE] %s not found - Vibe-Trading tools disabled.", _MCP_BINARY)
            return False
        logger.info("[VIBE] Using MCP binary: %s", binary_path)

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
            logger.warning("[VIBE] agent/.env not found at %s.", agent_env)

        try:
            agent_dir = bot_root / "agent"
            self._proc = subprocess.Popen(
                [binary_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=env,
                cwd=str(agent_dir),
            )

            # Drain stderr in background thread so pipe doesn't block
            threading.Thread(target=self._stderr_drain, daemon=True).start()

            await asyncio.sleep(1)
            if self._proc.poll() is not None:
                stderr_tail = "\n".join(self._stderr_lines[-20:])
                self._init_error = f"vibe-trading-mcp exited with code {self._proc.returncode}.\nstderr:\n{stderr_tail}"
                logger.warning("[VIBE] %s", self._init_error)
                self._proc = None
                return False

            init_delay = float(os.environ.get("VIBE_MCP_INIT_DELAY_S", "2").strip() or "2")
            if init_delay > 0:
                await asyncio.sleep(init_delay)

            init_timeout = float(os.environ.get("VIBE_MCP_INIT_TIMEOUT_S", "30").strip() or "30")
            logger.info("[VIBE] Sending initialize (timeout=%ss)...", init_timeout)
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self._sync_call, "initialize", {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "clawdbot", "version": "1.0"},
                        }
                    ),
                    timeout=init_timeout
                )
            except Exception as exc:
                self._init_error = f"Initialize exception: {exc}"
                logger.warning("[VIBE] MCP initialize exception: %s", exc)
                await self.stop()
                return False

            if result:
                self._send_sync_notification("notifications/initialized", {})
                self._available = True
                self._consecutive_failures = 0
                self._restart_backoff = 0
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                logger.info("[VIBE] MCP client started successfully.")
                return True

            stderr_tail = "\n".join(self._stderr_lines[-20:])
            self._init_error = f"MCP initialize returned no result.\nstderr:\n{stderr_tail}"
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

    def _stderr_drain(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        try:
            while True:
                raw = self._proc.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
                    if len(self._stderr_lines) > 200:
                        self._stderr_lines[:] = self._stderr_lines[-100:]
                    lower = line.lower()
                    if "error" in lower or "traceback" in lower or "exception" in lower:
                        logger.warning("[VIBE MCP stderr] %s", line)
                    else:
                        logger.debug("[VIBE MCP stderr] %s", line)
        except Exception:
            pass

    def _send_sync_notification(self, method: str, params: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            return
        notification = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self._proc.stdin.write(json.dumps(notification).encode("utf-8") + b"\n")
            self._proc.stdin.flush()
        except Exception:
            pass

    def _sync_call(self, method: str, params: dict[str, Any]) -> Any | None:
        """Synchronous JSON-RPC call (runs in executor thread)."""
        if not self._proc or not self._proc.stdin or not self._proc.stdout:
            return None
        with self._io_lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
            line = json.dumps(request) + "\n"
            try:
                self._proc.stdin.write(line.encode("utf-8"))
                self._proc.stdin.flush()
            except Exception as exc:
                logger.warning("[VIBE] Failed to write to MCP stdin: %s", exc)
                return None

            while True:
                try:
                    response_line = self._proc.stdout.readline()
                except Exception as exc:
                    logger.warning("[VIBE] Error reading MCP stdout: %s", exc)
                    return None
                if not response_line:
                    logger.warning("[VIBE] MCP stdout EOF (method=%s).", method)
                    return None

                text = response_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    response = json.loads(text)
                except json.JSONDecodeError:
                    logger.debug("[VIBE] Non-JSON line: %s", text[:200])
                    continue

                if "id" in response and response.get("id") == self._request_id:
                    if "error" in response:
                        logger.warning("[VIBE] MCP error: %s", response["error"])
                        return None
                    return response.get("result")
                # Not our response ID - skip (could be server notification)

    async def _health_check_loop(self) -> None:
        while True:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
            if self._restarting or self._busy or not self._available or not self._proc:
                continue
            elapsed = asyncio.get_event_loop().time() - self._last_tool_call_time
            if 0 < elapsed < _TOOL_CALL_COOLDOWN:
                continue
            if self._proc.poll() is not None:
                stderr = "\n".join(self._stderr_lines[-5:])
                logger.warning("[VIBE] MCP process died (exit=%s) - triggering restart.", self._proc.returncode)
                await self._trigger_restart()
                continue
            if self._proc.stdout and self._proc.stdout.closed:
                logger.warning("[VIBE] MCP stdout pipe closed - triggering restart.")
                await self._trigger_restart()

    async def _trigger_restart(self) -> None:
        if self._restarting:
            return
        self._restarting = True
        self._available = False
        self._consecutive_failures = 0
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            try:
                await asyncio.wait_for(asyncio.to_thread(lambda: self._proc.wait(timeout=3)), timeout=4.0)
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
        logger.warning("[VIBE] Restarting MCP in %ds... (stderr: %s)", self._restart_backoff, stderr[:300])
        await asyncio.sleep(self._restart_backoff)
        success = await self.start()
        if success:
            self._restart_backoff = 0
            logger.info("[VIBE] MCP restarted successfully.")
        else:
            logger.warning("[VIBE] MCP restart failed - will retry.")
        self._restarting = False

    async def stop(self) -> None:
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
                await asyncio.wait_for(asyncio.to_thread(lambda: self._proc.wait(timeout=5)), timeout=6.0)
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

    async def _ensure_process_alive(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            logger.warning("[VIBE] MCP process dead detected mid-call - triggering restart.")
            await self._trigger_restart()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        max_retries: int = 2,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        if not self.available:
            logger.debug("[VIBE] Client not available - skipping tool '%s'.", tool_name)
            return None
        effective_timeout = timeout if timeout is not None else _TOOL_TIMEOUTS.get(tool_name, _REQUEST_TIMEOUT)
        self._busy = True
        try:
            for attempt in range(max_retries + 1):
                async with self._lock:
                    try:
                        result = await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                None, self._sync_call, "tools/call", {"name": tool_name, "arguments": arguments}
                            ),
                            timeout=effective_timeout
                        )
                    except asyncio.TimeoutError:
                        result = None
                if result is not None:
                    if attempt > 0:
                        logger.info("[VIBE] %s succeeded after %d retries.", tool_name, attempt)
                    return result
                if attempt < max_retries:
                    backoff = 2 ** (attempt + 1)
                    logger.warning("[VIBE] %s failed - retrying in %ds...", tool_name, backoff)
                    await asyncio.sleep(backoff)
                    await self._ensure_process_alive()
            logger.warning("[VIBE] %s failed after %d attempts.", tool_name, max_retries + 1)
            return None
        finally:
            self._busy = False
            self._last_tool_call_time = asyncio.get_event_loop().time()

    async def backtest(self, run_dir: str) -> dict | None:
        return await self.call_tool("backtest", {"run_dir": run_dir})

    async def analyze_trade_journal(self, file_path: str, analysis_type: str = "full", filter_expr: str = "") -> dict | None:
        args: dict[str, Any] = {"file_path": file_path, "analysis_type": analysis_type}
        if filter_expr:
            args["filter_expr"] = filter_expr
        return await self.call_tool("analyze_trade_journal", args)

    async def extract_shadow_strategy(self, journal_path: str, min_support: int = 3, max_rules: int = 5) -> dict | None:
        return await self.call_tool("extract_shadow_strategy", {
            "journal_path": journal_path, "min_support": min_support, "max_rules": max_rules,
        })

    async def run_shadow_backtest(self, shadow_id: str, window_start: str = "", window_end: str = "", markets: list[str] | None = None, journal_path: str = "") -> dict | None:
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

    async def render_shadow_report(self, shadow_id: str, include_today_signals: bool = True, window_start: str = "", window_end: str = "", journal_path: str = "") -> dict | None:
        args: dict[str, Any] = {"shadow_id": shadow_id, "include_today_signals": include_today_signals}
        if window_start:
            args["window_start"] = window_start
        if window_end:
            args["window_end"] = window_end
        if journal_path:
            args["journal_path"] = journal_path
        return await self.call_tool("render_shadow_report", args)

    async def pattern_recognition(self, run_dir: str, timeout: float | None = None) -> dict | None:
        return await self.call_tool("pattern_recognition", {"run_dir": run_dir}, timeout=timeout)

    async def factor_analysis(self, codes: list[str], factor_name: str, start_date: str, end_date: str, source: str = "auto", top_n: int = 10, bottom_n: int = 10) -> dict | None:
        return await self.call_tool("factor_analysis", {
            "codes": codes, "factor_name": factor_name, "start_date": start_date,
            "end_date": end_date, "source": source, "top_n": top_n, "bottom_n": bottom_n,
        })

    async def get_market_data(self, codes: list[str], start_date: str, end_date: str, source: str = "auto", interval: str = "1D") -> dict | None:
        return await self.call_tool("get_market_data", {
            "codes": codes, "start_date": start_date, "end_date": end_date,
            "source": source, "interval": interval,
        })

    async def list_skills(self) -> list | None:
        return await self.call_tool("list_skills", {})

    async def run_swarm(self, preset: str, variables: dict[str, Any] | None = None, timeout: float | None = None) -> dict | None:
        args: dict[str, Any] = {"preset_name": preset, "variables": variables if variables is not None else {}}
        return await self.call_tool("run_swarm", args, timeout=timeout)
