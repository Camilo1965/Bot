"""
vibe.scheduled_tasks
~~~~~~~~~~~~~~~~~~~~
Background tasks that call Vibe-Trading tools on a schedule.
All tasks are no-ops if VibeMCPClient is not available.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing import Any

from vibe.journal_analyzer import analyze_journal
from vibe.backtest import run_backtest
from vibe.pattern_recognition import detect_patterns
from vibe.factor_research import analyze_factors
from vibe.shadow_account import extract_and_backtest_shadow

logger = logging.getLogger("clawdbot.vibe.tasks")

_JOURNAL_ANALYSIS_INTERVAL_S = 86400
_BACKTEST_INTERVAL_S = 604800
_PATTERN_INTERVAL_S = 900
_FACTOR_INTERVAL_S = 2592000
_SHADOW_INTERVAL_S = 604800

_VIBE_STATE_FILE = Path("logs") / "vibe_state.json"


def _load_vibe_state() -> dict[str, Any]:
    if _VIBE_STATE_FILE.is_file():
        try:
            return json.loads(_VIBE_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_vibe_state(state: dict[str, Any]) -> None:
    try:
        _VIBE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VIBE_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("[VIBE] Could not save state: %s", exc)


def _initial_sleep_with_state(loop_name: str, default_sleep: float, interval_s: float) -> float:
    """Return sleep seconds respecting last_run timestamp from persisted state."""
    state = _load_vibe_state()
    last_run = state.get(f"last_run_{loop_name}")
    if last_run is not None:
        try:
            elapsed = (datetime.now(tz=timezone.utc) - datetime.fromisoformat(last_run)).total_seconds()
            remaining = interval_s - elapsed
            if remaining > 5:
                return remaining
        except Exception:
            pass
    return default_sleep


def _record_run(loop_name: str) -> None:
    state = _load_vibe_state()
    state[f"last_run_{loop_name}"] = datetime.now(tz=timezone.utc).isoformat()
    _save_vibe_state(state)


async def journal_analysis_loop(
    client: Any,
    shared_state: dict[str, Any],
    interval_s: float = _JOURNAL_ANALYSIS_INTERVAL_S,
) -> None:
    """Periodically analyze the trade journal and store results in shared_state."""
    if not client.available:
        logger.info("[VIBE] Client not available - journal analysis disabled.")
        return
    await asyncio.sleep(_initial_sleep_with_state("journal", 60, interval_s))
    while True:
        try:
            result = await analyze_journal(client)
            if result:
                shared_state["vibe_journal_analysis"] = result
                logger.info("[VIBE] Journal analysis stored in shared_state.")
                _record_run("journal")
        except Exception as exc:
            logger.warning("[VIBE] Journal analysis failed: %s", exc)
        await asyncio.sleep(interval_s)


async def weekly_backtest_loop(
    client: Any,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = _BACKTEST_INTERVAL_S,
) -> None:
    """Weekly: run a backtest for each symbol via Vibe-Trading.

    Creates a run directory with config.json and signal_engine.py for
    each symbol, then calls the backtest MCP tool.

    Results are stored in shared_state['vibe_backtest'] for the web dashboard.
    """
    if not client.available:
        return
    await asyncio.sleep(_initial_sleep_with_state("backtest", 86400, interval_s))
    while True:
        for symbol in watchlist:
            try:
                result = await run_backtest(client, symbol)
                if result:
                    shared_state.setdefault("vibe_backtest", {})[symbol] = result
                    logger.info("[VIBE] Backtest complete for %s.", symbol)
            except Exception as exc:
                logger.warning("[VIBE] Backtest failed for %s: %s", symbol, exc)
        _record_run("backtest")
        await asyncio.sleep(interval_s)


async def pattern_detection_loop(
    client: Any,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = _PATTERN_INTERVAL_S,
) -> None:
    """Periodically detect technical patterns for each watched symbol."""
    if not client.available:
        return
    await asyncio.sleep(_initial_sleep_with_state("pattern", 120, interval_s))
    while True:
        for symbol in watchlist:
            try:
                result = await detect_patterns(client, symbol)
                if result:
                    shared_state.setdefault("vibe_patterns", {})[symbol] = result
            except Exception as exc:
                logger.warning("[VIBE] Pattern detection failed for %s: %s", symbol, exc)
        _record_run("pattern")
        await asyncio.sleep(interval_s)


async def factor_analysis_loop(
    client: Any,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = _FACTOR_INTERVAL_S,
) -> None:
    """Monthly: run IC/IR factor analysis across the watchlist."""
    if not client.available:
        return
    await asyncio.sleep(_initial_sleep_with_state("factor", 3600, interval_s))
    while True:
        try:
            result = await analyze_factors(client, symbols=watchlist)
            if result:
                shared_state["vibe_factors"] = result
                logger.info("[VIBE] Factor analysis complete for watchlist.")
                _record_run("factor")
        except Exception as exc:
            logger.warning("[VIBE] Factor analysis failed: %s", exc)
        await asyncio.sleep(interval_s)


async def shadow_account_loop(
    client: Any,
    shared_state: dict[str, Any],
    interval_s: float = _SHADOW_INTERVAL_S,
) -> None:
    """Weekly: generate shadow account report from trade journal."""
    if not client.available:
        return
    await asyncio.sleep(_initial_sleep_with_state("shadow", 172800, interval_s))
    while True:
        try:
            result = await extract_and_backtest_shadow(client)
            if result:
                shared_state["vibe_shadow_report"] = result
                logger.info("[VIBE] Shadow account report generated.")
                _record_run("shadow")
        except Exception as exc:
            logger.warning("[VIBE] Shadow account failed: %s", exc)
        await asyncio.sleep(interval_s)