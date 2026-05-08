"""
vibe.scheduled_tasks
~~~~~~~~~~~~~~~~~~~~
Background tasks that call Vibe-Trading tools on a schedule.
All tasks are no-ops if VibeMCPClient is not available.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient
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


async def journal_analysis_loop(
    client: VibeMCPClient,
    shared_state: dict[str, Any],
    interval_s: float = _JOURNAL_ANALYSIS_INTERVAL_S,
) -> None:
    """Periodically analyze the trade journal and store results in shared_state."""
    if not client.available:
        logger.info("[VIBE] Client not available — journal analysis disabled.")
        return
    await asyncio.sleep(60)
    while True:
        try:
            result = await analyze_journal(client)
            if result:
                shared_state["vibe_journal_analysis"] = result
                logger.info("[VIBE] Journal analysis stored in shared_state.")
        except Exception as exc:
            logger.warning("[VIBE] Journal analysis failed: %s", exc)
        await asyncio.sleep(interval_s)


async def weekly_backtest_loop(
    client: VibeMCPClient,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = _BACKTEST_INTERVAL_S,
) -> None:
    """Weekly: export DB data and run a backtest for each symbol.

    Results are stored in shared_state['vibe_backtest'] for the web dashboard.
    """
    if not client.available:
        return
    await asyncio.sleep(86400)
    while True:
        for symbol in watchlist:
            try:
                result = await client.backtest(
                    "Backtest {} using MACD + RSI strategy, "
                    "last 30 days, 15m timeframe. Show Sharpe ratio and max drawdown.".format(symbol)
                )
                if result:
                    shared_state.setdefault("vibe_backtest", {})[symbol] = result
                    logger.info("[VIBE] Backtest complete for %s.", symbol)
            except Exception as exc:
                logger.warning("[VIBE] Backtest failed for %s: %s", symbol, exc)
        await asyncio.sleep(interval_s)


async def pattern_detection_loop(
    client: VibeMCPClient,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = _PATTERN_INTERVAL_S,
) -> None:
    """Periodically detect technical patterns for each watched symbol."""
    if not client.available:
        return
    await asyncio.sleep(120)
    while True:
        for symbol in watchlist:
            try:
                result = await detect_patterns(client, symbol)
                if result:
                    shared_state.setdefault("vibe_patterns", {})[symbol] = result
            except Exception as exc:
                logger.warning("[VIBE] Pattern detection failed for %s: %s", symbol, exc)
        await asyncio.sleep(interval_s)


async def factor_analysis_loop(
    client: VibeMCPClient,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = _FACTOR_INTERVAL_S,
) -> None:
    """Monthly: run IC/IR factor analysis for each symbol."""
    if not client.available:
        return
    await asyncio.sleep(3600)
    while True:
        for symbol in watchlist:
            try:
                result = await analyze_factors(client, symbol)
                if result:
                    shared_state.setdefault("vibe_factors", {})[symbol] = result
                    logger.info("[VIBE] Factor analysis complete for %s.", symbol)
            except Exception as exc:
                logger.warning("[VIBE] Factor analysis failed for %s: %s", symbol, exc)
        await asyncio.sleep(interval_s)


async def shadow_account_loop(
    client: VibeMCPClient,
    shared_state: dict[str, Any],
    interval_s: float = _SHADOW_INTERVAL_S,
) -> None:
    """Weekly: generate shadow account report from trade journal."""
    if not client.available:
        return
    await asyncio.sleep(172800)
    while True:
        try:
            result = await extract_and_backtest_shadow(client)
            if result:
                shared_state["vibe_shadow_report"] = result
                logger.info("[VIBE] Shadow account report generated.")
        except Exception as exc:
            logger.warning("[VIBE] Shadow account failed: %s", exc)
        await asyncio.sleep(interval_s)