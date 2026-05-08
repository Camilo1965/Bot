"""
vibe.backtest
~~~~~~~~~~~~~
Runs Vibe-Trading backtest engine using historical data
from Vibe-Trading's data sources (ccxt, yfinance, etc.).

Creates a run directory with config.json and code/signal_engine.py,
then calls the backtest MCP tool with the run_dir path.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from vibe.mcp_client import VibeMCPClient

logger = logging.getLogger("clawdbot.vibe.backtest")

_RUN_DIR_BASE = Path("logs") / "vibe_runs"

_BACKTEST_CONFIG_TEMPLATE: dict[str, Any] = {
    "source": "ccxt",
    "codes": [],
    "start_date": "",
    "end_date": "",
    "initial_capital": 10000,
    "commission": 0.001,
    "timeframe": "15m",
}

_SIGNAL_ENGINE_CODE = '''\
"""MACD crossover + RSI filter signal engine for Vibe-Trading backtest."""

def signal(df):
    """Return 1 (buy), -1 (sell), or 0 (flat) based on MACD + RSI."""
    import pandas as pd

    close = df["close"]
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    signal_line = macd.ewm(span=9).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = delta.abs().rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    if macd.iloc[-1] > signal_line.iloc[-1] and rsi.iloc[-1] < 70:
        return 1
    elif macd.iloc[-1] < signal_line.iloc[-1] and rsi.iloc[-1] > 30:
        return -1
    return 0
'''


async def run_backtest(
    client: VibeMCPClient,
    symbol: str,
    source: str = "ccxt",
    days_back: int = 30,
    timeframe: str = "15m",
) -> dict[str, Any] | None:
    """Run a backtest for a symbol via Vibe-Trading's backtest MCP tool.

    Creates a run directory with config.json and code/signal_engine.py,
    then calls the backtest tool with the run_dir path.

    Returns backtest results dict or None.
    """
    _RUN_DIR_BASE.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = symbol.replace("/", "_")
    run_dir = _RUN_DIR_BASE / f"backtest_{safe_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    config = {
        **_BACKTEST_CONFIG_TEMPLATE,
        "source": source,
        "codes": [symbol],
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "timeframe": timeframe,
    }

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    signal_path = code_dir / "signal_engine.py"
    signal_path.write_text(_SIGNAL_ENGINE_CODE, encoding="utf-8")

    logger.info("[VIBE] Created backtest run_dir: %s", run_dir)
    return await client.backtest(run_dir=str(run_dir.resolve()))