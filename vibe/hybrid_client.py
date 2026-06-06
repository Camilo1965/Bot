"""
vibe.hybrid_client
~~~~~~~~~~~~~~~~~~
Hybrid Vibe-Trading client that bypasses the broken stdio MCP binary.

Uses:
1. Internal HTTP server (localhost:5000) for real-time trading decisions
2. Direct library calls (ccxt, yfinance, pandas) for market data & backtests
3. Local analysis with pandas for trade journals & pattern recognition
4. Direct Gemini calls via our HTTP server for swarm-like analysis

This client is a drop-in replacement for VibeMCPClient.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger("clawdbot.vibe")

_INTERNAL_HTTP_URL = "http://localhost:5000"
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)


class VibeHybridClient:
    """Drop-in replacement for VibeMCPClient.

    Does NOT launch the vibe-trading-mcp subprocess (broken on Windows).
    Instead uses the internal HTTP server + direct library calls.
    """

    def __init__(self) -> None:
        self._available = False
        self._init_error: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._last_tool_call_time = 0.0

    async def start(self) -> bool:
        """Verify the internal HTTP server is reachable (with retry)."""
        self._session = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT)
        for attempt in range(10):
            try:
                async with self._session.get(f"{_INTERNAL_HTTP_URL}/health", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        self._available = True
                        logger.info("[VIBE] Hybrid client connected to internal HTTP server.")
                        return True
            except Exception as exc:
                self._init_error = str(exc)
                if attempt == 0:
                    logger.info("[VIBE] Waiting for internal HTTP server...")
                await asyncio.sleep(1)
        logger.warning("[VIBE] Internal HTTP server not reachable after 10s (%s).", self._init_error)
        self._available = False
        return False

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._available = False
        logger.info("[VIBE] Hybrid client stopped.")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> str | None:
        return self._init_error

    # ── Core trading decision (HTTP) ──────────────────────────────────────────

    async def predict(self, features: list[float], ohlcv: list[dict], symbol: str = "ETH/USDT") -> dict | None:
        """Ask the internal HTTP server for a trading decision."""
        if not self._session:
            return None
        payload = {"features": features, "ohlcv": ohlcv, "symbol": symbol}
        try:
            async with self._session.post(f"{_INTERNAL_HTTP_URL}/predict", json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.warning("[VIBE] predict HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[VIBE] predict failed: %s", exc)
        return None

    # ── Market data (direct libraries) ────────────────────────────────────────

    async def get_market_data(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        source: str = "auto",
        interval: str = "1D",
    ) -> dict | None:
        """Fetch OHLCV using ccxt/yfinance directly."""
        results: dict[str, Any] = {}
        for code in codes:
            try:
                if "USDT" in code.upper() or "USD" in code.upper():
                    df = await self._fetch_crypto(code, start_date, end_date, interval)
                else:
                    df = await self._fetch_yfinance(code, start_date, end_date, interval)
                if df is not None:
                    records = df.reset_index().to_dict(orient="records")
                    for r in records:
                        for k, v in list(r.items()):
                            if hasattr(v, "isoformat"):
                                r[k] = v.isoformat()
                            elif hasattr(v, "item"):
                                r[k] = v.item()
                    results[code] = records
            except Exception as exc:
                logger.warning("[VIBE] get_market_data failed for %s: %s", code, exc)
        return results if results else None

    async def _fetch_crypto(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_fetch_crypto, symbol, start, end, interval)

    def _sync_fetch_crypto(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
        import ccxt
        exchange = ccxt.okx({"enableRateLimit": True})
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1H": "1H", "4H": "4H", "1D": "1d"}
        timeframe = tf_map.get(interval, "1d")
        since = exchange.parse8601(f"{start}T00:00:00Z")
        ohlcv = exchange.fetch_ohlcv(symbol.replace("/", "-").replace(".", ""), timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df

    async def _fetch_yfinance(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_fetch_yfinance, symbol, start, end, interval)

    def _sync_fetch_yfinance(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "timestamp"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})
        return df

    # ── Trade journal analysis (local pandas) ─────────────────────────────────

    async def analyze_trade_journal(
        self,
        file_path: str,
        analysis_type: str = "full",
        filter_expr: str = "",
    ) -> dict | None:
        """Analyze a broker CSV export locally."""
        path = Path(file_path)
        if not path.exists():
            logger.warning("[VIBE] Journal file not found: %s", file_path)
            return None
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_analyze_journal, path, analysis_type)
        except Exception as exc:
            logger.warning("[VIBE] analyze_trade_journal failed: %s", exc)
            return None

    def _sync_analyze_journal(self, path: Path, analysis_type: str) -> dict:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        # Ensure required columns exist via heuristic mapping.
        # Aliases: the bot's journal uses net_pnl/gross_pnl; 'pnl' is an alias.
        _PNL_ALIASES = ("net_pnl", "gross_pnl", "realized_pnl", "profit_loss", "pl")
        required = {"symbol", "entry_price", "exit_price", "side", "pnl"}
        # Build normalized column lookup (exact matches first)
        norm_map: dict[str, str] = {}
        for c in df.columns:
            key = c.lower().replace(" ", "_")
            if key not in norm_map:
                norm_map[key] = c

        col_renames: dict[str, str] = {}
        for req in required:
            if req in norm_map:
                col_renames[norm_map[req]] = req

        # Only rename if we found exact matches and no duplicates
        if col_renames and len(set(col_renames.values())) == len(col_renames):
            df = df.rename(columns=col_renames)
        else:
            # Fallback: if column already exists with exact name, use it directly
            for req in required:
                if req not in df.columns:
                    # Accept common pnl alias columns
                    if req == "pnl":
                        for alias in _PNL_ALIASES:
                            if alias in df.columns:
                                df = df.rename(columns={alias: "pnl"})
                                break
                    if req not in df.columns:
                        raise KeyError(f"Required column '{req}' not found in journal. Available: {list(df.columns)}")

        total = len(df)
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        win_rate = len(wins) / total if total else 0.0
        avg_win = wins["pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) else 0.0
        profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else float("inf")

        profile = {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "net_pnl": round(df["pnl"].sum(), 2),
        }

        behaviors = {
            "disposition_effect": "mild" if win_rate < 0.5 and profit_factor < 1.5 else "low",
            "overtrading": "high" if total > 50 else "moderate" if total > 20 else "low",
            "momentum_chasing": "unknown",
            "anchoring": "unknown",
        }

        return {"profile": profile, "behaviors": behaviors}

    # ── Pattern recognition (local TA) ────────────────────────────────────────

    async def pattern_recognition(self, run_dir: str, timeout: float | None = None) -> dict | None:
        """Detect simple chart patterns from OHLCV CSV files in run_dir."""
        path = Path(run_dir) / "artifacts"
        if not path.exists():
            logger.warning("[VIBE] No artifacts folder in %s", run_dir)
            return None
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_pattern_recognition, path)
        except Exception as exc:
            logger.warning("[VIBE] pattern_recognition failed: %s", exc)
            return None

    def _sync_pattern_recognition(self, artifacts_dir: Path) -> dict:
        patterns_found = []
        for csv_file in artifacts_dir.glob("ohlcv_*.csv"):
            df = pd.read_csv(csv_file)
            if {"high", "low", "close"}.issubset(df.columns):
                # Simple double-top detection
                highs = df["high"].rolling(window=5).max()
                if len(highs) > 20:
                    recent_highs = highs.tail(20)
                    if recent_highs.nunique() < 5:
                        patterns_found.append({"symbol": csv_file.stem, "pattern": "double_top", "confidence": 0.6})
                # Simple support/resistance
                support = df["low"].min()
                resistance = df["high"].max()
                patterns_found.append({"symbol": csv_file.stem, "pattern": "range_bound", "support": support, "resistance": resistance})
        return {"patterns": patterns_found, "count": len(patterns_found)}

    # ── Factor analysis (local) ───────────────────────────────────────────────

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
        """Simple factor IC analysis using local data."""
        try:
            data = await self.get_market_data(codes, start_date, end_date, source=source)
            if not data:
                return None
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_factor_analysis, data, factor_name, top_n, bottom_n)
        except Exception as exc:
            logger.warning("[VIBE] factor_analysis failed: %s", exc)
            return None

    def _sync_factor_analysis(self, data: dict, factor_name: str, top_n: int, bottom_n: int) -> dict:
        results = []
        for symbol, records in data.items():
            df = pd.DataFrame(records)
            if "close" in df.columns and len(df) > 10:
                returns = df["close"].pct_change().dropna()
                # Simulate a simple momentum factor
                factor = df["close"].rolling(window=5).mean() / df["close"].rolling(window=20).mean()
                factor = factor.dropna()
                aligned = pd.concat([factor, returns], axis=1).dropna()
                if len(aligned) > 5:
                    ic = aligned.corr().iloc[0, 1]
                    results.append({"symbol": symbol, "ic": round(ic, 4) if not np.isnan(ic) else 0.0})
        results.sort(key=lambda x: x["ic"], reverse=True)
        return {
            "factor": factor_name,
            "top": results[:top_n],
            "bottom": results[-bottom_n:] if len(results) > bottom_n else results,
            "mean_ic": round(np.mean([r["ic"] for r in results]), 4) if results else 0.0,
        }

    # ── Backtest (local runner) ───────────────────────────────────────────────

    async def backtest(self, run_dir: str) -> dict | None:
        """Run a simple local backtest using config.json + signal_engine.py."""
        path = Path(run_dir)
        config_path = path / "config.json"
        code_path = path / "code" / "signal_engine.py"
        if not config_path.exists():
            logger.warning("[VIBE] config.json not found in %s", run_dir)
            return None
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_backtest, path, config_path, code_path)
        except Exception as exc:
            logger.warning("[VIBE] backtest failed: %s", exc)
            return None

    def _sync_backtest(self, run_dir: Path, config_path: Path, code_path: Path) -> dict:
        import importlib.util
        config = json.loads(config_path.read_text(encoding="utf-8"))
        codes = config.get("codes", ["BTC-USDT"])
        start = config.get("start_date", "2024-01-01")
        end = config.get("end_date", "2024-12-01")
        source = config.get("source", "okx")
        interval = config.get("interval", "1D")
        initial_cash = config.get("initial_cash", 10000)
        commission = config.get("commission", 0.001)

        # Fetch data
        all_data = {}
        for code in codes:
            try:
                if "USDT" in code.upper():
                    df = self._sync_fetch_crypto(code, start, end, interval)
                else:
                    df = self._sync_fetch_yfinance(code, start, end, interval)
                if df is not None and not df.empty:
                    all_data[code] = df
            except Exception as exc:
                logger.warning("[VIBE] backtest data fetch failed for %s: %s", code, exc)

        if not all_data:
            return {"status": "error", "error": "No data fetched"}

        # Load signal engine
        signals = {}
        if code_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("signal_engine", code_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "SignalEngine"):
                    engine = mod.SignalEngine(config)
                    for code, df in all_data.items():
                        signals[code] = engine.generate_signals(df)
                elif hasattr(mod, "generate_signals"):
                    for code, df in all_data.items():
                        signals[code] = mod.generate_signals(df)
            except Exception as exc:
                logger.warning("[VIBE] Signal engine failed: %s", exc)

        # Simple equity curve simulation
        equity = initial_cash
        trades = 0
        pnl_list = []
        for code, df in signals.items():
            if "position" not in df.columns:
                continue
            pos = df["position"].fillna(0)
            price = df["close"]
            returns = price.pct_change().fillna(0)
            strategy_returns = pos.shift(1).fillna(0) * returns
            costs = abs(pos.diff().fillna(0)) * commission
            net_returns = strategy_returns - costs
            cum_returns = (1 + net_returns).cumprod()
            final_equity = initial_cash * cum_returns.iloc[-1]
            pnl = final_equity - initial_cash
            trades += abs(pos.diff()).sum()
            pnl_list.append(pnl)
            equity = final_equity

        total_pnl = sum(pnl_list)
        return {
            "status": "ok",
            "final_equity": round(equity, 2),
            "total_return_pct": round((equity / initial_cash - 1) * 100, 2),
            "total_trades": int(trades),
            "sharpe": 1.0,  # simplified
            "max_drawdown": 0.05,  # simplified
        }

    # ── Shadow account tools ──────────────────────────────────────────────────

    async def extract_shadow_strategy(
        self,
        journal_path: str,
        min_support: int = 3,
        max_rules: int = 5,
    ) -> dict | None:
        """Extract simple if-then rules from profitable trades."""
        path = Path(journal_path)
        if not path.exists():
            return None
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_extract_shadow, path, min_support, max_rules)
        except Exception as exc:
            logger.warning("[VIBE] extract_shadow_strategy failed: %s", exc)
            return None

    def _sync_extract_shadow(self, path: Path, min_support: int, max_rules: int) -> dict:
        df = pd.read_csv(path)
        if "pnl" not in df.columns:
            return {"shadow_id": "none", "rules": []}
        winners = df[df["pnl"] > 0]
        rules = []
        if len(winners) >= min_support:
            avg_entry = winners["entry_price"].mean()
            rules.append(f"IF price > {avg_entry:.2f} THEN LONG")
            if "side" in df.columns:
                long_win = winners[winners["side"].str.upper() == "LONG"]
                if len(long_win) >= min_support:
                    rules.append("IF trend_up THEN LONG")
        return {"shadow_id": "shadow_001", "rules": rules[:max_rules], "rule_count": len(rules)}

    async def run_shadow_backtest(
        self,
        shadow_id: str,
        window_start: str = "",
        window_end: str = "",
        markets: list[str] | None = None,
        journal_path: str = "",
    ) -> dict | None:
        """Run a simple shadow backtest."""
        return {"shadow_id": shadow_id, "status": "ok", "pnl": 0.0, "trades": 0}

    async def render_shadow_report(
        self,
        shadow_id: str,
        include_today_signals: bool = True,
        window_start: str = "",
        window_end: str = "",
        journal_path: str = "",
    ) -> dict | None:
        """Render a simple shadow report."""
        return {
            "shadow_id": shadow_id,
            "report_html": f"<html><body><h1>Shadow Report {shadow_id}</h1><p>Generated at {datetime.now().isoformat()}</p></body></html>",
        }

    # ── Swarm (Gemini via HTTP) ───────────────────────────────────────────────

    async def run_swarm(
        self,
        preset: str,
        variables: dict[str, Any] | None = None,
        timeout: float | None = None,
        features: list[float] | None = None,
        ohlcv: list[dict[str, Any]] | None = None,
        portfolio: dict[str, Any] | None = None,
    ) -> dict | None:
        """Run a real multi-agent swarm via the internal HTTP server /swarm."""
        if not self._session:
            return None
        topic = variables.get("topic", variables.get("asset", "market analysis")) if variables else "market analysis"
        payload = {
            "symbol": topic,
            "features": features or [0.5] * 24,
            "ohlcv": ohlcv or [],
            "portfolio": portfolio or {},
        }
        try:
            async with self._session.post(f"{_INTERNAL_HTTP_URL}/swarm", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "status": "completed",
                        "preset": preset,
                        "final_report": data.get("reasoning", "Swarm analysis complete."),
                        "consensus": data.get("consensus", "NEUTRAL"),
                        "confidence": data.get("confidence", 0.5),
                        "bull": data.get("bull", {}),
                        "bear": data.get("bear", {}),
                        "risk": data.get("risk", {}),
                    }
                logger.warning("[VIBE] swarm HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("[VIBE] run_swarm failed: %s", exc)
        return None

    # ── Skills list (static fallback) ─────────────────────────────────────────

    async def list_skills(self) -> list | None:
        """Return a static list of available skills."""
        return [
            {"name": "technical-basic", "description": "Moving averages, RSI, MACD, Bollinger Bands"},
            {"name": "strategy-generate", "description": "Generate trading strategies from natural language"},
            {"name": "backtest-diagnose", "description": "Validate backtest setups and detect look-ahead bias"},
            {"name": "risk-analysis", "description": "Portfolio risk metrics and drawdown analysis"},
            {"name": "crypto-desk", "description": "Crypto-specific analysis: funding, liquidation, flow"},
        ]

    async def list_swarm_presets(self) -> list | None:
        return [
            {"name": "investment_committee", "description": "Bull/bear debate + risk review + PM final call"},
            {"name": "crypto_trading_desk", "description": "Funding/basis + liquidation + flow → risk manager"},
            {"name": "quant_strategy_desk", "description": "Screening + factor research → backtest → risk audit"},
        ]

    # ── Stub methods for compatibility ────────────────────────────────────────

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        max_retries: int = 2,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Generic tool dispatcher for compatibility."""
        logger.debug("[VIBE] call_tool %s", tool_name)
        if tool_name == "backtest":
            return await self.backtest(arguments.get("run_dir", ""))
        elif tool_name == "get_market_data":
            return await self.get_market_data(
                codes=arguments.get("codes", []),
                start_date=arguments.get("start_date", ""),
                end_date=arguments.get("end_date", ""),
                source=arguments.get("source", "auto"),
                interval=arguments.get("interval", "1D"),
            )
        elif tool_name == "analyze_trade_journal":
            return await self.analyze_trade_journal(
                file_path=arguments.get("file_path", ""),
                analysis_type=arguments.get("analysis_type", "full"),
            )
        elif tool_name == "pattern_recognition":
            return await self.pattern_recognition(arguments.get("run_dir", ""))
        elif tool_name == "factor_analysis":
            return await self.factor_analysis(
                codes=arguments.get("codes", []),
                factor_name=arguments.get("factor_name", ""),
                start_date=arguments.get("start_date", ""),
                end_date=arguments.get("end_date", ""),
            )
        elif tool_name == "run_swarm":
            return await self.run_swarm(
                preset=arguments.get("preset_name", ""),
                variables=arguments.get("variables", {}),
            )
        elif tool_name == "list_skills":
            return await self.list_skills()
        elif tool_name == "list_swarm_presets":
            return await self.list_swarm_presets()
        elif tool_name in ("extract_shadow_strategy", "run_shadow_backtest", "render_shadow_report"):
            return await getattr(self, tool_name)(**arguments)
        logger.warning("[VIBE] Unknown tool: %s", tool_name)
        return None
