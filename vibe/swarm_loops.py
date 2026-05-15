"""
vibe.swarm_loops
~~~~~~~~~~~~~~~~

Background swarm-intelligence loop for VIBE-Trading (Fase 5).

Runs the ``crypto_trading_desk`` swarm preset periodically and stores the
parsed recommendation in ``shared_state["vibe_swarm_recommendation"]``.

v2 improvements:
- Passes real OHLCV data and portfolio context to the swarm.
- Uses semantic consensus from the multi-agent response instead of keyword matching.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("clawdbot.vibe.swarm")

_SWARM_BACKOFF_BASE = 2.0
_SWARM_BACKOFF_MAX = 300.0
_SWARM_TIMEOUT = 90


def _build_ohlcv_for_symbol(state: dict[str, Any], symbol: str) -> list[dict[str, Any]]:
    """Build OHLCV list from shared_state buffers for the swarm prompt."""
    prices = list(state.get("prices", {}).get(symbol, []))
    highs = list(state.get("highs", {}).get(symbol, []))
    lows = list(state.get("lows", {}).get(symbol, []))
    volumes = list(state.get("volumes", {}).get(symbol, []))
    n = min(len(prices), len(highs), len(lows))
    if n == 0:
        return []
    # Use last 30 candles max
    recent = min(n, 30)
    out = []
    for i in range(n - recent, n):
        out.append({
            "open": float(prices[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(prices[i]),
            "volume": float(volumes[i]) if i < len(volumes) else 0.0,
        })
    return out


def _build_portfolio_context(state: dict[str, Any]) -> dict[str, Any]:
    """Extract portfolio snapshot from shared_state."""
    executor = state.get("paper_executor")
    risk = state.get("risk_manager")
    positions = []
    if executor and hasattr(executor, "open_positions"):
        for sym, pos in executor.open_positions.items():
            positions.append({
                "symbol": sym,
                "entry_price": float(pos.entry_price),
                "size": float(pos.position_size),
                "pnl": float(getattr(pos, "unrealized_pnl", 0.0)),
            })
    total_pnl = float(getattr(executor, "total_pnl", 0.0)) if executor else 0.0
    balance = float(risk.balance) if risk and hasattr(risk, "balance") else 0.0
    return {
        "open_positions": positions,
        "position_count": len(positions),
        "total_pnl": round(total_pnl, 2),
        "balance": round(balance, 2),
    }


def _extract_text(data: Any) -> str:
    """Extract plain text from an MCP-style response dict."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data.lower()
    if isinstance(data, dict):
        parts: list[str] = []
        for item in data.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts).lower()
    return str(data).lower()


def parse_swarm_recommendation(result: dict | None) -> str:
    """Map a raw swarm result to one of five recommendation levels.

    Uses the semantic consensus field when available, falling back to
    keyword matching only when necessary.
    """
    if result is None:
        return "NEUTRAL"

    # Prefer direct consensus from multi-agent swarm
    consensus = str(result.get("consensus", "")).upper()
    if consensus and consensus in ("STRONG_BUY", "BUY", "REDUCE", "EXIT"):
        return consensus

    # Fallback: keyword matching on the final report text or legacy content
    text = str(result.get("final_report", "")).lower()
    if not text:
        text = _extract_text(result)
    _STRONG_BUY_KW = {"strong buy", "aggressive long", "conviction buy", "accumulate"}
    _BUY_KW = {"buy", "long", "bullish", "upside", "enter long"}
    _EXIT_KW = {"exit", "close", "sell", "flat", "neutralize", "take profit"}
    _REDUCE_KW = {"reduce", "trim", "scale out", "lighten", "de-risk"}

    strong_buy = sum(1 for kw in _STRONG_BUY_KW if kw in text)
    buy = sum(1 for kw in _BUY_KW if kw in text)
    exit_s = sum(1 for kw in _EXIT_KW if kw in text)
    reduce = sum(1 for kw in _REDUCE_KW if kw in text)

    scores = {
        "STRONG_BUY": strong_buy,
        "BUY": buy,
        "EXIT": exit_s,
        "REDUCE": reduce,
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "NEUTRAL"
    return best


async def crypto_desk_swarm_loop(
    client: Any,
    shared_state: dict[str, Any],
    watchlist: list[str],
    interval_s: float = 3600,
    initial_delay_s: float = 120.0,
) -> None:
    """Hourly: run ``crypto_trading_desk`` swarm for each symbol with real data."""
    if not client.available:
        logger.info("[VIBE SWARM] Client not available - swarm loop disabled.")
        return

    if initial_delay_s > 0:
        await asyncio.sleep(initial_delay_s)
    failures = 0

    while True:
        for symbol in watchlist:
            try:
                ohlcv = _build_ohlcv_for_symbol(shared_state, symbol)
                portfolio = _build_portfolio_context(shared_state)
                # Use ML probability as a feature hint if available
                ml_prob = shared_state.get("ml_probs", {}).get(symbol, 0.5)
                features = [ml_prob] + [0.0] * 23  # minimal feature vector

                result = await client.run_swarm(
                    preset="crypto_trading_desk",
                    variables={"asset": symbol, "timeframe": "1h"},
                    timeout=_SWARM_TIMEOUT,
                    features=features,
                    ohlcv=ohlcv,
                    portfolio=portfolio,
                )
                if result is not None:
                    rec = parse_swarm_recommendation(result)
                    shared_state.setdefault("vibe_swarm_recommendation", {})[symbol] = rec
                    logger.info(
                        "[VIBE SWARM] %s recommendation: %s (confidence=%.2f)",
                        symbol,
                        rec,
                        result.get("confidence", 0.5),
                    )
                    failures = 0
                else:
                    failures += 1
                    logger.warning("[VIBE SWARM] %s returned no result.", symbol)
            except asyncio.TimeoutError:
                failures += 1
                logger.warning("[VIBE SWARM] %s timed out after %ds.", symbol, _SWARM_TIMEOUT)
            except Exception as exc:
                failures += 1
                logger.warning("[VIBE SWARM] %s failed: %s", symbol, exc)

            if failures > 0:
                backoff = min(_SWARM_BACKOFF_BASE * (2 ** (failures - 1)), _SWARM_BACKOFF_MAX)
                logger.info("[VIBE SWARM] Backing off %.0fs after %d failures.", backoff, failures)
                await asyncio.sleep(backoff)

        await asyncio.sleep(interval_s)
