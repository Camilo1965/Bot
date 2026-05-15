"""
vibe_mcp/server.py
~~~~~~~~~~~~~~~~~~

Dual HTTP and FastMCP server for VIBE trading decisions.
Receives market data (features + OHLCV), calls Gemini for a trading decision,
and returns a structured JSON response.

Heavy initializations are deferred to request time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field

try:
    from mcp.server.fastmcp import FastMCP
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FASTMCP_AVAILABLE = False

try:
    from google.genai import Client
    from google.genai.types import GenerateContentConfig
    _GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    _GENAI_AVAILABLE = False

logger = logging.getLogger("vibe_mcp.server")

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
MAX_OHLCV_ROWS = int(os.environ.get("VIBE_MCP_MAX_OHLCV_ROWS", "400"))
GEMINI_TIMEOUT_S = int(os.environ.get("VIBE_MCP_GEMINI_TIMEOUT_S", "120"))
SERVER_HOST = os.environ.get("VIBE_MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("VIBE_MCP_PORT", "5000"))

# Feature names for enriched prompts (v2 quant features)
_FEATURE_NAMES = [
    "rsi", "macd_line", "macd_signal", "macd_hist", "atr",
    "bb_pct_b", "bb_width", "vol_delta_norm", "log_ret_1", "log_ret_5",
    "log_ret_10", "log_ret_20", "close_vs_sma200_1h", "vol_rel",
    "adx", "stoch_rsi_k", "stoch_rsi_d", "williams_r", "obv", "cmf",
    "ret_skew_20", "ret_kurt_20", "roll_sharpe_20", "hour_sin", "hour_cos", "dow",
    "vibe_pattern_score", "vibe_factor_ic", "vibe_journal_health", "vibe_backtest_sharpe",
]

# ── FastMCP Instance ──────────────────────────────────────────────────────────
if _FASTMCP_AVAILABLE:
    mcp = FastMCP("vibe-trading")
else:
    mcp = None


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class MarketDataInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    features: list[float] = Field(default_factory=list)
    ohlcv: list[dict[str, Any]] = Field(default_factory=list)
    symbol: str = "ETH/USDT"
    portfolio: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    decision: str = "NEUTRAL"
    confidence: float = 0.5
    reasoning: str = ""


class SwarmInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str = "ETH/USDT"
    features: list[float] = Field(default_factory=list)
    ohlcv: list[dict[str, Any]] = Field(default_factory=list)
    portfolio: dict[str, Any] = Field(default_factory=dict)


# ── Data sampler ──────────────────────────────────────────────────────────────
def _sample_ohlcv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(rows)
    if n <= MAX_OHLCV_ROWS:
        return rows
    sampled = rows[::2]
    if len(sampled) > MAX_OHLCV_ROWS:
        return sampled[-MAX_OHLCV_ROWS:]
    return sampled


def _build_feature_dict(features: list[float]) -> dict[str, float]:
    """Map raw feature vector to named features for the prompt."""
    return {
        name: float(features[i]) if i < len(features) else 0.0
        for i, name in enumerate(_FEATURE_NAMES)
    }


def _build_enriched_prompt(
    features: list[float],
    ohlcv: list[dict[str, Any]],
    symbol: str,
    portfolio: dict[str, Any],
) -> str:
    """Build a rich prompt with named features, recent candles, and portfolio context."""
    feature_dict = _build_feature_dict(features)
    feature_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in feature_dict.items())

    ohlcv_sample = _sample_ohlcv(ohlcv)
    # Show last 20 candles (or fewer if not enough)
    recent = ohlcv_sample[-20:] if len(ohlcv_sample) >= 20 else ohlcv_sample
    candle_lines = "\n".join(
        f"  {i+1}. O:{c.get('open',''):.4f} H:{c.get('high',''):.4f} L:{c.get('low',''):.4f} C:{c.get('close',''):.4f} V:{c.get('volume',''):.2f}"
        for i, c in enumerate(recent)
    )

    # Portfolio context
    pos = portfolio.get("open_positions", [])
    pnl = portfolio.get("total_pnl", 0.0)
    balance = portfolio.get("balance", 0.0)
    port_str = (
        f"Open positions: {len(pos)}\n"
        f"Total PnL: {pnl:.2f} USDT\n"
        f"Balance: {balance:.2f} USDT\n"
        f"Symbols held: {[p.get('symbol') for p in pos]}"
        if portfolio
        else "No portfolio context provided."
    )

    prompt = (
        f"You are an elite quantitative crypto trading analyst with 20 years of experience.\n\n"
        f"Analyze {symbol} and provide a trading decision.\n\n"
        f"## Technical Features (latest)\n{feature_lines}\n\n"
        f"## Recent Price Action (last {len(recent)} candles)\n{candle_lines}\n\n"
        f"## Portfolio Context\n{port_str}\n\n"
        f"Instructions:\n"
        f"- Evaluate trend strength, momentum, volatility, and volume.\n"
        f"- Consider the portfolio context (don't over-concentrate).\n"
        f"- Respond with a single JSON object containing exactly these keys:\n"
        f'  "decision": one of ["BUY", "SELL", "NEUTRAL"],\n'
        f'  "confidence": a float between 0.0 and 1.0,\n'
        f'  "reasoning": a concise but specific explanation (max 300 chars).\n'
        f"Do NOT output markdown fences, ONLY the raw JSON object."
    )
    return prompt


def _build_swarm_prompt(role: str, features: list[float], ohlcv: list[dict[str, Any]], symbol: str, portfolio: dict[str, Any]) -> str:
    """Build a specialized prompt for a swarm agent role."""
    feature_dict = _build_feature_dict(features)
    feature_lines = "\n".join(f"  {k}: {v:.4f}" for k, v in feature_dict.items())
    recent = ohlcv[-15:] if len(ohlcv) >= 15 else ohlcv
    candle_lines = "\n".join(
        f"  {c.get('close',''):.4f}"
        for c in recent
    )

    roles = {
        "bull": (
            f"You are a BULLISH analyst for {symbol}. Find the strongest bullish arguments.\n"
            f"Features:\n{feature_lines}\n\n"
            f"Recent closes:\n{candle_lines}\n\n"
            f"Respond ONLY with JSON: {{\"score\": 0-10, \"key_points\": [\"...\"], \"confidence\": 0.0-1.0}}"
        ),
        "bear": (
            f"You are a BEARISH analyst for {symbol}. Find the strongest bearish risks.\n"
            f"Features:\n{feature_lines}\n\n"
            f"Recent closes:\n{candle_lines}\n\n"
            f"Respond ONLY with JSON: {{\"score\": 0-10, \"key_points\": [\"...\"], \"confidence\": 0.0-1.0}}"
        ),
        "risk": (
            f"You are a RISK MANAGER for {symbol}. Assess downside, volatility, and position sizing.\n"
            f"Features:\n{feature_lines}\n\n"
            f"Recent closes:\n{candle_lines}\n\n"
            f"Portfolio: {portfolio}\n\n"
            f"Respond ONLY with JSON: {{\"risk_score\": 0-10, \"max_position_pct\": 0.0-1.0, \"warning\": \"...\"}}"
        ),
    }
    return roles.get(role, roles["bull"])


# ── Gemini caller ─────────────────────────────────────────────────────────────
async def _call_gemini(prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> dict[str, Any]:
    if not _GENAI_AVAILABLE:
        logger.warning("[SERVER] google-genai not available.")
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": "google-genai missing"}

    if not GEMINI_API_KEY:
        logger.warning("[SERVER] No GEMINI_API_KEY configured.")
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": "Missing GEMINI_API_KEY"}

    logger.info("[SERVER] Sending to Gemini (temp=%.1f, max_tokens=%d)...", temperature, max_tokens)

    client = Client(api_key=GEMINI_API_KEY)

    def _sync_call() -> str:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    timeout_s = min(GEMINI_TIMEOUT_S, 60)

    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_sync_call),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.error("[SERVER] Timeout from Gemini")
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": "Timeout from Gemini"}
    except Exception as exc:
        logger.error("[SERVER] Error from Gemini: %s", exc)
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": f"Error: {exc}"}

    logger.info("[SERVER] Gemini response received")

    # Strip markdown fences if present
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        raw = parts[-1]
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[SERVER] JSON decode failed: %s | raw=%s", exc, raw[:200])
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": f"JSON parse error: {exc}"}

    decision = str(parsed.get("decision", "NEUTRAL")).upper()
    if decision not in ("BUY", "SELL", "NEUTRAL"):
        decision = "NEUTRAL"

    return {
        "decision": decision,
        "confidence": float(parsed.get("confidence", 0.5)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


async def _call_gemini_swarm(prompt: str) -> dict[str, Any]:
    """Call Gemini for swarm agent (returns raw JSON dict)."""
    result = await _call_gemini(prompt, temperature=0.7, max_tokens=512)
    # Try to parse the reasoning as JSON if it's a swarm response
    raw = result.get("reasoning", "{}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "confidence": result.get("confidence", 0.5)}


# ── FastMCP Tool ──────────────────────────────────────────────────────────────
if mcp is not None:
    @mcp.tool()
    async def predict(features: list[float], ohlcv: list[dict[str, Any]], symbol: str = "ETH/USDT") -> dict[str, Any]:
        """Analyze market data and return a trading decision."""
        prompt = _build_enriched_prompt(features, ohlcv, symbol, {})
        return await _call_gemini(prompt)


# ── HTTP handlers ─────────────────────────────────────────────────────────────
async def handle_predict(request: web.Request) -> web.Response:
    start_time = time.time()

    try:
        body = await request.json()
    except Exception as exc:
        return web.json_response(
            PredictionResponse(decision="NEUTRAL", confidence=0.5, reasoning="Invalid JSON").model_dump(),
            status=400,
        )

    try:
        data = MarketDataInput.model_validate(body)
    except Exception as exc:
        return web.json_response(
            PredictionResponse(decision="NEUTRAL", confidence=0.5, reasoning=f"Validation error: {exc}").model_dump(),
            status=422,
        )

    prompt = _build_enriched_prompt(data.features, data.ohlcv, data.symbol, data.portfolio)
    result = await _call_gemini(prompt)

    elapsed = time.time() - start_time
    logger.info("[SERVER] /predict processed in %.2fs", elapsed)

    resp = PredictionResponse(
        decision=result.get("decision", "NEUTRAL"),
        confidence=result.get("confidence", 0.5),
        reasoning=result.get("reasoning", ""),
    )
    return web.json_response(resp.model_dump(), status=200)


async def handle_swarm(request: web.Request) -> web.Response:
    """Run a real multi-agent swarm: Bull + Bear + Risk Manager → Consensus."""
    start_time = time.time()

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    try:
        data = SwarmInput.model_validate(body)
    except Exception as exc:
        return web.json_response({"error": f"Validation error: {exc}"}, status=422)

    # Launch 3 agents in parallel
    bull_prompt = _build_swarm_prompt("bull", data.features, data.ohlcv, data.symbol, data.portfolio)
    bear_prompt = _build_swarm_prompt("bear", data.features, data.ohlcv, data.symbol, data.portfolio)
    risk_prompt = _build_swarm_prompt("risk", data.features, data.ohlcv, data.symbol, data.portfolio)

    bull_task = asyncio.create_task(_call_gemini_swarm(bull_prompt))
    bear_task = asyncio.create_task(_call_gemini_swarm(bear_prompt))
    risk_task = asyncio.create_task(_call_gemini_swarm(risk_prompt))

    bull_res, bear_res, risk_res = await asyncio.gather(bull_task, bear_task, risk_task, return_exceptions=True)

    # Consensus synthesis
    def _safe_score(res: Any, key: str) -> float:
        if isinstance(res, Exception):
            return 5.0
        return float(res.get(key, 5.0)) if isinstance(res, dict) else 5.0

    bull_score = _safe_score(bull_res, "score")
    bear_score = _safe_score(bear_res, "score")
    risk_score = _safe_score(risk_res, "risk_score")

    # Net sentiment: bullish - bearish, adjusted by risk
    net_sentiment = (bull_score - bear_score) / 10.0  # range [-1, 1]
    risk_adjustment = 1.0 - (risk_score / 20.0)  # 1.0 = low risk, 0.5 = high risk
    adjusted_score = net_sentiment * risk_adjustment

    if adjusted_score >= 0.3:
        consensus = "STRONG_BUY" if adjusted_score >= 0.6 else "BUY"
    elif adjusted_score <= -0.3:
        consensus = "EXIT" if adjusted_score <= -0.6 else "REDUCE"
    else:
        consensus = "NEUTRAL"

    confidence = min(1.0, abs(adjusted_score) + 0.3)

    elapsed = time.time() - start_time
    logger.info("[SERVER] /swarm processed in %.2fs → %s (conf=%.2f)", elapsed, consensus, confidence)

    return web.json_response({
        "consensus": consensus,
        "confidence": round(confidence, 4),
        "bull": bull_res if not isinstance(bull_res, Exception) else {"error": str(bull_res)},
        "bear": bear_res if not isinstance(bear_res, Exception) else {"error": str(bear_res)},
        "risk": risk_res if not isinstance(risk_res, Exception) else {"error": str(risk_res)},
        "reasoning": f"Bull={bull_score:.1f} Bear={bear_score:.1f} Risk={risk_score:.1f} → {consensus}",
    })


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/predict", handle_predict)
    app.router.add_post("/swarm", handle_swarm)
    app.router.add_get("/health", handle_health)
    return app


async def start_mcp_server(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    """Start the VIBE MCP HTTP server as an asyncio task."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        logger.info("🧠 VIBE MCP internal server listening on http://%s:%d", host, port)
        await site.start()
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()


def main() -> None:  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    # If run directly as a script, default to FastMCP if available, else HTTP
    if _FASTMCP_AVAILABLE:
        mcp.run()
    else:
        app = create_app()
        web.run_app(app, host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
