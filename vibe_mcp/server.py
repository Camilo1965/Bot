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


# ── FastMCP Instance ──────────────────────────────────────────────────────────
# We instantiate FastMCP without heavy tasks (no API clients initialized here).
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


class PredictionResponse(BaseModel):
    decision: str = "NEUTRAL"
    confidence: float = 0.5
    reasoning: str = ""


# ── Data sampler ──────────────────────────────────────────────────────────────
def _sample_ohlcv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(rows)
    if n <= MAX_OHLCV_ROWS:
        return rows
    sampled = rows[::2]
    if len(sampled) > MAX_OHLCV_ROWS:
        return sampled[-MAX_OHLCV_ROWS:]
    return sampled


def _build_prompt(features: list[float], ohlcv: list[dict[str, Any]], symbol: str) -> str:
    features_str = json.dumps(features)
    ohlcv_sample = _sample_ohlcv(ohlcv)
    preview = json.dumps(ohlcv_sample[:3]) + f" ... ({len(ohlcv_sample)} rows total)"
    prompt = (
        f"You are a quantitative crypto trading assistant. "
        f"Analyze the following market data for {symbol}.\n\n"
        f"Feature vector ({len(features)} values): {features_str}\n\n"
        f"Recent OHLCV (sampled to {len(ohlcv_sample)} rows): {preview}\n\n"
        "Respond with a single JSON object containing exactly these keys:\n"
        '  "decision": one of ["BUY", "SELL", "NEUTRAL"],\n'
        '  "confidence": a float between 0.0 and 1.0,\n'
        '  "reasoning": a short explanation (max 120 chars).\n'
        "Do NOT output markdown fences, ONLY the raw JSON object."
    )
    return prompt


# ── Gemini caller ─────────────────────────────────────────────────────────────
async def _call_gemini(prompt: str) -> dict[str, Any]:
    if not _GENAI_AVAILABLE:
        logger.warning("[SERVER] google-genai not available.")
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": "google-genai missing"}

    if not GEMINI_API_KEY:
        logger.warning("[SERVER] No GEMINI_API_KEY configured.")
        return {"decision": "NEUTRAL", "confidence": 0.5, "reasoning": "Missing GEMINI_API_KEY"}

    print("Enviando a Gemini...", flush=True)
    logger.info("[SERVER] Enviando a Gemini...")

    # Instantiate the client inside the call (no heavy startup)
    client = Client(api_key=GEMINI_API_KEY)

    def _sync_call() -> str:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
        return response.text or ""

    # Force max 30 seconds timeout to prevent hanging the bot
    timeout_s = min(GEMINI_TIMEOUT_S, 30)

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

    print("Respuesta de Gemini recibida", flush=True)
    logger.info("[SERVER] Respuesta de Gemini recibida")

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


# ── FastMCP Tool ──────────────────────────────────────────────────────────────
if mcp is not None:
    @mcp.tool()
    async def predict(features: list[float], ohlcv: list[dict[str, Any]], symbol: str = "ETH/USDT") -> dict[str, Any]:
        """Analyze market data and return a trading decision."""
        prompt = _build_prompt(features, ohlcv, symbol)
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

    prompt = _build_prompt(data.features, data.ohlcv, data.symbol)
    result = await _call_gemini(prompt)
    
    elapsed = time.time() - start_time
    logger.info("[SERVER] Request processed in %.2fs", elapsed)

    resp = PredictionResponse(
        decision=result.get("decision", "NEUTRAL"),
        confidence=result.get("confidence", 0.5),
        reasoning=result.get("reasoning", ""),
    )
    return web.json_response(resp.model_dump(), status=200)


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/predict", handle_predict)
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
