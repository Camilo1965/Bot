"""
vibe_mcp/server.py
~~~~~~~~~~~~~~~~~~

HTTP MCP server for VIBE trading decisions.
Receives market data (features + OHLCV), calls Gemini for a trading decision,
and returns a structured JSON response.

Designed to eliminate Pydantic ValidationErrors and timeouts by:
- Using a flexible Pydantic schema (extra="allow", no strict feature count).
- Sampling OHLCV data to keep prompts lightweight.
- Setting a 120s internal timeout for Gemini.
- Returning a safe NEUTRAL fallback on any error.
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

# google-genai (sync client — we run it in a thread)
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


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class MarketDataInput(BaseModel):
    """Flexible input schema: features can be any length (target 16)."""

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
    """If we receive > MAX_OHLCV_ROWS, take every 2nd row or the last N."""
    n = len(rows)
    if n <= MAX_OHLCV_ROWS:
        return rows
    # Strategy: every 2nd row, then cap to last MAX_OHLCV_ROWS if still too many
    sampled = rows[::2]
    if len(sampled) > MAX_OHLCV_ROWS:
        return sampled[-MAX_OHLCV_ROWS:]
    return sampled


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(data: MarketDataInput) -> str:
    features_str = json.dumps(data.features)
    ohlcv_sample = _sample_ohlcv(data.ohlcv)
    # Show only first/last few rows in the prompt to save tokens
    preview = json.dumps(ohlcv_sample[:3]) + f" ... ({len(ohlcv_sample)} rows total)"
    prompt = (
        f"You are a quantitative crypto trading assistant. "
        f"Analyze the following market data for {data.symbol}.\n\n"
        f"Feature vector ({len(data.features)} values): {features_str}\n\n"
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
        raise RuntimeError("google-genai not installed")

    if not GEMINI_API_KEY:
        logger.warning("[SERVER] No GEMINI_API_KEY configured.")
        raise RuntimeError("Missing GEMINI_API_KEY")

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

    text = await asyncio.wait_for(
        asyncio.to_thread(_sync_call),
        timeout=GEMINI_TIMEOUT_S,
    )

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
        raise

    decision = str(parsed.get("decision", "NEUTRAL")).upper()
    if decision not in ("BUY", "SELL", "NEUTRAL"):
        decision = "NEUTRAL"

    return {
        "decision": decision,
        "confidence": float(parsed.get("confidence", 0.5)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ── HTTP handlers ─────────────────────────────────────────────────────────────
async def handle_predict(request: web.Request) -> web.Response:
    """Main prediction endpoint with global try/except fallback."""
    start_time = time.time()

    try:
        body = await request.json()
    except Exception as exc:
        logger.error("[SERVER] Invalid JSON body: %s", exc)
        return web.json_response(
            PredictionResponse(
                decision="NEUTRAL", confidence=0.5, reasoning="Invalid JSON"
            ).model_dump(),
            status=400,
        )

    try:
        data = MarketDataInput.model_validate(body)
    except Exception as exc:
        logger.error("[SERVER] Validation error: %s", exc)
        return web.json_response(
            PredictionResponse(
                decision="NEUTRAL", confidence=0.5, reasoning=f"Validation error: {exc}"
            ).model_dump(),
            status=422,
        )

    print(f"[SERVER] Recibidas {len(data.features)} columnas y {len(data.ohlcv)} filas. Procesando...")
    logger.info("[SERVER] Recibidas %d columnas y %d filas. Procesando...", len(data.features), len(data.ohlcv))

    try:
        prompt = _build_prompt(data)
        result = await _call_gemini(prompt)
        elapsed = time.time() - start_time
        print(f"[SERVER] Gemini respondió en {elapsed:.2f}s")
        logger.info("[SERVER] Gemini responded in %.2fs", elapsed)

        resp = PredictionResponse(
            decision=result.get("decision", "NEUTRAL"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
        )
        return web.json_response(resp.model_dump(), status=200)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start_time
        logger.error("[SERVER] Error after %.2fs: %s", elapsed, exc)
        print(f"[SERVER] Error after {elapsed:.2f}s: {exc}")
        fallback = PredictionResponse(
            decision="NEUTRAL", confidence=0.5, reasoning=f"Error: {exc}"
        )
        return web.json_response(fallback.model_dump(), status=200)


async def handle_health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/predict", handle_predict)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    app = create_app()
    logger.info("Starting VIBE MCP server on %s:%d", SERVER_HOST, SERVER_PORT)
    web.run_app(app, host=SERVER_HOST, port=SERVER_PORT)


if __name__ == "__main__":  # pragma: no cover
    main()
