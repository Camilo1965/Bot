"""
strategy.sentiment_llm
~~~~~~~~~~~~~~~~~~~~~~~~

Scores crypto-news headlines using Google's Gemini LLM (``gemini-1.5-flash``).

The model acts as an expert crypto quantitative analyst and returns a single
float in **[-1.0, +1.0]**:

* **+1.0** → Extreme Greed / Bullish
*   **0.0** → Neutral
* **-1.0** → Extreme Fear / Bearish

The ``GEMINI_API_KEY`` environment variable (populated via ``.env``) is
required.  If the API call fails or is rate-limited the function falls back
to ``0.0`` so the trading loop is never interrupted.

Usage
-----
    from strategy.sentiment_llm import get_gemini_sentiment

    score = get_gemini_sentiment(["Bitcoin hits new ATH!", "ETH ETF rejected"])
    # → e.g. 0.45
"""

from __future__ import annotations

import logging
import os
import re

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are an elite cryptocurrency quantitative analyst for a high-frequency futures trading desk. "
    "Evaluate the provided news headlines for immediate, short-term price impact on major assets (BTC, ETH, SOL, BNB). "
    "Respond with ONLY a single float number between -1.0 (Extreme Fear/Bearish) and 1.0 (Extreme Greed/Bullish). "
    "CRITICAL RULES: "
    "1. Give massive weight to institutional news, SEC regulations, hacks, or macroeconomic data. "
    "2. Ignore generic clickbait, influencer opinions, or minor rumors. "
    "3. Focus strictly on the short-term market momentum. "
    "Return ONLY the raw float number. No text, no markdown, no explanation."
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return (and lazily initialise) the shared Gemini client."""
    global _client  # noqa: PLW0603
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialised for model '%s'.", _MODEL_NAME)
    return _client


def get_gemini_sentiment(headlines: list[str]) -> float:
    """Return a sentiment score in [-1.0, 1.0] for the given *headlines*.

    Calls the Gemini API synchronously.  Falls back to ``0.0`` on any error.

    Parameters
    ----------
    headlines:
        A list of news headline strings to analyse.

    Returns
    -------
    float
        Sentiment score in the range [-1.0, +1.0], or ``0.0`` on failure.
    """
    if not headlines:
        return 0.0

    prompt = "\n".join(f"- {h}" for h in headlines)
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
            ),
        )
        raw = response.text.strip()
        # Extract the first float-like token, supporting scientific notation
        # (e.g. 1e-2) and an explicit leading '+' sign.
        match = re.search(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", raw)
        if match is None:
            logger.warning("Gemini returned non-numeric response: %r", raw)
            return 0.0
        score = float(match.group())
        score = max(-1.0, min(1.0, score))
        logger.info(
            "Gemini sentiment score=%.4f (headlines=%d)", score, len(headlines)
        )
        return score
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Gemini API error (%s): %s – falling back to 0.0",
            type(exc).__name__,
            exc,
        )
        return 0.0
