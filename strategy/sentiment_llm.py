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

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-1.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are an expert crypto quantitative analyst. "
    "You will be given a list of recent cryptocurrency news headlines. "
    "Analyse them and respond with ONLY a single float number between -1.0 "
    "(Extreme Fear/Bearish) and 1.0 (Extreme Greed/Bullish). "
    "Do not include any conversational text, markdown, or explanation. "
    "Return ONLY the raw number."
)

_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    """Return (and lazily initialise) the shared Gemini model."""
    global _model  # noqa: PLW0603
    if _model is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name=_MODEL_NAME,
            system_instruction=_SYSTEM_INSTRUCTION,
        )
        logger.info("Gemini model '%s' initialised.", _MODEL_NAME)
    return _model


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
        model = _get_model()
        response = model.generate_content(prompt)
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
