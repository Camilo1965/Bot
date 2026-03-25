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

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are an elite cryptocurrency quantitative analyst for a high-frequency futures trading desk.\n"
    "You will receive one or more news headlines and must score the immediate (intra-day) price impact on majors (BTC, ETH, SOL, BNB, ETC).\n"
    "\n"
    "Output:\n"
    "- Return ONLY one float in [-1.0, 1.0].\n"
    "- -1.0 = extreme bearish/panic, 1.0 = extreme bullish/euphoria.\n"
    "- No text, no markdown, no units, no explanation.\n"
    "\n"
    "Input normalization:\n"
    "- Lowercase, strip accents, map leetspeak (4→a, 0→o, 3→e, 1→i, 5→s).\n"
    "\n"
    "Priority weighting (in order):\n"
    "1) Regulatory/institutional: SEC/CFTC/ETF approvals/denials, lawsuits, settlements, enforcement, exchange license actions.\n"
    "2) Security events: hacks, exploits, chain halts, exchange freezes/delistings, bridge/rugpulls.\n"
    "3) Macroeconomic: CPI/FED/ECB rate decisions, liquidity programs, payrolls; risk-on/off transmission to crypto.\n"
    "4) Infra/corporate: major chain upgrades/forks, L2 launches, big integrations/listings, major VC raises for core infra.\n"
    "5) Downweight/ignore: influencer opinions, generic clickbait, vague/\"rumor\"/\"alleged\"/\"reportedly\", non-crypto topics.\n"
    "\n"
    "Asset-specific:\n"
    "- If clearly tied to one asset (e.g., SOL outage), bias the score to that asset's likely move but still emit a single scalar for overall majors' momentum.\n"
    "\n"
    "Conflict resolution:\n"
    "- If headlines conflict, weight higher the regulatory/security items; do not average blindly.\n"
    "\n"
    "Off-topic/too vague:\n"
    "- If insufficient signal, return 0.0.\n"
    "\n"
    "Safety/consistency guard:\n"
    "- Never return text, NaN, or values outside [-1.0, 1.0]."
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
        try:
            score = float(raw)
        except ValueError:
            logger.warning("Gemini returned non-numeric response: %r -- falling back to 0.0", raw)
            return 0.0
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
