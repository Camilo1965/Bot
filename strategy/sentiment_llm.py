"""
strategy.sentiment_llm
~~~~~~~~~~~~~~~~~~~~~~~~

Scores crypto-news headlines using Google's Gemini LLM (``gemini-2.5-flash``).

The model acts as an expert crypto quantitative analyst and returns a single
float in **[-1.0, +1.0]**:

* **+1.0** → Extreme Greed / Bullish
*   **0.0** → Neutral
* **-1.0** → Extreme Fear / Bearish

The ``GEMINI_API_KEY`` environment variable (populated via ``.env``) is
required.  If the API call fails or is rate-limited the function falls back
to ``_FALLBACK_SENTIMENT`` (-0.1, i.e. exactly at ``BUY_SENTIMENT_THRESHOLD``)
so the trading loop is never interrupted and no overly-optimistic sentiment
value biases signal generation during an outage.

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

# Explicit import for rate-limit (HTTP 429) detection.  google-api-core is a
# transitive dependency of google-genai so this import should always succeed.
# A private sentinel class is used as a fallback so ``isinstance`` comparisons
# remain valid even if the import somehow fails (avoids ``isinstance(x, None)``
# which raises TypeError).
class _NeverRaised(Exception):
    """Private sentinel; never actually raised.  Stands in for unavailable types."""


try:
    from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
except ImportError:  # pragma: no cover – guard against unusual install layouts
    _ResourceExhausted = _NeverRaised  # type: ignore[assignment,misc]

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.5-flash"

# Safe fallback returned whenever the Gemini API is unavailable.  Using -0.1
# (exactly at BUY_SENTIMENT_THRESHOLD) prevents an artificially optimistic
# sentiment value from triggering BUY signals while the LLM is down, without
# completely suppressing signals that are already supported by the ML model.
_FALLBACK_SENTIMENT: float = -0.1

_SYSTEM_INSTRUCTION = (
    "You are an elite cryptocurrency quantitative analyst for a high-frequency futures trading desk.\n"
    "You will receive one or more news headlines and must score the immediate (intra-day) price impact on majors (BTC, ETH, SOL, BNB, ETC).\n"
    "\n"
    "Output:\n"
    "- Return ONLY one float in [-1.0, 1.0].\n"
    "- -1.0 = extreme bearish/panic, 0.0 = neutral/no impact, 1.0 = extreme bullish/euphoria.\n"
    "- No text, no markdown, no units, no explanation — ONLY the bare float.\n"
    "\n"
    "Scoring scale (objective, symmetric):\n"
    "- Use the FULL range symmetrically: positive events score positively, negative events score negatively.\n"
    "- +0.7 to +1.0: major bullish catalysts (ETF approval, institutional mass adoption, rate cut, major upgrade).\n"
    "- +0.3 to +0.6: moderate bullish (positive regulation news, L2 launch, big partnership, exchange listing).\n"
    "- -0.1 to +0.2: mildly bullish or neutral market noise.\n"
    "- -0.2 to -0.4: moderate bearish (uncertain regulation, minor exploit, macro headwind).\n"
    "- -0.5 to -1.0: major bearish catalysts (exchange hack/collapse, ETF denial, major law enforcement action, chain halt).\n"
    "\n"
    "Event weighting (by IMPACT MAGNITUDE – both positive and negative apply equally):\n"
    "1) Regulatory/institutional — both bullish (ETF approval, legal clarity, institutional adoption) and\n"
    "   bearish (enforcement action, exchange shutdown, lawsuit) events carry high weight.\n"
    "2) Security events — exchange hacks/collapses are bearish; successful audits/upgrades are bullish.\n"
    "3) Macroeconomic — risk-on events (rate cuts, liquidity injections) are bullish;\n"
    "   risk-off (rate hikes, recession signals) are bearish.\n"
    "4) Infrastructure/corporate — major protocol upgrades, big integrations, or large VC funding are bullish;\n"
    "   forks/splits, major bugs, or project failures are bearish.\n"
    "5) Ignore/downweight: influencer opinions, generic clickbait, vague rumors, non-crypto topics.\n"
    "\n"
    "Conflict resolution:\n"
    "- When headlines conflict, weight by IMPACT MAGNITUDE, not by category.\n"
    "- A confirmed ETF approval outweighs a minor exploit; a major exchange collapse outweighs a routine upgrade.\n"
    "- Do NOT systematically bias toward negative events — positive and negative catalysts of equal magnitude deserve equal weight.\n"
    "\n"
    "Off-topic/insufficient signal:\n"
    "- If headlines provide no actionable price signal, return 0.0.\n"
    "\n"
    "Consistency guard:\n"
    "- Never return text, NaN, or values outside [-1.0, 1.0].\n"
    "- Return ONLY the bare float value, nothing else."
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

    Calls the Gemini API synchronously.  Falls back to ``_FALLBACK_SENTIMENT``
    (-0.1) on any error so that the trading loop is never interrupted and no
    falsely-optimistic sentiment can trigger BUY signals during an outage.

    Parameters
    ----------
    headlines:
        A list of news headline strings to analyse.

    Returns
    -------
    float
        Sentiment score in the range [-1.0, +1.0], or ``_FALLBACK_SENTIMENT``
        (-0.1) on failure.
    """
    if not headlines:
        return _FALLBACK_SENTIMENT

    if len(headlines) < 5:
        logger.warning(
            "Low headline count (%d) – sentiment signal may be unreliable; "
            "consider checking news feed availability.",
            len(headlines),
        )

    prompt = "\n".join(f"- {h}" for h in headlines)
    logger.debug(
        "[LLM] Sending %d headline(s) to Gemini:\n%s",
        len(headlines),
        prompt,
    )
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
        logger.debug("[LLM] Gemini raw response: %r", raw[:200])
        try:
            score = float(raw)
        except ValueError:
            # Gemini occasionally returns a brief explanation alongside the number
            # (e.g. thinking-model output).  Extract the LAST numeric token so we
            # skip any leading counts/dates in the text (e.g. "Based on 5 headlines…").
            matches = re.findall(r"[-+]?\d*\.?\d+", raw)
            if matches:
                try:
                    score = float(matches[-1])
                    logger.debug(
                        "[LLM] Extracted float %s from verbose response via regex.", matches[-1]
                    )
                except ValueError:
                    logger.warning(
                        "Gemini returned non-numeric response: %r -- falling back to %.1f",
                        raw,
                        _FALLBACK_SENTIMENT,
                    )
                    return _FALLBACK_SENTIMENT
            else:
                logger.warning(
                    "Gemini returned non-numeric response: %r -- falling back to %.1f",
                    raw,
                    _FALLBACK_SENTIMENT,
                )
                return _FALLBACK_SENTIMENT
        score = max(-1.0, min(1.0, score))
        logger.info(
            "Gemini sentiment score=%.4f (headlines=%d)", score, len(headlines)
        )
        return score
    except Exception as exc:  # noqa: BLE001
        # Distinguish rate-limit (HTTP 429) from other API failures for clarity.
        if isinstance(exc, _ResourceExhausted):
            logger.warning(
                "Gemini API rate-limited (HTTP 429) – falling back to %.1f.  "
                "The bot will retry on the next sentiment refresh cycle.",
                _FALLBACK_SENTIMENT,
            )
        else:
            logger.warning(
                "Gemini API error (%s): %s – falling back to %.1f",
                type(exc).__name__,
                exc,
                _FALLBACK_SENTIMENT,
            )
        return _FALLBACK_SENTIMENT
