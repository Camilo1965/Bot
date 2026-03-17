"""
strategy.sentiment_analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scores crypto-news headlines using the VADER (Valence Aware Dictionary and
sEntiment Reasoner) algorithm.  The compound score is normalised to the
range **[-1, +1]**:

* **+1** → very bullish
*  **0** → neutral
* **-1** → very bearish

VADER requires no GPU or large model download, making it suitable for a
lightweight, always-on trading bot.

Usage
-----
    from strategy.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    score = analyzer.score_headline("Bitcoin surges to new all-time high!")
    # → 0.72 (example)

    avg = analyzer.score_headlines(["BTC crashes", "ETH pumps"])
    # → average compound score
"""

from __future__ import annotations

import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Wraps VADER to produce a single compound sentiment score per headline."""

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()
        logger.info("SentimentAnalyzer initialised (VADER).")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score_headline(self, headline: str) -> float:
        """Return the VADER compound score for *headline* in [-1, +1]."""
        scores = self._vader.polarity_scores(headline)
        compound: float = scores["compound"]
        logger.debug("headline=%r  compound=%.4f", headline, compound)
        return compound

    def score_headlines(self, headlines: list[str]) -> float:
        """Return the average compound score for a batch of *headlines*.

        Returns ``0.0`` when *headlines* is empty.
        """
        if not headlines:
            return 0.0
        scores = [self.score_headline(h) for h in headlines]
        avg = sum(scores) / len(scores)
        logger.debug(
            "Scored %d headline(s) – avg compound=%.4f", len(headlines), avg
        )
        return avg
