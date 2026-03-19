"""
data_ingestion.news_scraper
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches the latest crypto-news headlines from multiple RSS feeds using the
``feedparser`` library and returns up to ``MAX_HEADLINES_PER_FEED`` items
from each source (CoinTelegraph and CoinDesk).

Usage
-----
    from data_ingestion.news_scraper import fetch_crypto_headlines

    headlines = fetch_crypto_headlines()
"""

from __future__ import annotations

import logging
from typing import Final

import feedparser

logger = logging.getLogger(__name__)

FEED_URLS: Final[list[str]] = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]

#: Maximum number of headlines to collect per feed.
MAX_HEADLINES_PER_FEED: Final[int] = 10


def fetch_crypto_headlines() -> list[str]:
    """Return up to ``MAX_HEADLINES_PER_FEED`` headlines from each RSS feed.

    Combines results from all :data:`FEED_URLS`.  Returns an empty list when
    all feeds fail or yield no items.
    """
    headlines: list[str] = []
    for url in FEED_URLS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_HEADLINES_PER_FEED]:
                title: str = entry.get("title", "").strip()
                if title:
                    headlines.append(title)
            logger.debug(
                "Fetched %d headline(s) from %s",
                min(len(feed.entries), MAX_HEADLINES_PER_FEED),
                url,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("News scraper error for %s: %s", url, exc)
    return headlines
