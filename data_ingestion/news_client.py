"""
data_ingestion.news_client
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches the latest crypto-news headlines from a public RSS feed and pushes
them into an asyncio.Queue so that downstream consumers (e.g. the sentiment
analyser) can process them without blocking the event loop.

Usage
-----
    queue: asyncio.Queue[list[str]] = asyncio.Queue()
    client = CryptoNewsClient(queue=queue)
    await client.run()           # runs forever, polling every POLL_INTERVAL s
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Public RSS feed – Cointelegraph (no API key required).
DEFAULT_FEED_URL = "https://cointelegraph.com/rss"

#: How often (seconds) to re-fetch the feed.
POLL_INTERVAL = 300  # 5 minutes

#: HTTP request timeout in seconds.
REQUEST_TIMEOUT = 10


#: Maximum number of headline titles retained in the seen-set to bound memory.
_SEEN_TITLES_MAXLEN = 2_000


class CryptoNewsClient:
    """Polls an RSS feed and pushes batches of headlines onto *queue*."""

    def __init__(
        self,
        queue: asyncio.Queue[list[str]],
        feed_url: str = DEFAULT_FEED_URL,
        poll_interval: int = POLL_INTERVAL,
    ) -> None:
        self.queue = queue
        self.feed_url = feed_url
        self.poll_interval = poll_interval
        # Bounded deque used as an ordered set to cap memory usage.
        self._seen_titles: deque[str] = deque(maxlen=_SEEN_TITLES_MAXLEN)
        self._seen_titles_set: set[str] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll the RSS feed on *poll_interval* and enqueue new headlines."""
        logger.info("CryptoNewsClient starting – feed: %s", self.feed_url)
        while True:
            try:
                headlines = await self._fetch_headlines()
                new_headlines = [h for h in headlines if h not in self._seen_titles_set]
                if new_headlines:
                    for title in new_headlines:
                        if len(self._seen_titles) == _SEEN_TITLES_MAXLEN:
                            # The deque is full; the oldest entry is about to be
                            # evicted – remove it from the fast-lookup set too.
                            evicted = self._seen_titles[0]
                            self._seen_titles_set.discard(evicted)
                        self._seen_titles.append(title)
                        self._seen_titles_set.add(title)
                    await self.queue.put(new_headlines)
                    logger.info(
                        "Fetched %d new headline(s) from %s",
                        len(new_headlines),
                        self.feed_url,
                    )
                else:
                    logger.debug("No new headlines since last poll.")
            except asyncio.CancelledError:
                logger.info("CryptoNewsClient cancelled – shutting down.")
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "News fetch error (%s): %s – retrying in %ds",
                    type(exc).__name__,
                    exc,
                    self.poll_interval,
                )

            await asyncio.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_headlines(self) -> list[str]:
        """Download the RSS feed and return a list of *<title>* strings."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self.feed_url) as response:
                response.raise_for_status()
                body = await response.text()

        return self._parse_rss(body)

    @staticmethod
    def _parse_rss(xml_text: str) -> list[str]:
        """Extract item titles from an RSS 2.0 payload."""
        root = ET.fromstring(xml_text)  # noqa: S314 – local parse, trusted feed
        titles: list[str] = []
        # RSS 2.0: <rss><channel><item><title>…</title></item>…</channel></rss>
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                titles.append(title_el.text.strip())
        return titles


# ---------------------------------------------------------------------------
# Convenience runner (mirrors the pattern in websocket_client.py)
# ---------------------------------------------------------------------------


async def run_news_client(
    queue: asyncio.Queue[list[str]],
    feed_url: str = DEFAULT_FEED_URL,
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """Create and start a :class:`CryptoNewsClient`."""
    client = CryptoNewsClient(
        queue=queue,
        feed_url=feed_url,
        poll_interval=poll_interval,
    )
    await client.run()
