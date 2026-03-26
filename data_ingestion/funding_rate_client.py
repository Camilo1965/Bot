"""
data_ingestion.funding_rate_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetches the current perpetual-futures funding rate from the Binance USDT
Futures public REST API every 8 hours and writes the result into the shared
application state so that ``strategy.ml_predictor`` can apply a
funding-rate bias when generating trading signals.

Binance endpoint (no API key required):
    GET https://fapi.binance.com/fapi/v1/premiumIndex?symbol=<SYMBOL>

The ``lastFundingRate`` field is returned as a decimal string
(e.g. ``"0.00010000"`` ≡ 0.01 % per 8-hour window).

Threshold semantics used by the predictor
------------------------------------------
* lastFundingRate > +0.0003  (> +0.03 %) → extreme greed  → Short-Bias penalty on BUY
* lastFundingRate < -0.0003  (< -0.03 %) → extreme fear   → Long-Bias bonus on BUY
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
_FETCH_INTERVAL_SECONDS = 8 * 60 * 60  # 8 hours


def _ccxt_symbol_to_binance(symbol: str) -> str:
    """Convert a ccxt-style symbol (e.g. ``BTC/USDT``) to the Binance format (``BTCUSDT``)."""
    return symbol.replace("/", "")


class FundingRateClient:
    """Polls Binance Futures funding rates every 8 hours.

    Parameters
    ----------
    symbols:
        List of ccxt-style trading pairs (e.g. ``["BTC/USDT", "ETH/USDT"]``).
    state:
        Shared application-state dictionary.  The key ``"funding_rates"`` must
        exist and map each *symbol* to a ``float`` (initialised to ``0.0``).
    interval_seconds:
        Polling interval in seconds (default: 8 hours).
    """

    def __init__(
        self,
        symbols: list[str],
        state: dict[str, Any],
        interval_seconds: int = _FETCH_INTERVAL_SECONDS,
    ) -> None:
        self.symbols = symbols
        self.state = state
        self.interval_seconds = interval_seconds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Fetch funding rates immediately, then repeat every *interval_seconds*."""
        while True:
            await self._fetch_all_rates()
            await asyncio.sleep(self.interval_seconds)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_rate(self, session: aiohttp.ClientSession, symbol: str) -> float:
        """Return the latest 8-hour funding rate for *symbol* as a decimal.

        Returns ``0.0`` on any network or parse error so callers always receive
        a safe float value.
        """
        binance_sym = _ccxt_symbol_to_binance(symbol)
        try:
            async with session.get(
                _BINANCE_FUTURES_URL,
                params={"symbol": binance_sym},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    if resp.status != 400:
                        logger.warning(
                            "FundingRateClient: HTTP %d for %s", resp.status, symbol
                        )
                    return 0.0
                data = await resp.json()
                rate = float(data.get("lastFundingRate", 0.0))
                logger.info(
                    "Funding rate fetched  symbol=%s  rate=%.6f (%.4f%%)",
                    symbol,
                    rate,
                    rate * 100,
                )
                return rate
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("FundingRateClient: failed to fetch %s: %s", symbol, exc)
            return 0.0

    async def _fetch_all_rates(self) -> None:
        """Fetch funding rates for all symbols and update shared state."""
        async with aiohttp.ClientSession() as session:
            for symbol in self.symbols:
                rate = await self._fetch_rate(session, symbol)
                funding_rates: dict[str, float] = self.state.get("funding_rates", {})
                funding_rates[symbol] = rate
