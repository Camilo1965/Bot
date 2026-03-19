"""
execution.binance_executor
~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory helpers for creating and configuring a ccxt async Binance Futures
exchange client.

When ``USE_BINANCE_TESTNET=True`` is set in the environment the client
returned by :func:`create_exchange` is pointed at the Binance Futures Testnet
(``https://testnet.binancefuture.com``) instead of the production endpoints.
The same client is passed to :class:`~execution.paper_executor.PaperExecutor`
so that real orders are routed to the testnet while the paper-simulation
book-keeping still runs locally.

Usage example::

    from execution.binance_executor import create_exchange, fetch_usdt_balance

    exchange = create_exchange(api_key, secret, testnet=True)
    balance  = await fetch_usdt_balance(exchange)
    await exchange.close()
"""

from __future__ import annotations

import logging

import ccxt.async_support as ccxt_async

logger = logging.getLogger(__name__)


def create_exchange(
    api_key: str,
    secret: str,
    testnet: bool = False,
) -> ccxt_async.binanceusdm:
    """Create and return a configured ccxt async Binance Futures client.

    Parameters
    ----------
    api_key:
        Binance API key (production or testnet).
    secret:
        Binance API secret (production or testnet).
    testnet:
        When *True* the client is pointed at the Binance Futures Testnet
        (``https://testnet.binancefuture.com``) by overriding the exchange
        ``urls['api']`` directly.  The deprecated
        :meth:`ccxt.Exchange.set_sandbox_mode` call is intentionally avoided
        because it is no longer supported for ``binanceusdm`` futures.

    Returns
    -------
    ccxt.async_support.binanceusdm
        Ready-to-use async exchange client.
    """
    exchange = ccxt_async.binanceusdm(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }
    )
    if testnet:
        # set_sandbox_mode is deprecated for binanceusdm futures; point directly
        # to the Binance Futures Testnet base URL instead.
        exchange.urls["api"] = "https://testnet.binancefuture.com"
        logger.info(
            "[TESTNET] Binance Futures Testnet client created "
            "(base URL -> https://testnet.binancefuture.com)."
        )
    else:
        logger.info("[LIVE] Binance Futures live client created.")
    return exchange


async def fetch_usdt_balance(exchange: ccxt_async.binanceusdm) -> float | None:
    """Return the total USDT wallet balance from Binance Futures.

    Queries :meth:`ccxt.Exchange.fetch_balance` and extracts
    ``totalWalletBalance`` from the ``info`` payload (Binance Futures) or
    falls back to the top-level ``USDT.total`` field.

    Parameters
    ----------
    exchange:
        Authenticated ccxt async Binance Futures client.

    Returns
    -------
    float | None
        Total wallet balance in USDT, or *None* if the request fails.
    """
    try:
        balance = await exchange.fetch_balance()
        # Binance Futures returns `totalWalletBalance` inside `info`.
        total = balance.get("info", {}).get("totalWalletBalance")
        if total is None:
            # Fallback: ccxt normalised structure
            total = (balance.get("USDT") or {}).get("total")
        if total is not None:
            return float(total)
        logger.warning(
            "fetch_usdt_balance: could not extract USDT balance from response."
        )
        return None
    except (ccxt_async.NetworkError, ccxt_async.ExchangeError) as exc:
        logger.warning("fetch_usdt_balance failed: %s", exc)
        return None
