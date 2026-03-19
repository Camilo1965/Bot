"""
execution.binance_executor
~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory helpers for creating and configuring a ccxt async Binance Futures
exchange client.

When ``USE_BINANCE_TESTNET=True`` is set in the environment the client
returned by :func:`create_exchange` is pointed at the Binance Futures Demo
environment (``https://testnet.binancefuture.com``) instead of the production
endpoints.  This is the official Binance USDT-M Futures Demo/Paper-trading
environment; it uses the same hostname as the legacy testnet but is now
referred to as *Demo Mode* in the Binance documentation.

The same client is passed to :class:`~execution.paper_executor.PaperExecutor`
so that real orders are routed to the Demo environment while the
paper-simulation book-keeping still runs locally.

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
        Binance API key (production or Demo).
    secret:
        Binance API secret (production or Demo).
    testnet:
        When *True* the client is pointed at the Binance Futures Demo
        environment (``https://testnet.binancefuture.com``) by overriding the
        exchange ``urls['api']`` directly.  The deprecated
        :meth:`ccxt.Exchange.set_sandbox_mode` call is intentionally avoided
        because it is no longer supported for ``binanceusdm`` futures.

        The URL map covers all sub-keys used by USDT-M Futures calls:

        * ``fapiPublic`` / ``fapiPrivate`` → ``/fapi/v1``
        * ``fapiPrivateV2`` → ``/fapi/v2``  (required for the account-info
          endpoint used by :func:`fetch_usdt_balance`)

        The ``sapi`` key (Binance Spot/Wallet API) is intentionally **not**
        included; pointing it at the Futures Demo host would return error
        ``-5000`` because ``/capital/config/getall`` and related paths are
        not valid on a Futures connection.

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
        # set_sandbox_mode is deprecated for binanceusdm futures; point
        # directly to the Binance Futures Demo base URLs instead.
        # fapiPrivateV2 must be listed explicitly so that calls routed through
        # that sub-key (e.g. fapiPrivateV2GetAccount) reach the correct path.
        # sapi is deliberately omitted: it maps to Spot/Wallet endpoints that
        # do not exist on the Futures Demo host and cause error -5000.
        demo_base = "https://testnet.binancefuture.com"
        exchange.urls["api"] = {
            "fapiPublic": demo_base + "/fapi/v1",
            "fapiPrivate": demo_base + "/fapi/v1",
            "fapiPrivateV2": demo_base + "/fapi/v2",
        }
        logger.info(
            "[DEMO] Binance Futures Demo client created "
            "(fapiPublic/fapiPrivate -> %s/fapi/v1, "
            "fapiPrivateV2 -> %s/fapi/v2).",
            demo_base,
            demo_base,
        )
    else:
        logger.info("[LIVE] Binance Futures live client created.")
    return exchange


async def fetch_usdt_balance(exchange: ccxt_async.binanceusdm) -> float | None:
    """Return the total USDT wallet balance from Binance Futures.

    Uses the Futures-specific ``/fapi/v2/account`` endpoint directly via
    :meth:`ccxt.Exchange.fapiPrivateV2GetAccount` to avoid hitting any
    Spot/SAPI endpoints that are not valid for Futures-only credentials.

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
        response = await exchange.fapiPrivateV2GetAccount()
        assets = response.get("assets", [])
        for asset in assets:
            if asset.get("asset") == "USDT":
                total = asset.get("walletBalance")
                if total is not None:
                    return float(total)
        logger.warning(
            "fetch_usdt_balance: could not extract USDT balance from response."
        )
        return None
    except (ccxt_async.NetworkError, ccxt_async.ExchangeError) as exc:
        logger.warning("fetch_usdt_balance failed: %s", exc)
        return None
