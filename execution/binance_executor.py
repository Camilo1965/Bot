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

    from execution.binance_executor import create_exchange, fetch_total_wallet_balance

    exchange = create_exchange(api_key, secret, testnet=True)
    balance  = await fetch_total_wallet_balance(exchange)
    await exchange.close()
"""

from __future__ import annotations

import logging

import ccxt.async_support as ccxt_async

logger = logging.getLogger(__name__)

# Stablecoins whose wallet balance can be counted 1:1 against USD without
# fetching a market price.  BUSD is retained in the set because some accounts
# still hold residual BUSD balances after the February 2024 delisting; it
# is safe to count any such residual at face value.  New stablecoins are
# added here as Binance lists them as USDT-M margin collateral.
_STABLECOINS: frozenset[str] = frozenset(
    {"USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "USDE"}
)


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
          endpoint used by :func:`fetch_total_wallet_balance`)

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
                # Disable automatic currency metadata fetch which relies on
                # sapi endpoints that are not available on the Futures Testnet.
                "fetchCurrencies": False,
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


async def fetch_total_wallet_balance(exchange: ccxt_async.binanceusdm) -> float | None:
    """Return the total wallet balance (USD equivalent) from Binance Futures.

    Uses the Futures-specific ``/fapi/v2/account`` endpoint directly via
    :meth:`ccxt.Exchange.fapiPrivateV2GetAccount` to avoid hitting any
    Spot/SAPI endpoints that are not valid for Futures-only credentials.

    **Multi-Asset Mode support**

    When the account operates in Multi-Asset Mode the top-level
    ``totalWalletBalance`` field may only reflect the USDT portion of the
    wallet.  This function therefore inspects the per-asset ``assets`` array
    and aggregates the USD-equivalent value of every asset that carries a
    non-zero ``walletBalance``:

    * **Stablecoins** (USDT, USDC, BUSD, …) are counted 1 : 1 against USD.
    * **All other assets** (e.g. BTC) are converted to USDT by fetching the
      current mark price via :meth:`ccxt.Exchange.fetch_ticker`.  If the
      price cannot be obtained the asset is excluded from the total and a
      warning is logged so the issue is visible without crashing the bot.

    When the ``assets`` array is absent or every entry has a zero balance the
    function falls back to the top-level ``totalWalletBalance`` field
    (single-asset mode compatibility).

    Parameters
    ----------
    exchange:
        Authenticated ccxt async Binance Futures client.

    Returns
    -------
    float | None
        Total wallet balance across all margin assets, expressed in USDT, or
        *None* if the request fails entirely.
    """
    try:
        response = await exchange.fapiPrivateV2GetAccount()

        # ------------------------------------------------------------------
        # Multi-Asset Mode: iterate over the per-asset breakdown and convert
        # each non-zero balance to its USDT equivalent.
        # ------------------------------------------------------------------
        assets: list[dict] = response.get("assets") or []
        # Parse walletBalance once per entry to avoid redundant conversions.
        non_zero: list[tuple[dict, float]] = []
        for a in assets:
            balance = float(a.get("walletBalance") or 0)
            if balance != 0.0:
                non_zero.append((a, balance))

        if non_zero:
            total_usdt = 0.0
            for asset_info, balance in non_zero:
                asset_name: str = asset_info.get("asset", "")

                if asset_name in _STABLECOINS:
                    total_usdt += balance
                    logger.debug(
                        "fetch_total_wallet_balance: %s %.4f (stablecoin, 1:1)",
                        asset_name,
                        balance,
                    )
                else:
                    # Fetch the perpetual futures mark price for this asset.
                    symbol = f"{asset_name}/USDT"
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        price = float(ticker.get("last") or ticker.get("close") or 0)
                        if price == 0.0:
                            logger.warning(
                                "fetch_total_wallet_balance: ticker for %s returned"
                                " no valid price – asset excluded from total.",
                                symbol,
                            )
                        else:
                            usdt_value = balance * price
                            total_usdt += usdt_value
                            logger.debug(
                                "fetch_total_wallet_balance: %s %.8f @ %.4f = %.4f USDT",
                                asset_name,
                                balance,
                                price,
                                usdt_value,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "fetch_total_wallet_balance: could not fetch price for %s"
                            " (%s) – asset excluded from total.",
                            symbol,
                            exc,
                        )

            logger.info(
                "fetch_total_wallet_balance: aggregated %d asset(s) → %.4f USDT",
                len(non_zero),
                total_usdt,
            )
            return total_usdt

        # ------------------------------------------------------------------
        # Single-Asset Mode fallback: use the pre-aggregated top-level field.
        # ------------------------------------------------------------------
        total = response.get("totalWalletBalance")
        if total is not None:
            return float(total)

        logger.warning(
            "fetch_total_wallet_balance: could not extract balance from response."
        )
        return None
    except (ccxt_async.NetworkError, ccxt_async.ExchangeError) as exc:
        logger.warning("fetch_total_wallet_balance failed: %s", exc)
        return None
