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

# Quote currencies recognised as USDT-M Futures settlement assets.  Used when
# converting raw Binance symbol strings (e.g. "BTCUSDT") to ccxt format
# ("BTC/USDT").  Ordered longest-first to avoid partial matches (e.g. matching
# "USD" inside "USDT" when USD is ever added).
_FUTURES_QUOTE_CURRENCIES: tuple[str, ...] = ("USDT", "USDC", "BUSD")


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
        * ``fapiPrivateV3`` → ``/fapi/v2``  (Testnet does not expose
          ``/fapi/v3``; routing V3 sub-key calls through the V2 base path
          ensures that ``fetch_positions()`` and similar endpoints that newer
          ccxt versions route via ``fapiPrivateV3`` still resolve correctly)

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
            # Testnet does not expose /fapi/v3 endpoints; route them to /fapi/v2
            # so that calls such as fetch_positions() (which ccxt maps to
            # fapiPrivateV3GetPositionRisk on newer library versions) resolve
            # correctly on the Demo environment.
            "fapiPrivateV3": demo_base + "/fapi/v2",
        }
        logger.info(
            "[DEMO] Binance Futures Demo client created "
            "(fapiPublic/fapiPrivate -> %s/fapi/v1, "
            "fapiPrivateV2/fapiPrivateV3 -> %s/fapi/v2).",
            demo_base,
            demo_base,
        )
    else:
        logger.info("[LIVE] Binance Futures live client created.")
    return exchange


def _ccxt_symbol(binance_symbol: str) -> str:
    """Convert a Binance raw symbol string to ccxt format.

    For example ``"BTCUSDT"`` → ``"BTC/USDT"``.  The conversion handles the
    common quote currencies used in USDT-M Futures (USDT, USDC, BUSD) and
    returns the input unchanged when no matching suffix is found.
    """
    for quote in _FUTURES_QUOTE_CURRENCIES:
        if binance_symbol.endswith(quote):
            return binance_symbol[: -len(quote)] + "/" + quote
    return binance_symbol


async def fetch_open_positions(exchange: ccxt_async.binanceusdm) -> list[dict]:
    """Return a list of currently open positions from Binance Futures.

    Queries the Binance Futures positions endpoint and filters to entries
    with a non-zero contract amount (i.e. positions that are genuinely open).

    On the Binance Futures Testnet (Demo) environment the ccxt unified
    ``fetch_positions()`` method may be routed to a ``fapiPrivateV3`` endpoint
    that is not available on the Demo host.  If that primary call fails the
    function transparently falls back to the ``fapiPrivateV2GetPositionRisk``
    endpoint, which *is* supported on the Testnet, and maps the raw Binance
    response to the same output format.

    Parameters
    ----------
    exchange:
        Authenticated ccxt async Binance Futures client.

    Returns
    -------
    list[dict]
        One entry per open position, each containing:

        * ``symbol`` – ccxt-style trading pair (e.g. ``"BTC/USDT"``).
        * ``entry_price`` – average entry price as a float.
        * ``position_size`` – absolute notional value in USDT (``|contracts × entry_price|``).
        * ``side`` – ``"long"`` or ``"short"``.

        Returns an empty list if the request fails or no positions are open.
    """
    # ------------------------------------------------------------------
    # Primary path: unified ccxt method (works on Live / V3-capable hosts)
    # ------------------------------------------------------------------
    try:
        positions: list[dict] = await exchange.fetch_positions()
        open_positions: list[dict] = []
        for pos in positions:
            contracts = float(pos.get("contracts") or 0)
            if contracts == 0.0:
                continue
            symbol: str = pos.get("symbol", "")
            entry_price = float(pos.get("entryPrice") or 0)
            if not symbol or entry_price <= 0:
                continue
            # Prefer the exchange-reported notional; fall back to contracts × entry_price.
            notional = abs(float(pos.get("notional") or 0)) or abs(contracts * entry_price)
            side: str = pos.get("side") or "long"
            open_positions.append(
                {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "position_size": notional,
                    "side": side,
                }
            )
        logger.info(
            "fetch_open_positions: %d open position(s) found: %s",
            len(open_positions),
            [p["symbol"] for p in open_positions],
        )
        return open_positions
    except (ccxt_async.NetworkError, ccxt_async.ExchangeError) as exc:
        logger.warning(
            "fetch_open_positions: primary fetch_positions() failed (%s). "
            "Retrying via fapiPrivateV2GetPositionRisk (Testnet fallback)…",
            exc,
        )

    # ------------------------------------------------------------------
    # Fallback path: raw V2 endpoint – fully supported on Testnet/Demo.
    # ------------------------------------------------------------------
    try:
        raw: list[dict] = await exchange.fapiPrivateV2GetPositionRisk()
        open_positions = []
        for pos in raw:
            amt = float(pos.get("positionAmt") or 0)
            if amt == 0.0:
                continue
            raw_symbol: str = pos.get("symbol", "")
            entry_price = float(pos.get("entryPrice") or 0)
            if not raw_symbol or entry_price <= 0:
                continue
            symbol_ccxt = _ccxt_symbol(raw_symbol)
            notional = abs(amt * entry_price)
            side = "long" if amt > 0 else "short"
            open_positions.append(
                {
                    "symbol": symbol_ccxt,
                    "entry_price": entry_price,
                    "position_size": notional,
                    "side": side,
                }
            )
        logger.info(
            "fetch_open_positions (V2 fallback): %d open position(s) found: %s",
            len(open_positions),
            [p["symbol"] for p in open_positions],
        )
        return open_positions
    except (ccxt_async.NetworkError, ccxt_async.ExchangeError) as exc2:
        logger.error(
            "fetch_open_positions: V2 fallback also failed (%s). Returning empty list.",
            exc2,
        )
        return []


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
