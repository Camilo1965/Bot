"""Preload MT5 candles into shared state buffers at startup."""

from __future__ import annotations

import logging
from typing import Any

from bot.dashboard_helpers import dashboard_rsi_timeframe
from execution.mt5_executor import TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_M15, MT5Executor

_RSI_TF_TO_MT5 = {"15m": TIMEFRAME_M15, "1h": TIMEFRAME_H1, "4h": TIMEFRAME_H4}


async def preload_historical_data_mt5(
    state: dict[str, Any],
    watchlist: list[str],
    executor: MT5Executor,
    limit: int = 1000,
) -> None:
    """Preload 15m/1h/4h candles from MT5 to warm up buffers."""
    log = logging.getLogger("clawdbot.preload")
    for symbol in watchlist:
        try:
            df_15m = await executor.fetch_candles(
                symbol=symbol,
                timeframe=TIMEFRAME_M15,
                count=limit,
                start_pos=0,
            )
            if df_15m is not None and not df_15m.empty:
                for _, row in df_15m.iterrows():
                    close = row.get("close")
                    high = row.get("high")
                    low = row.get("low")
                    if close is not None and symbol in state["prices"]:
                        state["prices"][symbol].append(float(close))
                    if high is not None and symbol in state.get("highs", {}):
                        state["highs"][symbol].append(float(high))
                    if low is not None and symbol in state.get("lows", {}):
                        state["lows"][symbol].append(float(low))
                log.info("[MT5] Preloaded %d 15m candles for %s.", len(df_15m), symbol)
            else:
                log.warning("[MT5] No 15m candles available for %s.", symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("[MT5] Could not preload 15m candles for %s: %s", symbol, exc)

        rsi_tf_name = dashboard_rsi_timeframe()
        rsi_mt5 = _RSI_TF_TO_MT5.get(rsi_tf_name, TIMEFRAME_M15)
        try:
            df_rsi = await executor.fetch_candles(
                symbol=symbol,
                timeframe=rsi_mt5,
                count=limit,
                start_pos=0,
            )
            rsi_d = state.get("dashboard_rsi_closes", {}).get(symbol)
            if rsi_d is not None and df_rsi is not None and not df_rsi.empty:
                rsi_d.clear()
                for _, row in df_rsi.iterrows():
                    c = row.get("close")
                    if c is not None:
                        rsi_d.append(float(c))
                log.info(
                    "[MT5] Preloaded %d %s closes for dashboard RSI (%s).",
                    len(df_rsi),
                    rsi_tf_name,
                    symbol,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("[MT5] Could not preload RSI series for %s: %s", symbol, exc)

        for tf_name, tf_value in (("1h", TIMEFRAME_H1), ("4h", TIMEFRAME_H4)):
            try:
                df_htf = await executor.fetch_candles(
                    symbol=symbol,
                    timeframe=tf_value,
                    count=limit,
                    start_pos=0,
                )
                if df_htf is None or df_htf.empty:
                    log.warning("[MT5] No %s candles available for %s.", tf_name.upper(), symbol)
                    continue
                htf_closes = state.get("htf_closes", {}).get(symbol, {})
                htf_opens = state.get("htf_opens", {}).get(symbol, {})
                for _, row in df_htf.iterrows():
                    close = row.get("close")
                    open_price = row.get("open")
                    if close is not None and tf_name in htf_closes:
                        htf_closes[tf_name].append(float(close))
                    if open_price is not None and tf_name in htf_opens:
                        htf_opens[tf_name].append(float(open_price))
                log.info("[MT5] Preloaded %d %s candles for %s.", len(df_htf), tf_name.upper(), symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[MT5] Could not preload %s candles for %s: %s",
                    tf_name.upper(),
                    symbol,
                    exc,
                )
