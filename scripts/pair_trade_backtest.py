#!/usr/bin/env python3
"""
scripts/pair_trade_backtest.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BTC/ETH pair-trade backtest: 90d at 15m, train on first 60d, test on last 30d.

Each signal opens a simultaneous long+short position with equal notional
sized by ATR. Positions are closed when z-score reverts (CLOSE signal) or
stops out (z >= z_stop).

Usage::

    python scripts/pair_trade_backtest.py [--days 90] [--train-days 60] \
                                          [--window 60] [--z-entry 2.0] \
                                          [--z-exit 0.5] [--z-stop 4.0] \
                                          [--notional 1000]

Output: logs/pair_trade_backtest_90d.json + console summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.ohlcv_cache import get_cache
from strategy.pair_trader import PairTrader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("pair_trade_backtest")

_OUTPUT_JSON = _REPO / "logs" / "pair_trade_backtest_90d.json"

# ── Execution cost defaults (same as BTC/ETH symbol configs) ─────────────────
_SPREAD_FRAC = 2.0 / 10000     # 2 bps each leg
_COMMISSION_PCT = 0.0004        # 0.04% per open/close leg

# Candles per day for 15m
_CPD_15M = 96


@dataclass
class PairTrade:
    direction: str          # "BUY_ETH_SELL_BTC" or "BUY_BTC_SELL_ETH"
    open_idx: int
    btc_entry: float
    eth_entry: float
    notional: float         # USD notional per leg
    stop_zscore: float
    pnl_usd: float = 0.0
    close_idx: Optional[int] = None
    close_reason: str = ""


def _fetch_pair(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV via cache, fallback to CCXT."""
    try:
        df = get_cache().fetch(symbol, timeframe, limit=limit, max_age_s=3600)
        if df is not None and len(df) >= limit // 2:
            logger.info("[%s] Cache hit: %d rows", symbol, len(df))
            return df
    except Exception as exc:
        logger.warning("[%s] Cache miss: %s", symbol, exc)

    try:
        from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
        df = fetch_ohlcv_ccxt(symbol, timeframe=timeframe, limit=limit)
        if df is not None and len(df) > 0:
            logger.info("[%s] CCXT: %d rows", symbol, len(df))
            return df
    except Exception as exc2:
        logger.error("[%s] CCXT fallback failed: %s", symbol, exc2)

    return pd.DataFrame()


def _align_pair(btc: pd.DataFrame, eth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align BTC and ETH dataframes on common timestamps."""
    if btc.empty or eth.empty:
        return btc, eth

    btc = btc.set_index("timestamp").sort_index()
    eth = eth.set_index("timestamp").sort_index()
    common = btc.index.intersection(eth.index)
    if len(common) == 0:
        logger.error("No common timestamps between BTC and ETH — cannot align.")
        return pd.DataFrame(), pd.DataFrame()

    return btc.loc[common].reset_index(), eth.loc[common].reset_index()


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR series."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _simulate_pair(
    btc_test: pd.DataFrame,
    eth_test: pd.DataFrame,
    trader: PairTrader,
    notional_usd: float,
) -> list[PairTrade]:
    """Simulate pair trades on test split."""
    btc_close = btc_test["close"].astype(float).reset_index(drop=True)
    eth_close = eth_test["close"].astype(float).reset_index(drop=True)
    btc_atr = _atr_series(btc_test).reset_index(drop=True)
    eth_atr = _atr_series(eth_test).reset_index(drop=True)

    n = min(len(btc_close), len(eth_close))
    open_trade: PairTrade | None = None
    closed: list[PairTrade] = []

    # Rebuild ratio/zscore for test period using train + test (rolling)
    # We feed prices one by one, appending to internal state via signal()
    # The PairTrader.signal() method handles the rolling window itself.

    for i in range(n):
        btc_p = float(btc_close.iloc[i])
        eth_p = float(eth_close.iloc[i])
        if btc_p <= 0 or eth_p <= 0:
            continue

        sig = trader.signal(btc_p, eth_p)
        zscore = trader.current_zscore() or 0.0

        # Manage open trade
        if open_trade is not None:
            # Check stop-loss (z diverged further)
            if abs(zscore) >= open_trade.stop_zscore:
                pnl = _calc_pnl(open_trade, btc_p, eth_p)
                open_trade.pnl_usd = pnl
                open_trade.close_idx = i
                open_trade.close_reason = "STOP"
                closed.append(open_trade)
                open_trade = None
                continue

            # Check close signal (z reverted)
            if sig == "CLOSE":
                pnl = _calc_pnl(open_trade, btc_p, eth_p)
                open_trade.pnl_usd = pnl
                open_trade.close_idx = i
                open_trade.close_reason = "REVERT"
                closed.append(open_trade)
                open_trade = None
                continue

        # Open new trade (only if no position open)
        if open_trade is None and sig in ("BUY_ETH_SELL_BTC", "BUY_BTC_SELL_ETH"):
            # ATR-based notional adjustment: scale down if ATR is large
            atr_btc = float(btc_atr.iloc[i]) if i < len(btc_atr) else 0.0
            atr_eth = float(eth_atr.iloc[i]) if i < len(eth_atr) else 0.0
            atr_btc_pct = atr_btc / btc_p if btc_p > 0 else 0.01
            atr_eth_pct = atr_eth / eth_p if eth_p > 0 else 0.01
            avg_atr_pct = (atr_btc_pct + atr_eth_pct) / 2
            # Scale notional: target 1% risk per leg (SL = 1 ATR)
            atr_notional = (notional_usd * 0.01) / max(avg_atr_pct, 0.001)
            leg_notional = min(notional_usd, atr_notional)

            open_trade = PairTrade(
                direction=sig,
                open_idx=i,
                btc_entry=btc_p,
                eth_entry=eth_p,
                notional=leg_notional,
                stop_zscore=trader.z_stop,
            )

    # Close any remaining trade at last price
    if open_trade is not None:
        btc_last = float(btc_close.iloc[-1])
        eth_last = float(eth_close.iloc[-1])
        pnl = _calc_pnl(open_trade, btc_last, eth_last)
        open_trade.pnl_usd = pnl
        open_trade.close_idx = n - 1
        open_trade.close_reason = "END_OF_DATA"
        closed.append(open_trade)

    return closed


def _calc_pnl(trade: PairTrade, btc_exit: float, eth_exit: float) -> float:
    """
    Compute USD PnL for a pair trade.

    For BUY_ETH_SELL_BTC:
      Long ETH leg: (eth_exit - eth_entry) / eth_entry * notional
      Short BTC leg: (btc_entry - btc_exit) / btc_entry * notional

    For BUY_BTC_SELL_ETH:
      Long BTC leg: (btc_exit - btc_entry) / btc_entry * notional
      Short ETH leg: (eth_entry - eth_exit) / eth_entry * notional
    """
    if trade.direction == "BUY_ETH_SELL_BTC":
        long_ret = (eth_exit - trade.eth_entry) / trade.eth_entry
        short_ret = (trade.btc_entry - btc_exit) / trade.btc_entry
    else:  # BUY_BTC_SELL_ETH
        long_ret = (btc_exit - trade.btc_entry) / trade.btc_entry
        short_ret = (trade.eth_entry - eth_exit) / trade.eth_entry

    gross_pnl = (long_ret + short_ret) * trade.notional
    # Commission: 4 legs (open long, open short, close long, close short)
    commission = trade.notional * _COMMISSION_PCT * 4
    spread_cost = trade.notional * _SPREAD_FRAC * 4
    return gross_pnl - commission - spread_cost


def _calc_metrics(
    trades: list[PairTrade],
    btc_test: pd.DataFrame,
    notional: float,
) -> dict[str, Any]:
    """Compute backtest metrics from closed trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_usd": 0.0,
            "pnl_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "correlation_with_btc": None,
        }

    pnls = np.array([t.pnl_usd for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_pnl = float(pnls.sum())
    win_rate = len(wins) / len(pnls)
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    # Max drawdown on cumulative PnL curve
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd_usd = float(dd.max()) if len(dd) > 0 else 0.0

    # Sharpe (annualised): use per-trade returns relative to notional
    ret_pct = pnls / (notional * 2)  # both legs combined notional
    mean_r = float(np.mean(ret_pct))
    std_r = float(np.std(ret_pct))
    n_trades_per_year_est = len(trades) / max(1, len(btc_test) / _CPD_15M / 365)
    sharpe = (mean_r / std_r) * np.sqrt(n_trades_per_year_est) if std_r > 0 else 0.0

    # Correlation of pair trade PnL with BTC returns
    btc_ret = btc_test["close"].astype(float).pct_change().dropna()
    correlation: float | None = None
    if len(pnls) > 2 and len(btc_ret) > 2:
        # Map trade PnL to bars for correlation — use trade open_idx
        trade_indices = [t.open_idx for t in trades if t.open_idx < len(btc_ret)]
        if len(trade_indices) > 2:
            btc_rets_at_entry = btc_ret.iloc[trade_indices].values
            trade_pnls = np.array([pnls[i] for i, t in enumerate(trades) if t.open_idx < len(btc_ret)])
            if len(btc_rets_at_entry) > 2:
                corr = np.corrcoef(trade_pnls, btc_rets_at_entry)[0, 1]
                correlation = round(float(corr), 4) if not np.isnan(corr) else None

    pnl_pct = total_pnl / (notional * 2) * 100  # as % of total deployed notional

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "pnl_usd": round(total_pnl, 4),
        "pnl_pct": round(pnl_pct, 4),
        "max_drawdown_usd": round(max_dd_usd, 4),
        "max_drawdown_pct": round(max_dd_usd / (notional * 2) * 100, 4) if notional > 0 else 0.0,
        "sharpe": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "correlation_with_btc": correlation,
        "wins": int(len(wins)),
        "losses": int(len(losses)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BTC/ETH pair trade backtest (90d 15m)")
    parser.add_argument("--days", type=int, default=90, help="Total days to fetch (default 90)")
    parser.add_argument("--train-days", type=int, default=60, help="Training period in days (default 60)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe (default 15m)")
    parser.add_argument("--window", type=int, default=60, help="Z-score rolling window (default 60 bars)")
    parser.add_argument("--z-entry", type=float, default=2.0, help="Z-score entry threshold (default 2.0)")
    parser.add_argument("--z-exit", type=float, default=0.5, help="Z-score exit threshold (default 0.5)")
    parser.add_argument("--z-stop", type=float, default=4.0, help="Z-score stop threshold (default 4.0)")
    parser.add_argument("--notional", type=float, default=1000.0, help="USD notional per leg (default 1000)")
    args = parser.parse_args(argv)

    tf = args.timeframe
    cpd = _CPD_15M if tf == "15m" else max(1, int(1440 // int(tf.rstrip("m"))))
    total_limit = args.days * cpd
    train_bars = args.train_days * cpd

    logger.info("Fetching BTC/USDT and ETH/USDT (%dd at %s)...", args.days, tf)
    btc_raw = _fetch_pair("BTC/USDT", tf, total_limit)
    eth_raw = _fetch_pair("ETH/USDT", tf, total_limit)

    if btc_raw.empty or eth_raw.empty:
        logger.error("Failed to fetch data for one or both symbols.")
        return 1

    btc_aligned, eth_aligned = _align_pair(btc_raw, eth_raw)
    if btc_aligned.empty or eth_aligned.empty:
        logger.error("No overlapping data between BTC and ETH.")
        return 1

    logger.info("Aligned dataset: %d rows", len(btc_aligned))

    if len(btc_aligned) < train_bars + 50:
        logger.error(
            "Not enough aligned rows (%d) for train_bars=%d + test window.",
            len(btc_aligned), train_bars,
        )
        return 1

    # Split train/test
    btc_train = btc_aligned.iloc[:train_bars].copy()
    eth_train = eth_aligned.iloc[:train_bars].copy()
    btc_test = btc_aligned.iloc[train_bars:].copy().reset_index(drop=True)
    eth_test = eth_aligned.iloc[train_bars:].copy().reset_index(drop=True)

    test_days = len(btc_test) / cpd

    logger.info(
        "Train: %d bars (%.1fd) | Test: %d bars (%.1fd)",
        len(btc_train), len(btc_train) / cpd,
        len(btc_test), test_days,
    )

    # Fit PairTrader on train set
    trader = PairTrader(
        window=args.window,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        z_stop=args.z_stop,
    )
    is_coint = trader.fit(
        btc_train["close"].astype(float),
        eth_train["close"].astype(float),
    )

    logger.info(
        "PairTrader fit: cointegrated=%s (tested=%s) | z_last=%.4f",
        is_coint,
        trader.cointegration_tested(),
        trader.current_zscore() or float("nan"),
    )

    if not is_coint:
        logger.warning(
            "Pair NOT cointegrated (p > %.2f). "
            "Proceeding with backtest anyway — interpret results cautiously.",
            trader.coint_pvalue,
        )

    # Simulate on test set
    closed_trades = _simulate_pair(btc_test, eth_test, trader, notional_usd=args.notional)
    logger.info("Simulation complete: %d closed trades", len(closed_trades))

    metrics = _calc_metrics(closed_trades, btc_test, args.notional)

    # BTC buy-and-hold for reference
    bnh_btc = (float(btc_test["close"].iloc[-1]) - float(btc_test["close"].iloc[0])) / float(btc_test["close"].iloc[0]) * 100 if len(btc_test) > 1 else 0.0

    report = {
        "config": {
            "days_total": args.days,
            "train_days": args.train_days,
            "test_days": round(test_days, 1),
            "timeframe": tf,
            "window": args.window,
            "z_entry": args.z_entry,
            "z_exit": args.z_exit,
            "z_stop": args.z_stop,
            "notional_per_leg_usd": args.notional,
        },
        "cointegration": {
            "cointegrated": is_coint,
            "cointegration_tested": trader.cointegration_tested(),
            "pvalue": trader._coint_pvalue_result,
        },
        "metrics": metrics,
        "reference": {
            "btc_buy_hold_pct": round(bnh_btc, 4),
        },
        "trade_log": [
            {
                "direction": t.direction,
                "open_idx": t.open_idx,
                "close_idx": t.close_idx,
                "btc_entry": round(t.btc_entry, 4),
                "eth_entry": round(t.eth_entry, 4),
                "notional": round(t.notional, 2),
                "pnl_usd": round(t.pnl_usd, 4),
                "close_reason": t.close_reason,
            }
            for t in closed_trades
        ],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("PAIR TRADE BACKTEST RESULTS (BTC/ETH)")
    print("=" * 60)
    m = metrics
    print(f"  Trades:              {m['total_trades']}")
    print(f"  Win rate:            {m['win_rate']:.1%}")
    print(f"  PnL (USD):           ${m['pnl_usd']:,.2f}")
    print(f"  PnL (%):             {m['pnl_pct']:.2f}%")
    print(f"  Max drawdown:        ${m['max_drawdown_usd']:,.2f}  ({m['max_drawdown_pct']:.2f}%)")
    print(f"  Profit factor:       {m['profit_factor']:.2f}")
    print(f"  Sharpe:              {m['sharpe']:.2f}")
    print(f"  Corr with BTC:       {m['correlation_with_btc']}")
    print(f"  BTC buy-and-hold:    {bnh_btc:.2f}%")
    print(f"  Cointegrated:        {is_coint} (tested={trader.cointegration_tested()})")
    print("=" * 60)

    _OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved → %s", _OUTPUT_JSON)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
