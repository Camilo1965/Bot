"""
backtest.py
~~~~~~~~~~~
Historical backtesting engine for a multi-symbol XGBoost + MTA strategy.

Downloads the last 6 months of 15-minute, 1-hour and 4-hour OHLCV data
from Binance via ccxt for each symbol in ``_SYMBOLS``, trains an XGBoost
model on the first 70 % of the 15-minute series, and simulates the full
XGBoost + Multi-Timeframe Analysis (MTA) strategy on the remaining 30 %.

Risk parameters
---------------
* Starting capital    : 10,000 USDT  (shared across the portfolio)
* Risk per trade      : 15 % of current balance  (fixed fractional, from risk_manager)
* Initial Stop Loss   : −1.5 %  (hard floor from entry)
* Trailing activation : +3 % profit activates the trailing stop
* Trailing distance   : 2 % below the running price peak

Sentiment mock
--------------
Historical news sentiment is unavailable, so ``_SENTIMENT_MOCK`` is set
to 0.0 (neutral).  The XGBoost base-BUY gate requires sentiment > 0.3,
so with a neutral mock most BUY signals are generated through the
mean-reversion path (ADX < 20, RSI < 35).  Raise ``_SENTIMENT_MOCK``
toward +1.0 to open the sentiment gate and observe more BUY signals.

Run
---
    python backtest.py
"""

from __future__ import annotations

import logging
import time

import ccxt
import numpy as np
import pandas as pd

from risk.risk_manager import ACTIVATION_PCT, INITIAL_SL, RISK_PER_TRADE, TRAILING_DISTANCE, LEVERAGE
from strategy.ml_predictor import MLPredictor, compute_htf_trend

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

logger = logging.getLogger(__name__)

# ── Back-test parameters ──────────────────────────────────────────────────────
_SYMBOLS          = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",  # L1_MAJOR
    "LINK/USDT", "INJ/USDT",                           # DEFI
    "FET/USDT", "RENDER/USDT",                         # AI
    "DOGE/USDT", "PEPE/USDT",                          # MEME
]
_FETCH_LIMIT      = 1_000              # max candles per ccxt request
_SIX_MONTHS_MS    = 183 * 24 * 60 * 60 * 1_000

_TRAIN_RATIO      = 0.70               # 70 % train / 30 % test  (chronological)
_STARTING_CAPITAL = 10_000.0           # USDT

# Neutral sentiment mock (no live news in historical data).
# Increase toward +1.0 to open the ML sentiment gate for more BUY signals.
_SENTIMENT_MOCK   = 0.0

_BUY_SIGNAL       = "BUY"
_MIN_WARMUP       = 20                 # minimum prices needed for inference


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Download 6 months of *timeframe* klines for *symbol* from Binance.

    Handles pagination automatically and returns a deduplicated, chronologically
    sorted DataFrame with columns:
    ``timestamp (UTC), open, high, low, close, volume``.
    """
    since = exchange.milliseconds() - _SIX_MONTHS_MS
    all_candles: list[list] = []
    logger.info("Fetching %s %s …", symbol, timeframe)

    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=_FETCH_LIMIT
        )
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        tf_ms = exchange.parse_timeframe(timeframe) * 1_000
        if last_ts >= exchange.milliseconds() - tf_ms:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1_000)

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info("  → %d %s candles downloaded.", len(df), timeframe)
    return df


# ── HTF alignment helpers ─────────────────────────────────────────────────────

def _build_htf_arrays(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Convert HTF DataFrames to sorted numpy arrays for fast binary search.

    Returns
    -------
    ``(ts_1h, cl_1h, op_1h, ts_4h, cl_4h, op_4h)``
    where ``ts_*`` arrays contain UTC timestamps as int64 nanoseconds.
    """
    ts_1h = df_1h["timestamp"].values.astype("int64")
    cl_1h = df_1h["close"].values.astype(float)
    op_1h = df_1h["open"].values.astype(float)

    ts_4h = df_4h["timestamp"].values.astype("int64")
    cl_4h = df_4h["close"].values.astype(float)
    op_4h = df_4h["open"].values.astype(float)

    return ts_1h, cl_1h, op_1h, ts_4h, cl_4h, op_4h


def _htf_slice(
    current_ts_ns: int,
    htf_ts: np.ndarray,
    htf_cl: np.ndarray,
    htf_op: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Return HTF close/open lists for all candles whose open timestamp ≤ current_ts.

    Uses ``np.searchsorted`` for O(log n) lookup per step, avoiding look-ahead
    bias: only candles that have already opened are included.
    """
    idx = int(np.searchsorted(htf_ts, current_ts_ns, side="right"))
    return htf_cl[:idx].tolist(), htf_op[:idx].tolist()


# ── Simulation engine ─────────────────────────────────────────────────────────

def _simulate(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    predictor: MLPredictor,
    *,
    split_idx: int,
) -> dict:
    """Simulate the XGBoost + MTA strategy on the *test* portion of df_15m.

    Design
    ------
    * Price / high / low buffers include **all** 15m candles up to and
      including the current index – the model sees its full history, but
      never a future candle.
    * HTF buffers are capped at timestamps ≤ the current 15m candle
      timestamp, eliminating look-ahead bias on higher timeframes.
    * After a BUY signal the loop advances directly to the exit candle
      (SL, trailing SL, or end-of-data), so trades never overlap.
    * Intra-candle highs and lows are used to update the highest price
      seen and to detect stop-loss triggers accurately within each bar.

    Parameters
    ----------
    df_15m:     Full 15-minute OHLCV DataFrame (train + test).
    df_1h:      Full 1-hour OHLCV DataFrame.
    df_4h:      Full 4-hour OHLCV DataFrame.
    predictor:  MLPredictor already warm-started on the training portion.
    split_idx:  Index in df_15m where the test period begins.

    Returns
    -------
    dict with performance metrics.
    """
    closes_15m  = df_15m["close"].values.astype(float)
    highs_15m   = df_15m["high"].values.astype(float)
    lows_15m    = df_15m["low"].values.astype(float)
    ts_15m_ns   = df_15m["timestamp"].values.astype("int64")

    ts_1h, cl_1h, op_1h, ts_4h, cl_4h, op_4h = _build_htf_arrays(df_1h, df_4h)

    balance      = _STARTING_CAPITAL
    peak_balance = _STARTING_CAPITAL
    max_drawdown = 0.0
    trades: list[dict] = []

    total_test    = len(df_15m) - split_idx
    progress_step = max(1, total_test // 20)

    i = split_idx
    while i < len(df_15m):
        # Progress logging (~5 % increments)
        step = i - split_idx
        if step % progress_step == 0:
            logger.info(
                "Simulation %3.0f %%  –  balance=%.2f USDT  trades=%d",
                step / total_test * 100,
                balance,
                len(trades),
            )

        # ── 1. Build price buffers (no look-ahead) ─────────────────────────
        prices_buf = closes_15m[: i + 1].tolist()
        highs_buf  = highs_15m[: i + 1].tolist()
        lows_buf   = lows_15m[: i + 1].tolist()

        if len(prices_buf) < _MIN_WARMUP:
            i += 1
            continue

        # ── 2. Align HTF candles ────────────────────────────────────────────
        cur_ts_ns = ts_15m_ns[i]
        htf_cl_1h, htf_op_1h = _htf_slice(cur_ts_ns, ts_1h, cl_1h, op_1h)
        htf_cl_4h, htf_op_4h = _htf_slice(cur_ts_ns, ts_4h, cl_4h, op_4h)

        # ── 3. Multi-Timeframe Analysis (MTA) ──────────────────────────────
        trend_4h = compute_htf_trend(htf_cl_4h, htf_op_4h if htf_op_4h else None)
        trend_1h = compute_htf_trend(htf_cl_1h, htf_op_1h if htf_op_1h else None)

        # ── 4. ML signal ────────────────────────────────────────────────────
        signal = predictor.generate_signal(
            prices_buf,
            _SENTIMENT_MOCK,
            highs=highs_buf,
            lows=lows_buf,
            obi_ratio=1.0,
            funding_rate=0.0,
            htf_trend_4h=trend_4h,
            htf_trend_1h=trend_1h,
        )

        if signal != _BUY_SIGNAL:
            # Track drawdown even while no trade is open
            if peak_balance > 0:
                dd = (peak_balance - balance) / peak_balance * 100.0
                if dd > max_drawdown:
                    max_drawdown = dd
            i += 1
            continue

        # ── 5. Position sizing (fixed fractional risk) ─────────────────────
        entry_price      = float(closes_15m[i])
        position_usdt    = balance * RISK_PER_TRADE
        initial_sl_price = entry_price * (1.0 - INITIAL_SL)
        # Active SL starts at the hard initial stop and can only move up
        active_sl        = initial_sl_price
        highest_seen     = entry_price

        # ── 6. Scan subsequent candles for exit ─────────────────────────────
        # Use the candle's high to update the highest price seen and to
        # potentially raise the trailing SL, then use the candle's low to
        # check whether the active stop loss has been breached.
        outcome_pnl = None
        exit_idx    = len(df_15m) - 1
        exit_reason = "EOD"

        for j in range(i + 1, len(df_15m)):
            high_j = float(highs_15m[j])
            low_j  = float(lows_15m[j])

            # Track intra-candle high for trailing stop calculation
            if high_j > highest_seen:
                highest_seen = high_j

            # Activate trailing stop once ACTIVATION_PCT profit is reached
            if (highest_seen - entry_price) / entry_price >= ACTIVATION_PCT:
                tsl = highest_seen * (1.0 - TRAILING_DISTANCE)
                if tsl > active_sl:
                    active_sl = tsl

            # Exit when the candle's low falls to or below the active SL
            if low_j <= active_sl:
                exit_price  = active_sl
                pnl_pct     = (exit_price - entry_price) / entry_price
                outcome_pnl = (position_usdt * LEVERAGE) * pnl_pct
                # Label the exit: TSL when the trailing stop has moved above
                # the initial SL; plain SL otherwise.
                exit_reason = "TSL" if active_sl > initial_sl_price else "SL"
                exit_idx    = j
                break

        if outcome_pnl is None:
            # Reached end of data without hitting TP or SL; close at last close
            last_price  = float(closes_15m[-1])
            pnl_pct     = (last_price - entry_price) / entry_price
            outcome_pnl = (position_usdt * LEVERAGE) * pnl_pct

        # ── 7. Update portfolio ─────────────────────────────────────────────
        balance += outcome_pnl
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100.0 if peak_balance > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

        trades.append(
            {
                "entry_price": entry_price,
                "exit_reason": exit_reason,
                "pnl_usdt":   outcome_pnl,
                "win":        outcome_pnl > 0,
            }
        )

        # Advance to the candle after the exit (no overlapping trades)
        i = exit_idx + 1

    total_trades  = len(trades)
    wins          = sum(1 for t in trades if t["win"])
    total_pnl     = balance - _STARTING_CAPITAL
    win_rate      = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl_pct = (total_pnl / _STARTING_CAPITAL) * 100.0

    return {
        "total_trades":     total_trades,
        "wins":             wins,
        "win_rate_pct":     win_rate,
        "total_pnl_usdt":  total_pnl,
        "total_pnl_pct":   total_pnl_pct,
        "max_drawdown_pct": max_drawdown,
        "final_balance":    balance,
    }


# ── Performance report ────────────────────────────────────────────────────────

def _print_report(stats: dict, train_rows: int, test_rows: int, symbol: str = "") -> None:
    """Print a colorized performance summary to the terminal."""
    pnl     = stats["total_pnl_usdt"]
    pnl_pct = stats["total_pnl_pct"]
    dd      = stats["max_drawdown_pct"]
    wr      = stats["win_rate_pct"]

    pnl_color = _GREEN if pnl >= 0 else _RED
    wr_color  = _GREEN if wr >= 50 else _RED
    dd_color  = _GREEN if dd < 10 else (_YELLOW if dd < 20 else _RED)

    w = 60
    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'  BACKTEST PERFORMANCE REPORT':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"  {_DIM}Symbol       :{_RESET}  {_BOLD}{symbol}  (15m / 1H / 4H MTA){_RESET}")
    print(f"  {_DIM}Train candles:{_RESET}  {train_rows:,}")
    print(f"  {_DIM}Test candles :{_RESET}  {test_rows:,}")
    print(f"  {_DIM}Risk / trade :{_RESET}  {RISK_PER_TRADE * 100:.0f} %  "
          f"(activation {ACTIVATION_PCT * 100:.1f} % / trailing {TRAILING_DISTANCE * 100:.1f} % / "
          f"initial SL {INITIAL_SL * 100:.2f} %)")
    print(f"  {_DIM}Starting cap :{_RESET}  {_STARTING_CAPITAL:,.2f} USDT")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"  {_DIM}Total trades :{_RESET}  {_BOLD}{stats['total_trades']}{_RESET}")
    print(f"  {_DIM}Winning trades:{_RESET} {stats['wins']}")
    print(f"  {_DIM}Win Rate     :{_RESET}  "
          f"{wr_color}{_BOLD}{wr:.2f} %{_RESET}")
    print(f"  {_DIM}Total PnL    :{_RESET}  "
          f"{pnl_color}{_BOLD}{pnl:+,.2f} USDT  ({pnl_pct:+.2f} %){_RESET}")
    print(f"  {_DIM}Max Drawdown :{_RESET}  "
          f"{dd_color}{_BOLD}{dd:.2f} %{_RESET}")
    print(f"  {_DIM}Final balance:{_RESET}  "
          f"{_BOLD}{stats['final_balance']:,.2f} USDT{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print()


def _print_coin_summary(symbol: str, stats: dict) -> None:
    """Print a compact per-coin mini-summary."""
    pnl    = stats["total_pnl_usdt"]
    wr     = stats["win_rate_pct"]
    trades = stats["total_trades"]

    pnl_color = _GREEN if pnl >= 0 else _RED
    wr_color  = _GREEN if wr >= 50 else _RED

    w = 60
    print(f"{_BOLD}{_CYAN}{'─' * w}{_RESET}")
    print(
        f"  {_BOLD}{symbol:<12}{_RESET}"
        f"  Trades: {_BOLD}{trades:>4}{_RESET}"
        f"  Win Rate: {wr_color}{_BOLD}{wr:>6.2f} %{_RESET}"
        f"  PnL: {pnl_color}{_BOLD}{pnl:>+10,.2f} USDT{_RESET}"
    )


def _print_portfolio_report(symbol_stats: list[tuple[str, dict]]) -> None:
    """Print a global aggregated portfolio report across all symbols."""
    total_pnl    = sum(s["total_pnl_usdt"] for _, s in symbol_stats)
    total_trades = sum(s["total_trades"]    for _, s in symbol_stats)
    total_wins   = sum(s["wins"]            for _, s in symbol_stats)
    combined_wr  = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    max_dd       = max(s["max_drawdown_pct"] for _, s in symbol_stats)

    pnl_color = _GREEN if total_pnl >= 0 else _RED
    wr_color  = _GREEN if combined_wr >= 50 else _RED
    dd_color  = _GREEN if max_dd < 10 else (_YELLOW if max_dd < 20 else _RED)

    w = 60
    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'  GLOBAL PORTFOLIO REPORT':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"  {_DIM}Symbols      :{_RESET}  {', '.join(s for s, _ in symbol_stats)}")
    print(f"  {_DIM}Starting cap :{_RESET}  {_STARTING_CAPITAL:,.2f} USDT  (shared / simulated)")
    print(f"{_CYAN}{'─' * w}{_RESET}")
    print(f"  {_DIM}Total trades :{_RESET}  {_BOLD}{total_trades}{_RESET}")
    print(f"  {_DIM}Combined WR  :{_RESET}  "
          f"{wr_color}{_BOLD}{combined_wr:.2f} %{_RESET}")
    print(f"  {_DIM}Total PnL    :{_RESET}  "
          f"{pnl_color}{_BOLD}{total_pnl:+,.2f} USDT{_RESET}")
    print(f"  {_DIM}Max Port. DD :{_RESET}  "
          f"{dd_color}{_BOLD}{max_dd:.2f} %{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """End-to-end backtest: download → train → simulate → report (all symbols)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    exchange = ccxt.binance({"enableRateLimit": True})

    symbol_stats: list[tuple[str, dict]] = []

    w = 60
    print()
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'  MULTI-SYMBOL BACKTEST':^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'  Symbols: ' + ', '.join(_SYMBOLS):^{w}}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═' * w}{_RESET}")

    for symbol in _SYMBOLS:
        logger.info("=" * 60)
        logger.info("Starting backtest for %s", symbol)
        logger.info("=" * 60)

        # 1. Download 6 months of OHLCV data for all three timeframes
        df_15m = _fetch_ohlcv(exchange, symbol, "15m")
        df_1h  = _fetch_ohlcv(exchange, symbol, "1h")
        df_4h  = _fetch_ohlcv(exchange, symbol, "4h")

        # 2. Chronological 70/30 train/test split on the 15m series
        split_idx  = int(len(df_15m) * _TRAIN_RATIO)
        train_15m  = df_15m.iloc[:split_idx]
        test_count = len(df_15m) - split_idx

        logger.info(
            "Data split: train=%d  test=%d (15m candles)",
            len(train_15m),
            test_count,
        )

        # 3. Train the XGBoost model on the training portion
        predictor = MLPredictor()
        ok = predictor.warm_start(
            prices=train_15m["close"].tolist(),
            highs=train_15m["high"].tolist(),
            lows=train_15m["low"].tolist(),
        )
        if not ok:
            logger.error("MLPredictor training failed for %s – insufficient data.", symbol)
            continue

        logger.info("MLPredictor trained for %s.  Starting simulation …", symbol)

        # 4. Run the portfolio simulation on the test portion
        stats = _simulate(
            df_15m,
            df_1h,
            df_4h,
            predictor,
            split_idx=split_idx,
        )

        # 5. Print the colorized per-symbol performance report
        _print_report(stats, train_rows=len(train_15m), test_rows=test_count, symbol=symbol)

        # 6. Print per-coin mini-summary
        _print_coin_summary(symbol, stats)

        symbol_stats.append((symbol, stats))

    # 7. Print global portfolio report
    if symbol_stats:
        _print_portfolio_report(symbol_stats)


if __name__ == "__main__":
    main()
