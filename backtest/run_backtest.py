"""
backtest.run_backtest
~~~~~~~~~~~~~~~~~~~~~

Downloads 1 year of 15-minute BTC/USDT OHLCV data from Binance via ccxt,
engineers the same features used by MLPredictor, trains an XGBoost model
on the first 70 % of data, evaluates on the remaining 30 %, prints
backtest performance metrics (Total Return, Win Rate), and saves the
trained model to ``models/xgb_live.json`` so that ``main.py`` can load
it at startup instead of training from scratch.

Usage
-----
    python -m backtest.run_backtest
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must mirror strategy/ml_predictor.py)
# ---------------------------------------------------------------------------

_SYMBOL = "BTC/USDT"
_TIMEFRAME = "15m"
_FETCH_LIMIT = 1000         # ccxt max per request
_PREDICTION_HORIZON = 5     # ticks ahead used as the label
_SMA_PERIOD = 20
_RSI_PERIOD = 14
_MOMENTUM_PERIOD = 5
_ADX_PERIOD = 14
_ATR_PERIOD = 14
_TRAIN_RATIO = 0.70
_FEATURE_COLS = [
    "sentiment",
    "rsi",
    "sma_ratio",
    "volatility",
    "momentum",
    "obi_ratio",
    "adx",
    "atr",
]

# Path where the trained model is saved
_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "xgb_live.json"

# Simulated trade parameters (must mirror risk/risk_manager.py)
_ACTIVATION_PCT = 0.015    # 1.5 % profit activates trailing stop
_TRAILING_DISTANCE = 0.005 # 0.5 % trailing gap below running peak
_INITIAL_SL_PCT = 0.0075   # 0.75 % initial hard stop loss

# Maximum number of candles to hold a simulated position before closing at market
_MAX_HOLDING_PERIOD = 50


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str = _SYMBOL, timeframe: str = _TIMEFRAME) -> pd.DataFrame:
    """Download ~1 year of 15-minute OHLCV data from Binance.

    Returns a DataFrame with columns:
        timestamp (UTC), open, high, low, close, volume
    sorted in ascending chronological order.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    one_year_ms = 365 * 24 * 60 * 60 * 1000
    since = exchange.milliseconds() - one_year_ms

    all_candles: list[list] = []
    logger.info("Fetching %s %s OHLCV data from Binance…", symbol, timeframe)

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=_FETCH_LIMIT)
        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if last_ts >= exchange.milliseconds() - exchange.parse_timeframe(timeframe) * 1000:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info("Downloaded %d candles.", len(df))
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV data.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns ``timestamp``, ``open``, ``high``,
        ``low``, ``close``, and ``volume``.  The ``high`` and ``low``
        columns are required both for indicator calculation (ATR, ADX) and
        to support intra-candle trailing stop simulation.

    Adds the following columns (matching _FEATURE_COLS):
        sentiment  – always 0.0 (no live sentiment in historical data)
        rsi        – RSI-14 using Wilder's exponential smoothing
        sma_ratio  – close / SMA-20 (normalised price level)
        volatility – rolling 20-period standard deviation of close
        momentum   – percentage price change over 5 periods
        obi_ratio  – always 1.0 (no live order-book in historical data)
        adx        – Average Directional Index (14-period)
        atr        – Average True Range (14-period)
        label      – 1 if close rises after _PREDICTION_HORIZON ticks, else 0

    Also passes through ``high`` and ``low`` columns for the trailing stop
    simulation in :func:`simulate_backtest`.

    Rows without a complete window or future label are dropped.
    """
    series = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    # SMA-20 ratio
    sma_20 = series.rolling(_SMA_PERIOD).mean()
    sma_ratio = (series / sma_20).where(sma_20 != 0)

    # Volatility
    volatility = series.rolling(_SMA_PERIOD).std()

    # RSI-14 (Wilder's EMA)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=_RSI_PERIOD - 1, min_periods=_RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=_RSI_PERIOD - 1, min_periods=_RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).fillna(50.0)

    # Momentum
    momentum = series.pct_change(_MOMENTUM_PERIOD) * 100

    # ATR (Wilder's EMA of True Range)
    prev_c = series.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(com=_ATR_PERIOD - 1, min_periods=_ATR_PERIOD).mean()

    # ADX (Average Directional Index)
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smth_atr = tr.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean()
    smth_pdm = plus_dm.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean()
    smth_mdm = minus_dm.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean()

    safe_atr = smth_atr.replace(0.0, np.nan)
    plus_di = 100 * smth_pdm / safe_atr
    minus_di = 100 * smth_mdm / safe_atr
    di_sum = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(com=_ADX_PERIOD - 1, min_periods=_ADX_PERIOD).mean().fillna(25.0)  # neutral default

    # Label: 1 if price is higher after horizon steps
    label = (series.shift(-_PREDICTION_HORIZON) > series).astype(float)

    feat_df = df[["timestamp", "close", "high", "low"]].copy()
    feat_df["sentiment"] = 0.0
    feat_df["rsi"] = rsi
    feat_df["sma_ratio"] = sma_ratio
    feat_df["volatility"] = volatility
    feat_df["momentum"] = momentum
    feat_df["obi_ratio"] = 1.0   # no live order-book available in backtest
    feat_df["adx"] = adx
    feat_df["atr"] = atr
    feat_df["label"] = label

    # Fill NaN from indicator warm-up with 0
    feat_df[["sma_ratio", "volatility", "momentum"]] = feat_df[
        ["sma_ratio", "volatility", "momentum"]
    ].fillna(0.0)

    # Drop rows missing a future label (last _PREDICTION_HORIZON rows)
    feat_df = feat_df.dropna(subset=["label"]).reset_index(drop=True)
    return feat_df


# ---------------------------------------------------------------------------
# Backtest simulation
# ---------------------------------------------------------------------------

def simulate_backtest(
    feat_df: pd.DataFrame,
    model: XGBClassifier,
    buy_threshold: float = 0.62,
) -> dict[str, float]:
    """Simulate a simple long-only strategy on the test set using a trailing stop.

    A BUY signal is generated when the model's upward-probability exceeds
    ``buy_threshold``.  Each simulated trade is protected by an initial hard
    stop loss (``_INITIAL_SL_PCT``) and, once the position gains
    ``_ACTIVATION_PCT``, a trailing stop that follows the running price peak
    at a distance of ``_TRAILING_DISTANCE``.

    The intra-candle high is used to update the highest price seen and raise
    the trailing SL; the intra-candle low is used to detect stop-loss hits.

    Parameters
    ----------
    feat_df:        Feature DataFrame (test split) with ``close``, ``high``,
                    and ``low`` columns.
    model:          Trained XGBClassifier.
    buy_threshold:  Minimum predicted probability to open a long trade.

    Returns
    -------
    dict with keys ``total_return_pct`` and ``win_rate_pct``.
    """
    X = feat_df[_FEATURE_COLS]
    probas = model.predict_proba(X)[:, 1]

    closes = feat_df["close"].values
    highs  = feat_df["high"].values
    lows   = feat_df["low"].values
    total_return = 0.0
    wins = 0
    trades = 0

    i = 0
    while i < len(feat_df):
        if probas[i] > buy_threshold:
            entry        = closes[i]
            active_sl    = entry * (1 - _INITIAL_SL_PCT)
            highest_seen = entry
            exited = False
            # Scan subsequent candles for exit
            for j in range(i + 1, min(i + _MAX_HOLDING_PERIOD, len(feat_df))):
                high_j = highs[j]
                low_j  = lows[j]

                # Track intra-candle high for trailing stop calculation
                if high_j > highest_seen:
                    highest_seen = high_j

                # Update trailing SL once activation threshold is reached
                if (highest_seen - entry) / entry >= _ACTIVATION_PCT:
                    tsl = highest_seen * (1 - _TRAILING_DISTANCE)
                    if tsl > active_sl:
                        active_sl = tsl

                # Check if the candle's low hits the active stop loss
                if low_j <= active_sl:
                    exit_price = active_sl
                    pct = (exit_price - entry) / entry * 100
                    total_return += pct
                    if pct > 0:
                        wins += 1
                    trades += 1
                    i = j
                    exited = True
                    break
            if not exited:
                # Close at market price after max holding period
                end_idx = min(i + _MAX_HOLDING_PERIOD, len(feat_df) - 1)
                pct = (closes[end_idx] - entry) / entry * 100
                total_return += pct
                if pct > 0:
                    wins += 1
                trades += 1
                i = end_idx
            continue
        i += 1

    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    return {
        "total_return_pct": total_return,
        "win_rate_pct": win_rate,
        "total_trades": trades,
        "winning_trades": wins,
    }


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: XGBClassifier, path: Path = _MODEL_PATH) -> None:
    """Save *model* to *path* in XGBoost's native JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """End-to-end backtest: download → features → train → evaluate → save."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1. Download data
    df = fetch_ohlcv()

    # 2. Build features & labels
    feat_df = build_features(df)
    logger.info("Feature matrix shape: %s", feat_df.shape)

    # 3. Train / test split (70 / 30, chronological)
    split_idx = int(len(feat_df) * _TRAIN_RATIO)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:].reset_index(drop=True)

    X_train = train_df[_FEATURE_COLS]
    y_train = train_df["label"].astype(int)
    X_test = test_df[_FEATURE_COLS]
    y_test = test_df["label"].astype(int)

    logger.info(
        "Train samples: %d  |  Test samples: %d",
        len(X_train),
        len(X_test),
    )

    # 4. Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test.values))

    backtest_stats = simulate_backtest(test_df, model)

    print("\n" + "=" * 55)
    print("          BACKTEST PERFORMANCE REPORT")
    print("=" * 55)
    print(f"  Symbol        : {_SYMBOL}  ({_TIMEFRAME} candles)")
    print(f"  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")
    print(f"  Label accuracy: {accuracy * 100:.2f} %")
    print("-" * 55)
    print(f"  Total trades  : {backtest_stats['total_trades']}")
    print(f"  Winning trades: {backtest_stats['winning_trades']}")
    print(f"  Win Rate      : {backtest_stats['win_rate_pct']:.2f} %")
    print(f"  Total Return  : {backtest_stats['total_return_pct']:.2f} %")
    print("=" * 55 + "\n")

    # 6. Save the trained model
    save_model(model)


if __name__ == "__main__":
    run()
