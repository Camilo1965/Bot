#!/usr/bin/env python3
"""Deep strategy audit: high-fidelity DOGE/USDT backtest aligned with risk_manager + MT5 execution."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
import numpy as np
import pandas as pd
from rich.box import ASCII
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from strategy.quant_features import (  # noqa: E402
    MIN_OHLC_ROWS,
    QUANT_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("deep_strategy_audit")

# --- Constants (match risk_manager / execution) ---
SYMBOL = "DOGE/USDT"
RISK_PER_TRADE = 0.01
LEVERAGE = 5
MAX_POSITIONS = 2
INITIAL_SL = 0.025
ACTIVATION_PCT = 0.02
TRAILING_DISTANCE = 0.02
TTL_HOURS = 12.0
FEE_RATE = 0.0005  # per side (0.05%); round-trip = 2 * FEE_RATE
BUY_THRESHOLD = 0.70
_TRAIN_RATIO = 0.80
_LABEL_HORIZON = 5
_VOLUME_STEP = 0.01
_FALLBACK_CONTRACT_SIZE = 1.0


def _timeframe_to_delta(tf: str) -> timedelta:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return timedelta(days=int(tf[:-1]))
    raise ValueError(f"unsupported timeframe: {tf}")


def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    import ccxt  # noqa: PLC0415

    ex = ccxt.binance(
        {
            "enableRateLimit": True,
            "timeout": 120_000,
            "options": {"defaultType": "spot", "recvWindow": 10_000},
        }
    )
    ex.load_markets()
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    target_start = ex.milliseconds() - int(limit * tf_ms)
    since = target_start
    all_rows: list[list[float]] = []
    remaining = limit
    while remaining > 0:
        batch_limit = min(1000, remaining)
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
        if not batch:
            break
        all_rows.extend(batch)
        since = int(batch[-1][0]) + 1
        remaining = limit - len(all_rows)
        if len(batch) < batch_limit:
            break
    if len(all_rows) > limit:
        all_rows = all_rows[-limit:]
    if len(all_rows) < limit:
        logger.warning("got %d candles (requested %d)", len(all_rows), limit)
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def build_dataset(df: pd.DataFrame, horizon: int, round_trip_label_cost: float) -> pd.DataFrame:
    feat = add_quant_features(df)
    feat["timestamp"] = pd.to_datetime(df["timestamp"].values, utc=True)
    feat["label"] = forward_return_label(feat["close"], horizon, round_trip_label_cost)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"]).reset_index(drop=True)
    return feat


def calculate_lot_size_sim(
    account_balance: float,
    entry_price: float,
    sl_pct: float,
    risk_pct: float,
    contract_size: float,
    volume_step: float,
) -> float:
    """Mirror execution.mt5_executor.calculate_lot_size without MT5 (Decimal floor)."""
    if entry_price <= 0 or account_balance <= 0 or contract_size <= 0:
        return volume_step
    sl_distance_price = entry_price * sl_pct
    risk_amount = account_balance * risk_pct
    lots_raw = risk_amount / (sl_distance_price * contract_size)
    lots_decimal = Decimal(str(lots_raw))
    step_d = Decimal(str(volume_step))
    lots_quotient = (lots_decimal / step_d).quantize(Decimal(1), rounding=ROUND_DOWN)
    lots_stepped = float(lots_quotient * step_d)
    return max(volume_step, lots_stepped)


def cap_position_quote(
    lots: float,
    contract_size: float,
    entry_price: float,
    balance: float,
    risk_per_trade: float,
) -> tuple[float, float]:
    """Apply RiskManager-style notional cap: min(leverage*risk*balance, balance/max_positions)."""
    raw_quote = lots * contract_size * entry_price
    cap_lev = balance * risk_per_trade * LEVERAGE
    cap_share = balance / MAX_POSITIONS
    cap = min(cap_lev, cap_share)
    if raw_quote <= cap:
        return raw_quote, lots
    scale = cap / raw_quote if raw_quote > 0 else 0.0
    new_lots = math.floor((lots * scale) / _VOLUME_STEP) * _VOLUME_STEP
    new_lots = max(_VOLUME_STEP, new_lots)
    return new_lots * contract_size * entry_price, new_lots


@dataclass
class SimPosition:
    trade_id: int
    entry_bar: int
    entry_price: float
    entry_ts: pd.Timestamp
    position_quote: float
    lots: float
    contract_size: float
    sl_price: float
    activation_price: float
    peak_price: float
    trailing_active: bool = False
    ml_prob: float = 0.0


@dataclass
class ClosedTrade:
    entry_bar: int
    exit_bar: int
    gross_pnl: float
    net_pnl: float
    fees: float
    reason: str
    ml_prob: float
    entry_price: float
    exit_price: float


def process_long_bar(
    pos: SimPosition,
    o: float,
    h: float,
    l: float,
    c: float,
    bar_open_ts: pd.Timestamp,
    bar_duration: timedelta,
) -> tuple[str | None, float | None]:
    """Single-bar OHLC path (long): pessimistic SL vs peak/trailing; TTL at bar end."""
    bar_end = bar_open_ts + bar_duration
    if (bar_end - pos.entry_ts).total_seconds() >= TTL_HOURS * 3600.0:
        return "ttl_expiry", float(c)

    if not pos.trailing_active:
        if l <= pos.sl_price:
            return "stop_loss", float(pos.sl_price)
    else:
        if l <= pos.sl_price:
            return "trailing_stop", float(pos.sl_price)

    if h > pos.peak_price:
        pos.peak_price = h
    if not pos.trailing_active and h >= pos.activation_price:
        pos.trailing_active = True
    if pos.trailing_active:
        new_sl = pos.peak_price * (1.0 - TRAILING_DISTANCE)
        if new_sl > pos.sl_price:
            pos.sl_price = new_sl

    if l <= pos.sl_price:
        return ("trailing_stop" if pos.trailing_active else "stop_loss", float(pos.sl_price))

    return None, None


def train_ts_cv(train_df: pd.DataFrame, n_splits: int = 5) -> XGBClassifier:
    X = train_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int32)
    if len(X) < MIN_OHLC_ROWS + 50:
        raise ValueError(f"too few train rows: {len(X)}")
    params = dict(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    tsc = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 500)))
    scores: list[float] = []
    for tr_idx, va_idx in tsc.split(X):
        if len(tr_idx) < 200:
            continue
        m = XGBClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[va_idx])
        scores.append(float((pred == y[va_idx]).mean()))
    if scores:
        logger.info("TS-CV acc folds: %s (mean=%.4f)", ",".join(f"{s:.4f}" for s in scores), float(np.mean(scores)))
    final = XGBClassifier(**params)
    final.fit(X, y)
    return final


def run_backtest(
    feat_df: pd.DataFrame,
    model: XGBClassifier,
    test_start_idx: int,
    contract_size: float,
    bar_delta: timedelta,
    start_balance: float,
    buy_threshold: float,
    risk_per_trade: float,
) -> tuple[list[ClosedTrade], list[float]]:
    """Walk-forward on rows test_start_idx .. end; signal at bar k -> fill open[k+1]."""
    timestamps = feat_df["timestamp"].values
    opens = feat_df["open"].astype(float).values
    highs = feat_df["high"].astype(float).values
    lows = feat_df["low"].astype(float).values
    closes = feat_df["close"].astype(float).values

    X_all = feat_df[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    probas = model.predict_proba(X_all)[:, 1]

    balance = start_balance
    positions: list[SimPosition] = []
    closed: list[ClosedTrade] = []
    equity_curve: list[float] = []
    pending_q: deque[tuple[int, float]] = deque()
    next_trade_id = 0
    n = len(feat_df)

    def try_fill_pending_bar(k: int) -> None:
        nonlocal balance, next_trade_id
        while pending_q and pending_q[0][0] < k:
            pending_q.popleft()
        while pending_q and pending_q[0][0] == k and len(positions) < MAX_POSITIONS:
            prob_entry = pending_q[0][1]
            entry_px = float(opens[k])
            ts_k = pd.Timestamp(timestamps[k])
            lots = calculate_lot_size_sim(balance, entry_px, INITIAL_SL, risk_per_trade, contract_size, _VOLUME_STEP)
            pos_quote, lots_eff = cap_position_quote(lots, contract_size, entry_px, balance, risk_per_trade)
            fee_open = pos_quote * FEE_RATE
            if pos_quote <= 0 or balance < pos_quote + fee_open:
                break
            balance -= pos_quote + fee_open
            sl_p = entry_px * (1.0 - INITIAL_SL)
            act_p = entry_px * (1.0 + ACTIVATION_PCT)
            positions.append(
                SimPosition(
                    trade_id=next_trade_id,
                    entry_bar=k,
                    entry_price=entry_px,
                    entry_ts=ts_k,
                    position_quote=pos_quote,
                    lots=lots_eff,
                    contract_size=contract_size,
                    sl_price=sl_p,
                    activation_price=act_p,
                    peak_price=entry_px,
                    ml_prob=prob_entry,
                )
            )
            next_trade_id += 1
            pending_q.popleft()

    for k in range(test_start_idx, n):
        try_fill_pending_bar(k)

        ts_k = pd.Timestamp(timestamps[k])
        o_k, h_k, l_k, c_k = float(opens[k]), float(highs[k]), float(lows[k]), float(closes[k])

        still_open: list[SimPosition] = []
        for pos in positions:
            reason, exit_px = process_long_bar(pos, o_k, h_k, l_k, c_k, ts_k, bar_delta)
            if reason is None or exit_px is None:
                still_open.append(pos)
                continue
            gross = pos.position_quote * (exit_px / pos.entry_price - 1.0)
            exit_notional = pos.lots * pos.contract_size * exit_px
            fee_close = exit_notional * FEE_RATE
            fee_open_alloc = pos.position_quote * FEE_RATE
            net = gross - fee_open_alloc - fee_close
            balance += pos.position_quote + gross - fee_close
            closed.append(
                ClosedTrade(
                    entry_bar=pos.entry_bar,
                    exit_bar=k,
                    gross_pnl=gross,
                    net_pnl=net,
                    fees=fee_open_alloc + fee_close,
                    reason=reason,
                    ml_prob=pos.ml_prob,
                    entry_price=pos.entry_price,
                    exit_price=exit_px,
                )
            )
            equity_curve.append(balance)
        positions = still_open

        if k < n - 1:
            p = float(probas[k])
            if p >= buy_threshold and len(positions) < MAX_POSITIONS:
                pending_q.append((k + 1, p))

    last_k = n - 1
    for pos in list(positions):
        exit_px = float(closes[last_k])
        gross = pos.position_quote * (exit_px / pos.entry_price - 1.0)
        exit_notional = pos.lots * pos.contract_size * exit_px
        fee_close = exit_notional * FEE_RATE
        fee_open_alloc = pos.position_quote * FEE_RATE
        net = gross - fee_open_alloc - fee_close
        balance += pos.position_quote + gross - fee_close
        closed.append(
            ClosedTrade(
                entry_bar=pos.entry_bar,
                exit_bar=last_k,
                gross_pnl=gross,
                net_pnl=net,
                fees=fee_open_alloc + fee_close,
                reason="eod_flat",
                ml_prob=pos.ml_prob,
                entry_price=pos.entry_price,
                exit_price=exit_px,
            )
        )
        equity_curve.append(balance)
        positions.remove(pos)

    return closed, equity_curve


def profit_factor(trades: list[ClosedTrade]) -> float:
    wins = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    losses = sum(-t.net_pnl for t in trades if t.net_pnl < 0)
    if losses < 1e-12:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def max_drawdown(equity_curve: list[float], initial: float) -> float:
    if not equity_curve:
        return 0.0
    peak = initial
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep DOGE/USDT strategy audit (execution-aligned).")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--contract-size", type=float, default=_FALLBACK_CONTRACT_SIZE)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=_TRAIN_RATIO)
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=BUY_THRESHOLD,
        help="min ML proba to enter (default: 0.70)",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=RISK_PER_TRADE,
        help="fraction of equity risked per trade for sizing (default: 0.01)",
    )
    args = parser.parse_args()

    bar_delta = _timeframe_to_delta(args.timeframe)

    round_trip_label = 2.0 * FEE_RATE
    console = Console(legacy_windows=False)

    console.print(
        Panel.fit(
            f"[bold]Deep strategy audit[/] DOGE/USDT | prob>={args.buy_threshold:.2f} risk={args.risk_per_trade:.4f}",
            box=ASCII,
        )
    )

    logger.info("Fetching %d %s candles for %s ...", args.limit, args.timeframe, args.symbol)
    raw = fetch_ohlcv_ccxt(args.symbol, args.timeframe, args.limit)
    full = build_dataset(raw, _LABEL_HORIZON, round_trip_label)

    split_i = int(len(full) * args.train_ratio)
    split_i = max(split_i, MIN_OHLC_ROWS + 100)
    split_i = min(split_i, len(full) - 200)
    train_df = full.iloc[:split_i].copy()

    logger.info("Hold-out: train_rows=%d test_rows=%d (no overlap)", len(train_df), len(full) - split_i)

    model = train_ts_cv(train_df, n_splits=args.cv_splits)

    feat_full = full.reset_index(drop=True)
    closed, equity_curve = run_backtest(
        feat_full,
        model,
        split_i,
        args.contract_size,
        bar_delta,
        args.balance,
        args.buy_threshold,
        args.risk_per_trade,
    )

    nets = np.array([t.net_pnl for t in closed], dtype=float)
    ttl_trades = [t for t in closed if t.reason == "ttl_expiry"]
    high_prob_trades = [t for t in closed if t.ml_prob >= args.buy_threshold]

    pf = profit_factor(closed)
    wr = float((nets > 0).mean()) if len(nets) else 0.0
    wr_high = (
        float(sum(1 for t in high_prob_trades if t.net_pnl > 0) / len(high_prob_trades))
        if high_prob_trades
        else 0.0
    )
    expectancy = float(nets.mean()) if len(nets) else float("nan")
    total_net = float(nets.sum()) if len(nets) else 0.0
    net_return_pct = (total_net / args.balance) * 100.0 if args.balance > 0 else 0.0
    final_balance = float(equity_curve[-1]) if equity_curve else args.balance
    equity_series = [args.balance] + equity_curve
    mdd = max_drawdown(equity_series, args.balance)
    ttl_avg = float(np.mean([t.net_pnl for t in ttl_trades])) if ttl_trades else float("nan")

    table = Table(title="Results (test segment, fees included)", box=ASCII)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Profit factor (net)", f"{pf:.4f}" if math.isfinite(pf) else "inf/nan")
    table.add_row("Win rate (all closed trades)", f"{wr * 100:.2f}%")
    table.add_row(f"Win rate (ML prob >= {args.buy_threshold:.2f})", f"{wr_high * 100:.2f}%")
    table.add_row("Max drawdown (equity)", f"{mdd * 100:.2f}%")
    table.add_row("Expectancy / trade (net USDT)", f"{expectancy:.4f}" if len(nets) else "n/a")
    table.add_row("Total net PnL (USDT)", f"{total_net:.2f}")
    table.add_row("Net return on start balance", f"{net_return_pct:.2f}%")
    table.add_row("Final balance (USDT)", f"{final_balance:.2f}")
    table.add_row("Closed trades (frequency)", str(len(closed)))
    table.add_row("TTL closes (count)", str(len(ttl_trades)))
    table.add_row("TTL avg net PnL", f"{ttl_avg:.4f}" if ttl_trades else "n/a")

    console.print(table)

    risk_tbl = Table(title="Risk / execution model (this run)", box=ASCII)
    risk_tbl.add_column("Parameter")
    risk_tbl.add_column("Value", justify="right")
    risk_tbl.add_row("BUY_THRESHOLD (ML)", str(args.buy_threshold))
    risk_tbl.add_row("RISK_PER_TRADE", str(args.risk_per_trade))
    risk_tbl.add_row("LEVERAGE (cap)", str(LEVERAGE))
    risk_tbl.add_row("MAX_POSITIONS", str(MAX_POSITIONS))
    risk_tbl.add_row("INITIAL_SL", str(INITIAL_SL))
    risk_tbl.add_row("TRAILING activation / distance", f"{ACTIVATION_PCT} / {TRAILING_DISTANCE}")
    risk_tbl.add_row("TTL (hours)", str(TTL_HOURS))
    risk_tbl.add_row("Fee per side", str(FEE_RATE))
    risk_tbl.add_row("contract_size (sim)", str(args.contract_size))
    console.print(risk_tbl)

    console.print(
        f"\n[dim]Train held-out from test temporally. Label horizon={_LABEL_HORIZON} bars. "
        f"Entry at open[t+1] after signal at bar t close.[/dim]\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
