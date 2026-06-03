#!/usr/bin/env python3
"""
scripts/backtest_full_bot.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backtesting COMPLETO del bot ClawdBot v2 con:
- Regime prediction (TRENDING vs RANGING)
- XGBoost direction prediction
- Risk manager (sizing, max positions, daily loss limit)
- Execution (SL, TP ATR-based, trailing stop, TTL)
- Vibe gating (simplificado)

Out-of-sample: entrena en 70%, backtestea en 30%.
Compara: bot completo vs sin regime vs buy-and-hold.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features, forward_return_label, DEFAULT_LABEL_ROUND_TRIP

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backtest_full_bot")

# ── Configuración del bot ────────────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0
MAX_POSITIONS = 2
RISK_PER_TRADE = 0.015      # match production
LEVERAGE = 5
TTL_HOURS = 5.0             # match horizon=16-20 bars (4-5h)
TP_ATR_MULT = 2.5
ACTIVATION_PCT = 0.05       # activate trailing only ABOVE TP=4-6%
TRAILING_DISTANCE = 0.02
DAILY_LOSS_LIMIT = 0.03
# Module-level overrides for param sweep (monkey-patched in param_sweep_v2.py)
REGIME_BUY_THRESHOLD = 0.65  # entry only if regime_prob >= this
MAX_ATR_PCT = None           # if set, skip entry when atr_pct > this (vol filter)
USE_KELLY = False            # if True, scale risk by fractional Kelly (1/4 cap)
KELLY_FRACTION = 0.25
USE_HTF_FILTER = True        # match production htf_sma200_1h_allows_long (proxy on entry TF)
HTF_SLOW_PERIOD = 100        # SMA period on entry timeframe (proxy for trend)
COOLDOWN_AFTER_LOSSES = 0    # 0 disabled; else pause N bars after 2 consec losses

SYMBOL_CONFIG = {
    # Mirrors strategy/ml_predictor.py:SYMBOL_CONFIG (canonical). Tuned 2026-06-02
    # via disk-loaded backtest + post-retrain calibration refit + per-symbol param sweep.
    "BTC/USDT":  {"prob_threshold": 0.70, "fixed_sl_pct": 0.020, "fixed_tp_pct": 0.040, "risk": 0.015, "timeframe": "30m", "horizon": 36, "regime_adx": 25.0, "max_spw": None, "skip_regime": True, "exec_costs": {"spread_bps": 2.0, "slippage_atr_mult": 0.05}},
    "ETH/USDT":  {"prob_threshold": 0.50, "fixed_sl_pct": 0.020, "fixed_tp_pct": 0.030, "risk": 0.015, "timeframe": "15m", "horizon": 20, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 2.0, "slippage_atr_mult": 0.05}},
    "XRP/USDT":  {"prob_threshold": 0.95, "fixed_sl_pct": 0.025, "fixed_tp_pct": 0.040, "risk": 0.005, "timeframe": "15m", "horizon": 16, "regime_adx": 20.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 2.0, "slippage_atr_mult": 0.05}},  # DEMOTED
    "SOL/USDT":  {"prob_threshold": 0.55, "fixed_sl_pct": 0.025, "fixed_tp_pct": 0.035, "risk": 0.015, "timeframe": "30m", "horizon": 24, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 5.0, "slippage_atr_mult": 0.10}},
    "DOGE/USDT": {"prob_threshold": 0.55, "fixed_sl_pct": 0.025, "fixed_tp_pct": 0.045, "risk": 0.010, "timeframe": "15m", "horizon": 16, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 5.0, "slippage_atr_mult": 0.10}},
    # Phase D survivors (2026-06-02, inline 70/30 OOS)
    "NEAR/USDT": {"prob_threshold": 0.55, "fixed_sl_pct": 0.030, "fixed_tp_pct": 0.050, "risk": 0.010, "timeframe": "15m", "horizon": 20, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 8.0, "slippage_atr_mult": 0.15}},
    "ATOM/USDT": {"prob_threshold": 0.45, "fixed_sl_pct": 0.025, "fixed_tp_pct": 0.035, "risk": 0.010, "timeframe": "15m", "horizon": 20, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 8.0, "slippage_atr_mult": 0.15}},
    "LINK/USDT": {"prob_threshold": 0.55, "fixed_sl_pct": 0.030, "fixed_tp_pct": 0.050, "risk": 0.008, "timeframe": "15m", "horizon": 20, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 5.0, "slippage_atr_mult": 0.10}},
    # Phase D+ survivor (scan 2026-06-03): WR 60.6%, PnL +15.84%, score 32.84. Needs retrain before live.
    "JTO/USDT":  {"prob_threshold": 0.50, "fixed_sl_pct": 0.025, "fixed_tp_pct": 0.040, "risk": 0.010, "timeframe": "15m", "horizon": 20, "regime_adx": 25.0, "max_spw": 8.0,   "skip_regime": True, "exec_costs": {"spread_bps": 8.0, "slippage_atr_mult": 0.15}},
}


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float  # notional en USD
    sl_price: float
    tp_price: float | None
    peak_price: float
    current_stop: float
    ml_confidence: float
    regime_prob: float
    ttl_hours: float = TTL_HOURS
    close_reason: str | None = None
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestState:
    balance: float = INITIAL_BALANCE
    equity: float = INITIAL_BALANCE
    total_pnl: float = 0.0
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    closed_trades: list[OpenPosition] = field(default_factory=list)
    daily_loss: float = 0.0
    current_day: pd.Timestamp | None = None
    max_drawdown: float = 0.0
    peak_equity: float = INITIAL_BALANCE
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    gross_wins: float = 0.0
    gross_losses: float = 0.0


# ── Modelos ──────────────────────────────────────────────────────────────────

def train_regime_model(df_train: pd.DataFrame, look_ahead: int, adx_threshold: float = 25.0) -> XGBClassifier:
    margin = max(4.0, adx_threshold * 0.20)
    feat = add_quant_features(df_train.copy(), base_timeframe_min=15)
    feat["future_adx"] = feat["adx"].shift(-look_ahead)
    feat["label"] = (feat["future_adx"] > adx_threshold).astype(int)
    feat = feat[(feat["adx"] > adx_threshold + margin) | (feat["adx"] < adx_threshold - margin)].copy()
    feat = feat.dropna(subset=list(QUANT_FEATURE_COLS) + ["label"])
    if len(feat) < 100:
        # Fallback: neutral model
        return XGBClassifier(n_estimators=10, max_depth=2)
    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = feat["label"].to_numpy(dtype=np.int32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model


LABEL_THRESHOLD: float = float(os.environ.get("BACKTEST_LABEL_THRESHOLD", "0.015"))  # 1.5% default

def train_direction_model(df_train: pd.DataFrame, horizon: int, max_spw: float | None = None) -> XGBClassifier:
    feat = add_quant_features(df_train.copy(), base_timeframe_min=15)
    # CRITICAL FIX: label must exceed SL to justify the risk
    fwd = feat["close"].shift(-horizon) / feat["close"] - 1.0
    feat["label"] = (fwd > LABEL_THRESHOLD).astype(int)
    feat = feat.dropna(subset=list(QUANT_FEATURE_COLS) + ["label"])
    if len(feat) < 100:
        return XGBClassifier(n_estimators=10, max_depth=2)
    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = feat["label"].to_numpy(dtype=np.int32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    raw_spw = max(1.0, n_neg / n_pos) if n_pos > 0 else 1.0
    spw = min(raw_spw, max_spw) if max_spw is not None else raw_spw
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model


# ── Simulación del bot ───────────────────────────────────────────────────────

def simulate_bot(
    df_test: pd.DataFrame,
    regime_model: XGBClassifier,
    direction_model: XGBClassifier,
    symbol: str,
    use_regime: bool = True,
) -> BacktestState:
    """Simula el bot completo en datos de test (inline-trained probs)."""
    cfg = SYMBOL_CONFIG[symbol]
    tf_min = int(cfg["timeframe"][:-1])
    feat = add_quant_features(df_test.copy(), base_timeframe_min=tf_min)
    feat = feat.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)
    if len(feat) < 10:
        return BacktestState()

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    regime_probs = regime_model.predict_proba(X)[:, 1] if hasattr(regime_model, "predict_proba") else np.full(len(X), 0.5)
    direction_probs = direction_model.predict_proba(X)[:, 1] if hasattr(direction_model, "predict_proba") else np.full(len(X), 0.5)
    return _run_simulation_loop(feat, regime_probs, direction_probs, symbol, use_regime=use_regime)


def _run_simulation_loop(
    feat: pd.DataFrame,
    regime_probs: np.ndarray,
    direction_probs: np.ndarray,
    symbol: str,
    use_regime: bool = True,
) -> BacktestState:
    """Trading-loop core. Accepts pre-computed feature df + prob arrays so disk-loaded
    backtests (calibrated probs from {SYMBOL}_v2.json + {SYMBOL}_calibration.json) and
    inline-trained backtests share the exact same execution logic."""
    state = BacktestState()
    cfg = SYMBOL_CONFIG[symbol]
    sl_frac = cfg["fixed_sl_pct"]
    prob_threshold = cfg["prob_threshold"]
    risk_pct = cfg["risk"]
    tf_min = int(cfg["timeframe"][:-1])
    horizon_bars = int(cfg.get("horizon", 20))
    ttl_hours = max(TTL_HOURS, horizon_bars * tf_min / 60.0)

    # Execution costs (slippage + spread) — see roadmap [I-3].
    # Defaults sized for mid-cap USDT pair on Binance spot.
    _ec = cfg.get("exec_costs", {})
    spread_bps = float(_ec.get("spread_bps", 3.0))
    slip_atr_mult = float(_ec.get("slippage_atr_mult", 0.07))
    spread_frac = spread_bps / 10000.0

    if len(feat) < 10 or len(direction_probs) != len(feat) or len(regime_probs) != len(feat):
        return state

    closes = feat["close"].to_numpy()
    highs = feat["high"].to_numpy()
    lows = feat["low"].to_numpy()
    atrs = feat["atr"].fillna(0.0).to_numpy()
    htf_sma_slow = feat["close"].rolling(HTF_SLOW_PERIOD, min_periods=HTF_SLOW_PERIOD).mean().to_numpy()
    cooldown_until = -1  # bar index; -1 = no cooldown
    last_losses_streak = 0
    timestamps = feat["timestamp"] if "timestamp" in feat.columns else pd.RangeIndex(len(feat))

    for i in range(len(feat)):
        ts = timestamps.iloc[i] if hasattr(timestamps, "iloc") else timestamps[i]
        current_price = float(closes[i])
        current_high = float(highs[i])
        current_low = float(lows[i])
        current_atr = float(atrs[i]) if i < len(atrs) else 0.0

        # ── Daily reset ──
        if state.current_day is None or pd.Timestamp(ts).date() != state.current_day.date():
            state.daily_loss = 0.0
            state.current_day = pd.Timestamp(ts)

        # ── Update open positions ──
        symbols_to_close = []
        for sym, pos in state.open_positions.items():
            # Update peak
            if current_high > pos.peak_price:
                pos.peak_price = current_high

            # Trailing stop update
            if pos.peak_price >= pos.entry_price * (1.0 + ACTIVATION_PCT):
                new_stop = pos.peak_price * (1.0 - TRAILING_DISTANCE)
                if new_stop > pos.current_stop:
                    pos.current_stop = new_stop

            # Check SL / TP / TTL
            hit_sl = current_low <= pos.current_stop
            hit_tp = pos.tp_price is not None and current_high >= pos.tp_price
            hours_held = (pd.Timestamp(ts) - pos.entry_time).total_seconds() / 3600.0
            hit_ttl = hours_held >= ttl_hours

            atr_pct_cur = (current_atr / current_price) if current_price > 0 else 0.0
            if hit_sl:
                # SL fill: adverse slippage + spread (stop sweep takes liquidity)
                exit_p = pos.current_stop * (1.0 - spread_frac - slip_atr_mult * atr_pct_cur * 0.5)
                reason = "SL"
            elif hit_tp:
                # TP fill: better fill in a limit; just pay spread
                exit_p = pos.tp_price * (1.0 - spread_frac)
                reason = "TP"
            elif hit_ttl:
                # TTL: market close
                exit_p = current_price * (1.0 - spread_frac)
                reason = "TTL"
            else:
                continue

            # Calculate PnL
            pnl_pct = (exit_p - pos.entry_price) / pos.entry_price
            pnl_usd = pnl_pct * pos.position_size
            state.total_pnl += pnl_usd
            state.balance += pnl_usd
            state.daily_loss += min(0.0, pnl_usd)
            state.n_trades += 1
            if pnl_usd > 0:
                state.n_wins += 1
                state.gross_wins += pnl_usd
                last_losses_streak = 0
            else:
                state.n_losses += 1
                state.gross_losses += abs(pnl_usd)
                last_losses_streak += 1
                if COOLDOWN_AFTER_LOSSES > 0 and last_losses_streak >= 2:
                    cooldown_until = i + COOLDOWN_AFTER_LOSSES
                    last_losses_streak = 0

            pos.close_reason = reason
            pos.exit_price = exit_p
            pos.exit_time = pd.Timestamp(ts)
            pos.pnl_usd = pnl_usd
            pos.pnl_pct = pnl_pct
            state.closed_trades.append(pos)
            symbols_to_close.append(sym)

            # Commission
            commission = pos.position_size * 0.0004 * 2  # open + close
            state.balance -= commission
            state.total_pnl -= commission

        for sym in symbols_to_close:
            del state.open_positions[sym]

        # ── Update equity & drawdown ──
        unrealized = 0.0
        for pos in state.open_positions.values():
            unrealized += ((current_price - pos.entry_price) / pos.entry_price) * pos.position_size
        state.equity = state.balance + unrealized
        if state.equity > state.peak_equity:
            state.peak_equity = state.equity
        dd = (state.peak_equity - state.equity) / state.peak_equity
        if dd > state.max_drawdown:
            state.max_drawdown = dd

        # ── Daily loss limit ──
        if abs(state.daily_loss) >= state.balance * DAILY_LOSS_LIMIT:
            continue

        # ── Max positions ──
        if len(state.open_positions) >= MAX_POSITIONS:
            continue

        # ── Skip if position already open ──
        if symbol in state.open_positions:
            continue

        # ── Cooldown after losses streak ──
        if cooldown_until > 0 and i < cooldown_until:
            continue

        # ── HTF trend filter (price > SMA slow on entry TF as trend proxy) ──
        if USE_HTF_FILTER and not np.isnan(htf_sma_slow[i]):
            if current_price < float(htf_sma_slow[i]):
                continue

        # ── Regime filter ──
        regime_prob = float(regime_probs[i])
        is_trending = regime_prob >= REGIME_BUY_THRESHOLD
        if use_regime and not is_trending:
            continue

        # ── Volatility filter (skip if ATR_pct exceeds threshold) ──
        if MAX_ATR_PCT is not None and current_price > 0:
            atr_pct_cur = current_atr / current_price
            if atr_pct_cur > MAX_ATR_PCT:
                continue

        # ── Direction filter ──
        direction_prob = float(direction_probs[i])
        if direction_prob < prob_threshold:
            continue

        # ── Risk manager sizing (optional fractional Kelly) ──
        effective_risk = risk_pct
        if USE_KELLY:
            # Need TP fraction for Kelly. Approx with cfg fixed_tp_pct.
            tp_frac_est = cfg.get("fixed_tp_pct", 0.03)
            b = max(0.1, tp_frac_est / max(0.001, sl_frac))
            p = max(0.0, min(1.0, direction_prob))
            kelly_f = max(0.0, (p * b - (1.0 - p)) / b)
            f_safe = min(kelly_f * KELLY_FRACTION, risk_pct)
            if f_safe < 0.001:
                continue
            effective_risk = f_safe
        risk_amount = state.balance * effective_risk
        position_size = risk_amount / max(0.001, sl_frac)
        max_allocation = (state.balance / MAX_POSITIONS) * LEVERAGE
        position_size = min(position_size, max_allocation)
        if position_size <= 0:
            continue
        if position_size > state.balance * LEVERAGE:
            continue

        # ── Open position ──
        # Entry fill: adverse slippage + spread on a market buy
        atr_pct_entry = (current_atr / current_price) if current_price > 0 else 0.0
        entry_fill = current_price * (1.0 + spread_frac + slip_atr_mult * atr_pct_entry)
        sl_price = entry_fill * (1.0 - sl_frac)
        # Match production: TP = max(fixed_tp_pct, ATR_TP_MULT * atr_pct)
        fixed_tp = cfg.get("fixed_tp_pct")
        if fixed_tp is not None and current_atr > 0:
            atr_tp_frac = (current_atr * TP_ATR_MULT) / entry_fill
            tp_frac = max(float(fixed_tp), atr_tp_frac)
            tp_price = entry_fill * (1.0 + tp_frac)
        elif fixed_tp is not None:
            tp_price = entry_fill * (1.0 + float(fixed_tp))
        elif current_atr > 0:
            tp_price = entry_fill + (current_atr * TP_ATR_MULT)
        else:
            tp_price = None

        pos = OpenPosition(
            symbol=symbol,
            entry_time=pd.Timestamp(ts),
            entry_price=entry_fill,
            position_size=position_size,
            sl_price=sl_price,
            tp_price=tp_price,
            peak_price=entry_fill,
            current_stop=sl_price,
            ml_confidence=direction_prob,
            regime_prob=regime_prob,
        )
        state.open_positions[symbol] = pos

        # Commission on open
        commission_open = position_size * 0.0004
        state.balance -= commission_open
        state.total_pnl -= commission_open

    # Close any remaining positions at last price (market close + spread)
    for sym, pos in list(state.open_positions.items()):
        exit_p = float(closes[-1]) * (1.0 - spread_frac)
        pnl_pct = (exit_p - pos.entry_price) / pos.entry_price
        pnl_usd = pnl_pct * pos.position_size
        state.total_pnl += pnl_usd
        state.balance += pnl_usd
        state.n_trades += 1
        if pnl_usd > 0:
            state.n_wins += 1
            state.gross_wins += pnl_usd
        else:
            state.n_losses += 1
            state.gross_losses += abs(pnl_usd)
        pos.close_reason = "END_OF_DATA"
        pos.exit_price = exit_p
        pos.exit_time = pd.Timestamp(timestamps.iloc[-1] if hasattr(timestamps, "iloc") else timestamps[-1])
        pos.pnl_usd = pnl_usd
        pos.pnl_pct = pnl_pct
        state.closed_trades.append(pos)
        commission = pos.position_size * 0.0004
        state.balance -= commission
        state.total_pnl -= commission
    state.open_positions.clear()
    state.equity = state.balance

    return state


def calc_metrics(state: BacktestState, days: float) -> dict[str, float]:
    if state.n_trades == 0:
        return {"trades": 0, "win_rate": 0.0, "pnl_pct": 0.0}
    win_rate = state.n_wins / state.n_trades
    pnl_pct = (state.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    avg_win = state.gross_wins / state.n_wins if state.n_wins > 0 else 0
    avg_loss = state.gross_losses / state.n_losses if state.n_losses > 0 else 1
    profit_factor = state.gross_wins / state.gross_losses if state.gross_losses > 0 else float("inf")
    # Sharpe simplificado
    returns = [t.pnl_pct for t in state.closed_trades]
    sharpe = 0.0
    if len(returns) > 1:
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r > 0:
            sharpe = (mean_r / std_r) * np.sqrt(365 / max(days, 1))
    return {
        "trades": state.n_trades,
        "win_rate": round(win_rate, 4),
        "pnl_pct": round(pnl_pct, 2),
        "total_pnl_usd": round(state.total_pnl, 2),
        "final_balance": round(state.balance, 2),
        "max_drawdown_pct": round(state.max_drawdown * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "trades_per_week": round(state.n_trades / max(days / 7, 1), 1),
    }


def run_backtest(symbol: str, limit: int = 30_000) -> dict[str, Any]:
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    hor = cfg["horizon"]
    tf_min = int(tf[:-1])

    logger.info("Fetching %d %s candles for %s...", limit, tf, symbol)
    raw = fetch_ohlcv_ccxt(symbol, timeframe=tf, limit=limit)
    if len(raw) < 1000:
        return {"error": "insufficient data"}

    # Split 70/30
    split = int(len(raw) * 0.7)
    train_df = raw.iloc[:split].copy()
    test_df = raw.iloc[split:].copy()
    test_days = (pd.to_datetime(test_df["timestamp"].iloc[-1]) - pd.to_datetime(test_df["timestamp"].iloc[0])).total_seconds() / 86400

    logger.info("Training models on %d rows, testing on %d rows (%.1f days)...", len(train_df), len(test_df), test_days)

    # Train models
    regime_model = train_regime_model(train_df, hor, adx_threshold=cfg.get("regime_adx", 25.0))
    direction_model = train_direction_model(train_df, hor, max_spw=cfg.get("max_spw", 12.0))

    # Backtest: bot completo (skip_regime overrides use_regime for this symbol)
    _use_regime_full = not cfg.get("skip_regime", False)
    state_full = simulate_bot(test_df, regime_model, direction_model, symbol, use_regime=_use_regime_full)
    metrics_full = calc_metrics(state_full, test_days)

    # Backtest: sin regime
    state_no_regime = simulate_bot(test_df, regime_model, direction_model, symbol, use_regime=False)
    metrics_no_regime = calc_metrics(state_no_regime, test_days)

    # Buy and hold
    start_p = float(test_df["close"].iloc[0])
    end_p = float(test_df["close"].iloc[-1])
    bnh_pct = (end_p - start_p) / start_p * 100

    return {
        "symbol": symbol,
        "test_days": round(test_days, 1),
        "bot_with_regime": metrics_full,
        "bot_without_regime": metrics_no_regime,
        "buy_and_hold_pct": round(bnh_pct, 2),
    }


def main() -> int:
    results = []
    for symbol in ["BTC/USDT", "ETH/USDT", "XRP/USDT"]:
        try:
            res = run_backtest(symbol, limit=30_000)
            results.append(res)
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", symbol, exc, exc_info=True)

    # Summary
    logger.info("=" * 80)
    logger.info("BACKTEST RESULTS SUMMARY (Out-of-sample, 30%% test set)")
    logger.info("=" * 80)
    logger.info(
        "%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-6s",
        "Symbol", "Trades", "Win%", "PnL%", "DD%", "PF", "Sharpe", "Trd/Wk"
    )
    logger.info("-" * 80)
    for r in results:
        if "error" in r:
            continue
        m = r["bot_with_regime"]
        if m["trades"] == 0:
            logger.info("%-10s | NO TRADES (regime+threshold too strict)", r["symbol"])
            continue
        logger.info(
            "%-10s | %-8d | %-8.1f | %-8.2f | %-8.2f | %-8.2f | %-8.2f | %-6.1f",
            r["symbol"], m["trades"], m["win_rate"] * 100, m["pnl_pct"],
            m.get("max_drawdown_pct", 0.0), m.get("profit_factor", 0.0), m.get("sharpe", 0.0), m.get("trades_per_week", 0.0),
        )

    logger.info("-" * 80)
    logger.info("WITHOUT REGIME FILTER:")
    for r in results:
        if "error" in r:
            continue
        m = r["bot_without_regime"]
        if m["trades"] == 0:
            logger.info("%-10s | NO TRADES", r["symbol"])
            continue
        logger.info(
            "%-10s | %-8d | %-8.1f | %-8.2f | %-8.2f | %-8.2f | %-8.2f | %-6.1f",
            r["symbol"], m["trades"], m["win_rate"] * 100, m["pnl_pct"],
            m.get("max_drawdown_pct", 0.0), m.get("profit_factor", 0.0), m.get("sharpe", 0.0), m.get("trades_per_week", 0.0),
        )

    logger.info("-" * 80)
    logger.info("BUY & HOLD:")
    for r in results:
        if "error" not in r:
            logger.info("  %s: %.2f%%", r["symbol"], r["buy_and_hold_pct"])

    # Save
    out = Path("logs/backtest_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Results saved to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
