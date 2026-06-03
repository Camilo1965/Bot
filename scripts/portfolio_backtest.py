#!/usr/bin/env python3
"""
scripts/portfolio_backtest.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Portfolio-level backtest: single shared balance, MAX_POSITIONS global cap,
per-symbol disk-loaded models + calibrations. Mirrors the real bot's capital
allocation more faithfully than per-symbol isolated backtests.

Usage:
    python scripts/portfolio_backtest.py --days 30 --balance 100
    python scripts/portfolio_backtest.py --days 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.backtest_disk_loaded import _fetch_for_symbol, _load_booster, _slice_recent_days
from scripts.backtest_full_bot import (
    ACTIVATION_PCT,
    DAILY_LOSS_LIMIT,
    INITIAL_BALANCE,
    LEVERAGE,
    MAX_POSITIONS,
    SYMBOL_CONFIG,
    TP_ATR_MULT,
    TRAILING_DISTANCE,
    TTL_HOURS,
    USE_HTF_FILTER,
    HTF_SLOW_PERIOD,
    MAX_ATR_PCT,
)
from strategy.prob_calibration import load_calibration
from strategy.quant_features import QUANT_FEATURE_COLS, add_quant_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("portfolio_backtest")

MODELS_DIR = _REPO / "models"

MAX_PORTFOLIO_RISK_PCT = 0.10  # max fraction of balance in open risk across all symbols
MAX_SYMBOL_NOTIONAL_FRACTION = 0.35  # single symbol notional cap: max 35% of balance
MAX_SYMBOL_PNL_FRACTION = 0.25  # 2026-06-03: tightened from 35% — NEAR was 66% of portfolio PnL
KELLY_MAX_FRACTION = 0.25       # half-Kelly cap: never exceed 25% of balance on a single bet
MIN_ATR_PCT = 0.005             # 2026-06-03: skip flat markets (ATR < 0.5% of price) to avoid chop

ACTIVE_SYMBOLS = list(SYMBOL_CONFIG.keys())


@dataclass
class PortPos:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float
    sl_price: float
    tp_price: float | None
    peak_price: float
    current_stop: float
    ml_confidence: float
    ttl_hours: float
    spread_frac: float
    slip_mult: float
    sl_frac: float
    close_reason: str | None = None
    exit_price: float | None = None
    exit_time: pd.Timestamp | None = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0


def _prep_symbol(symbol: str, days: int) -> dict[str, Any] | None:
    cfg = SYMBOL_CONFIG[symbol]
    tf = cfg["timeframe"]
    tf_min = int(tf[:-1])

    raw = _fetch_for_symbol(symbol, days, buffer_days=30)
    if raw is None or len(raw) < 500:
        logger.warning("[%s] insufficient data", symbol)
        return None

    sym_key = symbol.replace("/", "_")
    dir_model = _load_booster(MODELS_DIR / f"{sym_key}_v2.json")
    if dir_model is None:
        logger.warning("[%s] model missing — skipping", symbol)
        return None

    feat_full = add_quant_features(raw.copy(), base_timeframe_min=tf_min)
    feat_full = feat_full.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)
    feat = _slice_recent_days(feat_full, days)
    if len(feat) < 50:
        return None

    X = feat[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    raw_probs = dir_model.predict_proba(X)[:, 1]
    cal = load_calibration(symbol)
    if cal is not None:
        xs, ys = cal
        cal_probs = np.clip(np.interp(raw_probs, xs, ys), 0.0, 1.0)
    else:
        cal_probs = raw_probs.astype(np.float64)

    feat = feat.copy()
    feat["cal_prob"] = cal_probs
    feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)
    feat["htf_sma"] = feat["close"].rolling(HTF_SLOW_PERIOD, min_periods=HTF_SLOW_PERIOD).mean()
    feat = feat.set_index("timestamp").sort_index()

    logger.info("[%s] %d bars, cal_prob mean=%.3f", symbol, len(feat), float(cal_probs.mean()))
    return {"symbol": symbol, "feat": feat, "cfg": cfg}


def _common_grid(sym_data: list[dict]) -> pd.DatetimeIndex:
    """15m UTC grid over intersection of all symbol windows."""
    starts = [sd["feat"].index.min() for sd in sym_data]
    ends = [sd["feat"].index.max() for sd in sym_data]
    return pd.date_range(max(starts), min(ends), freq="15min", tz="UTC")


def run_portfolio(days: int = 30, initial_balance: float = INITIAL_BALANCE) -> dict[str, Any]:
    sym_data = []
    for sym in ACTIVE_SYMBOLS:
        sd = _prep_symbol(sym, days)
        if sd is not None:
            sym_data.append(sd)

    if not sym_data:
        return {"error": "no_symbols_loaded"}

    grid = _common_grid(sym_data)
    logger.info("Common grid: %d 15m bars (%s → %s)", len(grid), grid[0].date(), grid[-1].date())

    sym_lookup = {sd["symbol"]: sd for sd in sym_data}

    balance = initial_balance
    peak_equity = initial_balance
    max_drawdown = 0.0
    open_pos: dict[str, PortPos] = {}
    closed: list[PortPos] = []
    daily_loss = 0.0
    current_day = None

    signals_total = 0
    signals_executed = 0
    signals_dropped_cap = 0
    cap_util_sum = 0.0
    sym_realized_pnl: dict[str, float] = {s: 0.0 for s in ACTIVE_SYMBOLS}

    for bar_ts in grid:
        bar_date = bar_ts.date()
        if current_day is None or bar_date != current_day:
            daily_loss = 0.0
            current_day = bar_date

        # ── Exit checks ──
        to_close = []
        for sym, pos in open_pos.items():
            feat = sym_lookup[sym]["feat"]
            idx = feat.index.searchsorted(bar_ts, side="right") - 1
            if idx < 0:
                continue
            row = feat.iloc[idx]
            cur_p = float(row["close"])
            cur_hi = float(row["high"])
            cur_lo = float(row["low"])
            cur_atr = float(row.get("atr", 0.0))
            atr_pct = cur_atr / cur_p if cur_p > 0 else 0.0

            if cur_hi > pos.peak_price:
                pos.peak_price = cur_hi
            if pos.peak_price >= pos.entry_price * (1.0 + ACTIVATION_PCT):
                new_stop = pos.peak_price * (1.0 - TRAILING_DISTANCE)
                if new_stop > pos.current_stop:
                    pos.current_stop = new_stop

            hours_held = (bar_ts - pos.entry_time).total_seconds() / 3600.0
            hit_sl = cur_lo <= pos.current_stop
            hit_tp = pos.tp_price is not None and cur_hi >= pos.tp_price
            hit_ttl = hours_held >= pos.ttl_hours

            if hit_sl:
                exit_p = pos.current_stop * (1.0 - pos.spread_frac - pos.slip_mult * atr_pct * 0.5)
                reason = "SL"
            elif hit_tp:
                exit_p = pos.tp_price * (1.0 - pos.spread_frac)
                reason = "TP"
            elif hit_ttl:
                exit_p = cur_p * (1.0 - pos.spread_frac)
                reason = "TTL"
            else:
                continue

            pnl_pct = (exit_p - pos.entry_price) / pos.entry_price
            pnl_usd = pnl_pct * pos.position_size - pos.position_size * 0.0004 * 2
            balance += pnl_usd
            daily_loss += min(0.0, pnl_usd)
            sym_realized_pnl[sym] = sym_realized_pnl.get(sym, 0.0) + pnl_usd
            pos.close_reason = reason
            pos.exit_price = exit_p
            pos.exit_time = bar_ts
            pos.pnl_usd = pnl_usd
            pos.pnl_pct = pnl_pct
            closed.append(pos)
            to_close.append(sym)

        for sym in to_close:
            del open_pos[sym]

        # Equity + drawdown
        unrealized = 0.0
        for sym, pos in open_pos.items():
            feat = sym_lookup[sym]["feat"]
            idx = feat.index.searchsorted(bar_ts, side="right") - 1
            if idx >= 0:
                cp = float(feat.iloc[idx]["close"])
                unrealized += (cp - pos.entry_price) / pos.entry_price * pos.position_size
        equity = balance + unrealized
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

        cap_util_sum += len(open_pos) / MAX_POSITIONS

        # ── Entry logic ──
        if abs(daily_loss) >= balance * DAILY_LOSS_LIMIT:
            continue
        if len(open_pos) >= MAX_POSITIONS:
            continue

        candidates = []
        for sym in ACTIVE_SYMBOLS:
            if sym in open_pos:
                continue
            sd = sym_lookup.get(sym)
            if sd is None:
                continue
            feat = sd["feat"]
            cfg = sd["cfg"]
            pt = cfg["prob_threshold"]

            idx = feat.index.searchsorted(bar_ts, side="right") - 1
            if idx < 0:
                continue
            row = feat.iloc[idx]
            cal_prob = float(row.get("cal_prob", 0.0))
            if cal_prob < pt:
                continue

            cur_p = float(row["close"])
            cur_atr = float(row.get("atr", 0.0))
            htf_sma = row.get("htf_sma", np.nan)
            if USE_HTF_FILTER and not np.isnan(htf_sma) and cur_p < float(htf_sma):
                continue
            if MAX_ATR_PCT is not None and cur_p > 0 and cur_atr / cur_p > MAX_ATR_PCT:
                continue
            # 2026-06-03: volatility floor — skip flat markets (chop kills WR)
            if cur_p > 0 and cur_atr / cur_p < MIN_ATR_PCT:
                continue
            # E3: ADX-based regime gate (only when skip_regime=False in config)
            if not cfg.get("skip_regime", True):
                adx_val = float(row.get("adx", 0.0))
                if adx_val < cfg.get("regime_adx", 25.0):
                    continue

            signals_total += 1
            candidates.append((cal_prob - pt, sym, cal_prob, row, cfg))

        candidates.sort(key=lambda x: x[0], reverse=True)

        open_risk = sum(p.position_size * p.sl_frac for p in open_pos.values())

        for edge, sym, cal_prob, row, cfg in candidates:
            if len(open_pos) >= MAX_POSITIONS:
                signals_dropped_cap += 1
                continue

            base_sl = cfg["fixed_sl_pct"]
            risk_pct = cfg["risk"]
            ec = cfg.get("exec_costs", {})
            spread_frac = ec.get("spread_bps", 3.0) / 10000.0
            slip_mult = ec.get("slippage_atr_mult", 0.07)
            cur_p = float(row["close"])
            cur_atr = float(row.get("atr", 0.0))
            atr_pct = cur_atr / cur_p if cur_p > 0 else 0.0
            # E4: ATR-adaptive SL — tighter in quiet markets, wider in volatile
            # Clamp: [fixed_sl * 0.5, fixed_sl * 2.0] so we never go absurd
            if cur_atr > 0 and cur_p > 0:
                atr_sl = 2.0 * atr_pct
                sl_frac = float(np.clip(atr_sl, base_sl * 0.5, base_sl * 2.0))
            else:
                sl_frac = base_sl

            # B1: if one symbol dominates realized PnL, halve its risk
            total_realized = sum(sym_realized_pnl.values())
            sym_pnl_share = sym_realized_pnl.get(sym, 0.0) / total_realized if total_realized > 0 else 0.0
            if sym_pnl_share > MAX_SYMBOL_PNL_FRACTION:
                risk_pct = risk_pct * 0.5

            new_risk = balance * risk_pct
            if open_risk + new_risk > balance * MAX_PORTFOLIO_RISK_PCT:
                signals_dropped_cap += 1
                continue

            # B2: Kelly fraction sizing
            # f* = (p*b - q) / b  where b = tp/sl R:R ratio
            fixed_tp = cfg.get("fixed_tp_pct", sl_frac * 2.0)
            b = float(fixed_tp) / max(sl_frac, 0.001)  # R:R ratio
            p = float(cal_prob)
            q = 1.0 - p
            kelly_f = (p * b - q) / b if b > 0 else 0.0
            half_kelly = max(0.0, kelly_f * 0.5)  # half-Kelly reduces over-betting
            kelly_risk = min(half_kelly, KELLY_MAX_FRACTION)
            # Use Kelly if it's more conservative than config risk; else use config
            effective_risk = min(risk_pct, kelly_risk) if kelly_risk > 0 else risk_pct

            pos_size = (balance * effective_risk) / max(0.001, sl_frac)
            # B1: hard cap — single symbol notional ≤ 35% of balance
            pos_size = min(pos_size, balance * MAX_SYMBOL_NOTIONAL_FRACTION)
            pos_size = min(pos_size, (balance / MAX_POSITIONS) * LEVERAGE)
            if pos_size <= 0 or pos_size > balance * LEVERAGE:
                continue

            entry_fill = cur_p * (1.0 + spread_frac + slip_mult * atr_pct)
            sl_price = entry_fill * (1.0 - sl_frac)
            fixed_tp = cfg.get("fixed_tp_pct")
            if fixed_tp is not None and cur_atr > 0:
                # E4: TP scales proportionally with adaptive SL to keep R:R constant
                rr_ratio = float(fixed_tp) / max(base_sl, 0.001)
                tp_frac = max(sl_frac * rr_ratio, (cur_atr * TP_ATR_MULT) / entry_fill)
                tp_price = entry_fill * (1.0 + tp_frac)
            elif fixed_tp is not None:
                tp_price = entry_fill * (1.0 + float(fixed_tp))
            else:
                tp_price = None

            tf_min = int(cfg["timeframe"][:-1])
            ttl_h = max(TTL_HOURS, int(cfg.get("horizon", 20)) * tf_min / 60.0)

            open_pos[sym] = PortPos(
                symbol=sym,
                entry_time=bar_ts,
                entry_price=entry_fill,
                position_size=pos_size,
                sl_price=sl_price,
                tp_price=tp_price,
                peak_price=entry_fill,
                current_stop=sl_price,
                ml_confidence=cal_prob,
                ttl_hours=ttl_h,
                spread_frac=spread_frac,
                slip_mult=slip_mult,
                sl_frac=sl_frac,
            )
            balance -= pos_size * 0.0004
            open_risk += new_risk
            signals_executed += 1

    # Close remaining at final bar
    last_ts = grid[-1]
    for sym, pos in list(open_pos.items()):
        feat = sym_lookup[sym]["feat"]
        exit_p = float(feat.iloc[-1]["close"]) * (1.0 - pos.spread_frac)
        pnl_pct = (exit_p - pos.entry_price) / pos.entry_price
        pnl_usd = pnl_pct * pos.position_size - pos.position_size * 0.0004 * 2
        balance += pnl_usd
        pos.close_reason = "END_OF_DATA"
        pos.exit_price = exit_p
        pos.exit_time = last_ts
        pos.pnl_usd = pnl_usd
        pos.pnl_pct = pnl_pct
        closed.append(pos)

    # Compute aggregate metrics
    test_days = (grid[-1] - grid[0]).total_seconds() / 86400.0 if len(grid) > 1 else 1.0
    n = len(closed)
    n_wins = sum(1 for t in closed if t.pnl_usd > 0)
    gross_w = sum(t.pnl_usd for t in closed if t.pnl_usd > 0)
    gross_l = sum(abs(t.pnl_usd) for t in closed if t.pnl_usd <= 0)
    pnl_pct = (balance - initial_balance) / initial_balance * 100.0
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = n_wins / n if n > 0 else 0.0
    returns = [t.pnl_pct for t in closed]
    sharpe = 0.0
    if len(returns) > 1:
        mr = np.mean(returns)
        sr = np.std(returns)
        if sr > 0:
            sharpe = (mr / sr) * np.sqrt(365 / max(test_days, 1))

    per_sym: dict[str, dict] = {}
    for t in closed:
        s = t.symbol
        if s not in per_sym:
            per_sym[s] = {"trades": 0, "wins": 0, "pnl_usd": 0.0, "close_reasons": {}}
        per_sym[s]["trades"] += 1
        per_sym[s]["wins"] += 1 if t.pnl_usd > 0 else 0
        per_sym[s]["pnl_usd"] += t.pnl_usd
        r = t.close_reason or "?"
        per_sym[s]["close_reasons"][r] = per_sym[s]["close_reasons"].get(r, 0) + 1

    n_bars = len(grid)
    return {
        "test_days": round(test_days, 1),
        "initial_balance": round(initial_balance, 2),
        "final_balance": round(balance, 2),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(balance - initial_balance, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "trades": n,
        "trades_per_week": round(n / max(test_days / 7, 1), 1),
        "signals_total": signals_total,
        "signals_executed": signals_executed,
        "signals_dropped_by_cap": signals_dropped_cap,
        "capital_utilization_avg_pct": round(cap_util_sum / max(n_bars, 1) * 100, 1),
        "per_symbol": {
            s: {
                "trades": v["trades"],
                "win_rate": round(v["wins"] / v["trades"], 3) if v["trades"] > 0 else 0.0,
                "pnl_usd": round(v["pnl_usd"], 2),
                "pnl_share_pct": round(v["pnl_usd"] / (balance - initial_balance) * 100, 1) if (balance - initial_balance) != 0 else 0.0,
                "close_reasons": v["close_reasons"],
            }
            for s, v in sorted(per_sym.items(), key=lambda x: x[1]["pnl_usd"], reverse=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--balance", type=float, default=INITIAL_BALANCE)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Portfolio backtest | %dd | $%.2f initial | symbols: %s",
                args.days, args.balance, ", ".join(ACTIVE_SYMBOLS))
    logger.info("=" * 70)

    result = run_portfolio(days=args.days, initial_balance=args.balance)

    if "error" in result:
        logger.error("Failed: %s", result["error"])
        return 1

    logger.info("")
    logger.info("Portfolio Summary")
    logger.info("-" * 50)
    logger.info("Test window    : %.1f days", result["test_days"])
    logger.info("Initial balance: $%.2f", result["initial_balance"])
    logger.info("Final balance  : $%.2f", result["final_balance"])
    logger.info("PnL            : %+.2f%% ($%.2f)", result["pnl_pct"], result["pnl_usd"])
    logger.info("Max drawdown   : %.2f%%", result["max_drawdown_pct"])
    logger.info("Win rate       : %.1f%%", result["win_rate"] * 100)
    logger.info("Profit factor  : %.2f", result["profit_factor"])
    logger.info("Sharpe         : %.2f", result["sharpe"])
    logger.info("Trades         : %d (%.1f/wk)", result["trades"], result["trades_per_week"])
    logger.info("Signals        : %d total → %d executed, %d dropped by cap",
                result["signals_total"], result["signals_executed"], result["signals_dropped_by_cap"])
    logger.info("Cap utilization: %.1f%% avg", result["capital_utilization_avg_pct"])
    logger.info("")
    logger.info("%-12s | %-6s | %-6s | %-10s | %-8s | %s", "Symbol", "Trades", "WR%", "PnL USD", "Share%", "Exits")
    logger.info("-" * 70)
    for s, v in result["per_symbol"].items():
        reasons = ", ".join(f"{k}:{n}" for k, n in v["close_reasons"].items())
        logger.info("%-12s | %-6d | %-6.1f | %+.2f      | %+.1f%%    | %s",
                    s, v["trades"], v["win_rate"] * 100, v["pnl_usd"], v["pnl_share_pct"], reasons)

    out = _REPO / "logs" / "portfolio_backtest_30d.json"
    out.parent.mkdir(exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **result}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("")
    logger.info("Saved %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
