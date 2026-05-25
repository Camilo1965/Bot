"""Walk-forward validation for ClawdBot XGBoost models.

Simulates live trading by training on a rolling window and testing on the
next out-of-sample block, repeating across the full historical data.

Usage:
    python scripts/walk_forward_validate.py --symbol ETH/USDT --days 180
    python scripts/walk_forward_validate.py --symbol BTC/USDT --days 365 --train-days 90 --test-days 30

Reports per-window AUC, Precision, Recall, Profit Factor, and win rate to
let you detect model degradation across market regimes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wfv")

from strategy.quant_features import (
    QUANT_FEATURE_COLS,
    add_quant_features,
    triple_barrier_label,
)
from strategy.ml_predictor import get_symbol_config
from strategy.prob_calibration import fit_and_save_calibration, calibrate_probability


def _calc_spw(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return max(1.0, min(20.0, n_neg / n_pos)) if n_pos > 0 else 1.0


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    auc = 0.5
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return {
        "auc": round(auc, 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _simulate_trades(
    prices_close: np.ndarray,
    prices_high: np.ndarray,
    prices_low: np.ndarray,
    y_pred: np.ndarray,
    sl_pct: float,
    tp_pct: float,
    horizon: int,
) -> dict[str, float]:
    """Simulate forward-return P&L for predicted BUY signals."""
    wins = losses = 0
    gross_pnl = 0.0
    for i in range(len(y_pred) - horizon):
        if y_pred[i] != 1:
            continue
        entry = prices_close[i]
        sl_price = entry * (1.0 - sl_pct)
        tp_price = entry * (1.0 + tp_pct)
        outcome = "time"
        for j in range(i + 1, min(i + 1 + horizon, len(prices_close))):
            if prices_low[j] <= sl_price:
                outcome = "sl"
                break
            if prices_high[j] >= tp_price:
                outcome = "tp"
                break
        if outcome == "tp":
            gross_pnl += tp_pct
            wins += 1
        elif outcome == "sl":
            gross_pnl -= sl_pct
            losses += 1
        else:
            final = prices_close[min(i + horizon, len(prices_close) - 1)]
            ret = (final - entry) / entry
            gross_pnl += ret
            if ret > 0:
                wins += 1
            else:
                losses += 1
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    profit_factor = (wins * tp_pct) / (losses * sl_pct) if losses > 0 and sl_pct > 0 else float("inf")
    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "gross_pnl_pct": round(gross_pnl * 100, 2),
    }


async def _fetch_ohlcv(symbol: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLCV from TimescaleDB."""
    import os
    import asyncpg
    url = os.environ.get("DATABASE_URL")
    if not url:
        user = os.environ.get("DB_USER", "clawdbot")
        pw = os.environ.get("DB_PASSWORD", "clawdbot_secret")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        db = os.environ.get("DB_NAME", "clawdbot")
        url = f"postgres://{user}:{pw}@{host}:{port}/{db}"
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM market_data WHERE symbol = $1 AND timestamp >= $2 "
        "ORDER BY timestamp ASC"
    )
    try:
        conn = await asyncpg.connect(dsn=url)
        rows = await conn.fetch(query, symbol, cutoff)
        await conn.close()
    except Exception as exc:
        logger.error("DB fetch failed: %s", exc)
        return None
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def walk_forward_validate(
    symbol: str,
    df: pd.DataFrame,
    train_days: int = 90,
    test_days: int = 30,
    timeframe_min: int = 15,
    prob_threshold: float = 0.50,
    sl_pct: float = 0.02,
    tp_pct: float = 0.075,
    horizon: int = 36,
    save_calibration: bool = True,
) -> list[dict[str, Any]]:
    """Run walk-forward validation on *df*.

    Returns a list of per-window result dicts.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return []

    feat = add_quant_features(df.copy(), base_timeframe_min=timeframe_min)
    feat["label"] = triple_barrier_label(
        feat["close"], feat["high"], feat["low"],
        horizon=horizon,
        fixed_sl_pct=sl_pct,
        fixed_tp_pct=tp_pct,
    )
    feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"]).reset_index(drop=True)

    bars_per_day = int(24 * 60 / timeframe_min)
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day

    if len(feat) < train_bars + test_bars:
        logger.warning("Not enough data: %d rows (need %d)", len(feat), train_bars + test_bars)
        return []

    windows: list[dict[str, Any]] = []
    all_y_val: list[np.ndarray] = []
    all_p_val: list[np.ndarray] = []
    cursor = 0

    while cursor + train_bars + test_bars <= len(feat):
        train = feat.iloc[cursor : cursor + train_bars]
        test = feat.iloc[cursor + train_bars : cursor + train_bars + test_bars]

        X_tr = train[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
        y_tr = train["label"].to_numpy(dtype=np.int32)
        X_te = test[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
        y_te = test["label"].to_numpy(dtype=np.int32)

        spw = _calc_spw(y_tr)
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        model.fit(X_tr, y_tr)

        y_prob_raw = model.predict_proba(X_te)[:, 1]
        y_prob_cal = np.array([calibrate_probability(float(p), symbol) for p in y_prob_raw])
        y_pred = (y_prob_cal >= prob_threshold).astype(int)

        metrics = _compute_metrics(y_te, y_pred, y_prob_cal)
        close_arr = test["close"].to_numpy()
        high_arr = test["high"].to_numpy()
        low_arr = test["low"].to_numpy()
        trade_stats = _simulate_trades(close_arr, high_arr, low_arr, y_pred, sl_pct, tp_pct, horizon)

        window_result: dict[str, Any] = {
            "window": len(windows) + 1,
            "train_start": feat.index[cursor] if hasattr(feat.index[cursor], "isoformat") else cursor,
            "test_start": feat.index[cursor + train_bars] if hasattr(feat.index[cursor + train_bars], "isoformat") else cursor + train_bars,
            "train_rows": len(X_tr),
            "test_rows": len(X_te),
            "pos_rate_train": round(float(y_tr.mean()), 4),
            "pos_rate_test": round(float(y_te.mean()), 4),
            **metrics,
            **trade_stats,
        }
        windows.append(window_result)
        all_y_val.append(y_te)
        all_p_val.append(y_prob_raw)

        cursor += test_bars

    # Summary
    if windows:
        logger.info("\n=== Walk-Forward Validation: %s ===", symbol)
        logger.info("%-8s %-8s %-8s %-8s %-8s %-8s %-8s %-10s", "Window", "AUC", "Prec", "Rec", "Trades", "WinRate", "PF", "PnL%")
        for w in windows:
            logger.info(
                "%-8d %-8.4f %-8.4f %-8.4f %-8d %-8.3f %-8.3f %-10.2f",
                w["window"], w["auc"], w["precision"], w["recall"],
                w["trades"], w["win_rate"], w["profit_factor"], w["gross_pnl_pct"],
            )
        avg_auc = np.mean([w["auc"] for w in windows])
        avg_pf = np.mean([w["profit_factor"] for w in windows if w["profit_factor"] != float("inf")])
        total_trades = sum(w["trades"] for w in windows)
        total_pnl = sum(w["gross_pnl_pct"] for w in windows)
        logger.info("─" * 80)
        logger.info("AVG AUC=%.4f  AVG PF=%.3f  TOTAL trades=%d  TOTAL PnL=%.2f%%", avg_auc, avg_pf, total_trades, total_pnl)

    # Fit and save calibration from all walk-forward hold-out data
    if save_calibration and all_y_val and all_p_val:
        try:
            cal_path = fit_and_save_calibration(symbol, np.concatenate(all_y_val), np.concatenate(all_p_val))
            logger.info("Calibration updated → %s", cal_path)
        except Exception as exc:
            logger.warning("Calibration fit failed: %s", exc)

    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validation for ClawdBot models")
    parser.add_argument("--symbol", default="ETH/USDT")
    parser.add_argument("--days", type=int, default=180, help="Total history days to load")
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to path")
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration update")
    args = parser.parse_args()

    cfg = get_symbol_config(args.symbol)
    tf_str = str(cfg.get("timeframe", "15m"))
    tf_min = int(tf_str[:-1]) if tf_str[:-1].isdigit() else 15
    sl_pct = float(cfg.get("fixed_sl_pct", 0.02))
    tp_pct = float(cfg.get("fixed_tp_pct", 0.04))
    horizon = int(cfg.get("horizon", 36))
    prob_threshold = float(cfg.get("prob_threshold", 0.50))

    logger.info("Symbol=%s  SL=%.1f%%  TP=%.1f%%  horizon=%d  threshold=%.2f", args.symbol, sl_pct*100, tp_pct*100, horizon, prob_threshold)

    df = asyncio.run(_fetch_ohlcv(args.symbol, args.days))
    if df is None or len(df) < 500:
        logger.error("Insufficient data for %s", args.symbol)
        return 1

    logger.info("Loaded %d rows for %s", len(df), args.symbol)
    windows = walk_forward_validate(
        args.symbol, df,
        train_days=args.train_days,
        test_days=args.test_days,
        timeframe_min=tf_min,
        prob_threshold=prob_threshold,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        horizon=horizon,
        save_calibration=not args.no_calibration,
    )

    if args.output and windows:
        Path(args.output).write_text(json.dumps(windows, indent=2, default=str))
        logger.info("Results saved → %s", args.output)

    return 0 if windows else 1


if __name__ == "__main__":
    sys.exit(main())
