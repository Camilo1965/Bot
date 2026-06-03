#!/usr/bin/env python3
"""
scripts/refit_calibrations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refit the isotonic calibration `{SYMBOL}_calibration.json` against the
*current* `{SYMBOL}_v2.json` model on data the model has not seen.

The previous calibrations were saved by `scripts/walk_forward_validate.py`
before the 2026-05-27 retrain — they are stale and bias probabilities
toward the old booster.  This script rebuilds them so live signal probs
match observed win rates.

Procedure per symbol:
  1. Read `{SYMBOL}_v2.meta.json["trained_at"]` as the train-window cutoff.
  2. Fetch 45 days of CCXT at the symbol's timeframe.
  3. Slice rows with timestamp > cutoff (truly out-of-sample).
  4. Compute quant features, predict raw probs from the disk booster.
  5. Label with `forward_return_label(close, horizon, DEFAULT_LABEL_ROUND_TRIP)`.
  6. Drop the last `horizon` rows (label unobservable).
  7. Fit isotonic regression → write `{SYMBOL}_calibration.json`.
  8. Fallback: if the post-cutoff slice yields < 800 labeled rows, use the
     last 30 days of the 45-day fetch instead and tag the JSON `_meta.warning`.

Usage:
    python scripts/refit_calibrations.py --all
    python scripts/refit_calibrations.py --symbol BTC/USDT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.backtest_full_bot import SYMBOL_CONFIG
from data.ohlcv_cache import get_cache as _get_ohlcv_cache
from scripts.deep_strategy_audit import fetch_ohlcv_ccxt
from strategy.prob_calibration import _calibration_cache, _calibration_path, fit_and_save_calibration
from strategy.quant_features import (
    DEFAULT_LABEL_ROUND_TRIP,
    QUANT_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
    forward_return_label_short,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("refit_calibrations")

MODELS_DIR: Path = _REPO / "models"
_CANDLES_PER_DAY = {"5m": 288, "15m": 96, "30m": 48, "1h": 24}
MIN_ROWS_POST_CUTOFF = 800


def _load_meta(symbol: str) -> dict | None:
    p = MODELS_DIR / f"{symbol.replace('/', '_')}_v2.meta.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Bad meta json %s: %s", p.name, exc)
        return None


def _load_booster(symbol: str) -> XGBClassifier | None:
    p = MODELS_DIR / f"{symbol.replace('/', '_')}_v2.json"
    if not p.is_file():
        logger.warning("Missing model: %s", p.name)
        return None
    m = XGBClassifier()
    m.load_model(str(p))
    return m


def refit_one(symbol: str, days: int = 45, side: str = "long") -> bool:
    cfg = SYMBOL_CONFIG.get(symbol)
    if cfg is None:
        logger.error("No SYMBOL_CONFIG entry for %s", symbol)
        return False
    tf = cfg["timeframe"]
    horizon = int(cfg.get("horizon", 20))

    if side == "short":
        from strategy.ml_predictor import load_short_booster
        booster = load_short_booster(symbol)
        if booster is None:
            logger.warning("[%s] No SHORT model found — skipping short refit", symbol)
            return False
    else:
        booster = _load_booster(symbol)
    if booster is None:
        return False
    meta = _load_meta(symbol)
    trained_at = None
    if meta and "trained_at" in meta:
        try:
            trained_at = pd.to_datetime(meta["trained_at"], utc=True)
        except Exception:
            trained_at = None

    per_day = _CANDLES_PER_DAY.get(tf, 96)
    limit = days * per_day
    logger.info("[%s] Fetching %d candles (~%dd %s) ...", symbol, limit, days, tf)
    raw = _get_ohlcv_cache().fetch(symbol, tf, limit)
    if raw is None or len(raw) < 500:
        logger.error("[%s] Insufficient fetch (rows=%s)", symbol, 0 if raw is None else len(raw))
        return False

    feat = add_quant_features(raw.copy(), base_timeframe_min=int(tf[:-1]))
    feat = feat.dropna(subset=QUANT_FEATURE_COLS).reset_index(drop=True)

    used_fallback = False
    if trained_at is not None and "timestamp" in feat.columns:
        ts = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")
        oos = feat[ts > trained_at].reset_index(drop=True)
        logger.info("[%s] trained_at=%s, post-cutoff rows=%d", symbol, trained_at.isoformat(), len(oos))
        if len(oos) >= MIN_ROWS_POST_CUTOFF + horizon:
            feat_eval = oos
        else:
            logger.warning("[%s] post-cutoff slice too thin (%d < %d). Falling back to last %dd of fetch.",
                           symbol, len(oos), MIN_ROWS_POST_CUTOFF + horizon, min(days, 30))
            used_fallback = True
            tail_n = min(30, days) * per_day
            feat_eval = feat.tail(tail_n).reset_index(drop=True)
    else:
        logger.warning("[%s] no trained_at — using full window", symbol)
        used_fallback = True
        feat_eval = feat

    # Drop last `horizon` rows (label unobservable)
    if len(feat_eval) <= horizon + 50:
        logger.error("[%s] not enough rows after slicing: %d", symbol, len(feat_eval))
        return False
    feat_eval = feat_eval.iloc[:-horizon].reset_index(drop=True)

    X = feat_eval[QUANT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_prob = booster.predict_proba(X)[:, 1].astype(np.float64)
    if side == "short":
        y_true = forward_return_label_short(feat_eval["close"], horizon, DEFAULT_LABEL_ROUND_TRIP).to_numpy(dtype=np.int32)
    else:
        y_true = forward_return_label(feat_eval["close"], horizon, DEFAULT_LABEL_ROUND_TRIP).to_numpy(dtype=np.int32)

    if len(y_true) != len(y_prob):
        logger.error("[%s] length mismatch: y_true=%d y_prob=%d", symbol, len(y_true), len(y_prob))
        return False
    if int(y_true.sum()) == 0:
        logger.error("[%s] no positive labels in slice — cannot fit calibration", symbol)
        return False

    pos_rate = float(y_true.mean())
    logger.info("[%s] fitting calibration: side=%s n=%d pos_rate=%.4f raw_mean=%.4f",
                symbol, side, len(y_true), pos_rate, float(y_prob.mean()))

    cal_symbol = f"{symbol}_short" if side == "short" else symbol
    path = fit_and_save_calibration(cal_symbol, y_true, y_prob)

    if used_fallback:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_meta"] = {
            "warning": "calibration_overlaps_training",
            "reason": "post_cutoff_slice_too_thin",
            "fallback_window_days": min(days, 30),
            "n_samples": int(len(y_true)),
            "pos_rate": pos_rate,
            "side": side,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _calibration_cache.pop(cal_symbol, None)

    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--side", choices=["long", "short", "both"], default="long",
                        help="Which side calibration to refit")
    args = parser.parse_args()

    if args.symbol is None and not args.all:
        parser.error("Pass --symbol SYM or --all")

    symbols = [args.symbol] if args.symbol else list(SYMBOL_CONFIG.keys())
    sides = ["long", "short"] if args.side == "both" else [args.side]
    ok = 0
    fail = 0
    for sym in symbols:
        for side in sides:
            try:
                if refit_one(sym, days=args.days, side=side):
                    ok += 1
                else:
                    fail += 1
            except Exception as exc:
                logger.error("[%s/%s] refit failed: %s", sym, side, exc, exc_info=True)
                fail += 1

    logger.info("=" * 60)
    logger.info("Refit summary: %d ok, %d failed (of %d)", ok, fail, len(symbols))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
