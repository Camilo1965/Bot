"""XGBoost model retraining script with market-data features and VIBE injection (v2 pipeline).

Usage (24 quant features):
    python scripts/retrain_model.py --symbols BTC/USDT ETH/USDT --days 90

Usage (28 features with VIBE, saves as _v3.json):
    python scripts/retrain_model.py --symbols BTC/USDT ETH/USDT --days 90 --vibe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd

logger = logging.getLogger("clawdbot.retrain")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from strategy.quant_features import (
    FINAL_FEATURE_ORDER,
    QUANT_FEATURE_COLS,
    VIBE_FEATURE_COLS,
    add_quant_features,
    triple_barrier_label,
    DEFAULT_LABEL_ROUND_TRIP,
)
from strategy.ml_predictor import get_symbol_config
from vibe.feature_bridge import VIBE_FEATURE_NEUTRAL

_MODELS_DIR = _ROOT / "models"

_VIBE_NEUTRAL_NOISE_STD: float = 0.05


def _build_dsn() -> str:
    """Build PostgreSQL DSN from env vars or defaults."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    user = os.environ.get("DB_USER", "clawdbot")
    password = os.environ.get("DB_PASSWORD", "clawdbot_secret")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "clawdbot")
    return f"postgres://{user}:{password}@{host}:{port}/{dbname}"


async def _fetch_ohlcv(symbol: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLCV from TimescaleDB for the last *days*."""
    dsn = _build_dsn()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    query = (
        "SELECT timestamp, open, high, low, close, volume "
        "FROM market_data WHERE symbol = $1 AND timestamp >= $2 "
        "ORDER BY timestamp ASC"
    )
    try:
        conn = await asyncpg.connect(dsn=dsn)
        rows = await conn.fetch(query, symbol, cutoff)
        await conn.close()
    except Exception as exc:
        logger.error("DB fetch failed for %s: %s", symbol, exc)
        return None

    if not rows:
        logger.warning("No OHLCV data for %s in the last %d days.", symbol, days)
        return None

    df = pd.DataFrame([dict(r) for r in rows])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def _add_vibe_features(df: pd.DataFrame, add_noise: bool = True) -> pd.DataFrame:
    """Append 4 VIBE columns with neutral values + optional Gaussian noise."""
    neutral = np.array(VIBE_FEATURE_NEUTRAL, dtype=float)
    n = len(df)
    for i, col in enumerate(VIBE_FEATURE_COLS):
        base = np.full(n, neutral[i], dtype=float)
        if add_noise:
            base += np.random.normal(0.0, _VIBE_NEUTRAL_NOISE_STD, size=n)
        df[col] = base
    return df


def _calc_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def _compute_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = 0.5
    return metrics


def retrain_and_validate(
    symbol: str,
    model_path: Path,
    lookback_days: int = 90,
    min_auc: float = 0.55,
    use_vibe: bool = False,
) -> bool:
    """Retrain XGBoost v2 with TimeSeriesSplit and full metrics quality gate.

    Parameters
    ----------
    symbol: Trading pair to train on.
    model_path: Where to save the XGB JSON model.
    lookback_days: Days of OHLCV history to load.
    min_auc: Minimum mean ROC-AUC across CV splits to accept the model.
    use_vibe: If True, inject 4 VIBE features (28-dim vector) and save as _v3.json.

    Returns
    -------
    True if model was saved, False if quality gate rejected it.
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return False

    # 1. Load market data
    df = asyncio.run(_fetch_ohlcv(symbol, lookback_days))
    if df is None or len(df) < 500:
        logger.warning(
            "Not enough OHLCV data for %s (need >=500 rows, got %s).",
            symbol,
            len(df) if df is not None else 0,
        )
        return False

    # 2. Compute quant features
    feat = add_quant_features(df)
    # Use symbol-specific SL/TP/horizon so training barriers match live execution
    _sym_cfg = get_symbol_config(symbol)
    _sl_pct = float(_sym_cfg.get("fixed_sl_pct", 0.02))
    _tp_pct = float(_sym_cfg.get("fixed_tp_pct", 0.04))
    _horizon = int(_sym_cfg.get("horizon", 8))
    feat["label"] = triple_barrier_label(
        feat["close"],
        feat["high"],
        feat["low"],
        horizon=_horizon,
        fixed_sl_pct=_sl_pct,
        fixed_tp_pct=_tp_pct,
    )
    logger.info(
        "[LABEL] %s  triple-barrier  SL=%.1f%%  TP=%.1f%%  horizon=%d bars",
        symbol, _sl_pct * 100, _tp_pct * 100, _horizon,
    )

    # 3. Inject VIBE features if requested
    if use_vibe:
        feat = _add_vibe_features(feat)
        feature_cols = list(FINAL_FEATURE_ORDER)
    else:
        feature_cols = list(QUANT_FEATURE_COLS)

    # Drop rows with NaN in features or label
    feat = feat.dropna(subset=feature_cols + ["label"])
    if len(feat) < 200:
        logger.warning("Not enough labelled samples for %s (need >=200, got %d).", symbol, len(feat))
        return False

    X = feat[feature_cols]
    y = feat["label"]

    # 4. TimeSeriesSplit cross-validation
    n_splits = min(5, max(2, len(X) // 200))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: list[float] = []
    cv_metrics: list[dict[str, float]] = []
    # Accumulate hold-out predictions across all folds for calibration fitting
    all_y_va: list[np.ndarray] = []
    all_probas: list[np.ndarray] = []

    for train_idx, val_idx in tscv.split(X):
        x_tr = X.iloc[train_idx].to_numpy(dtype=np.float32)
        y_tr = y.iloc[train_idx].to_numpy(dtype=np.int32)
        x_va = X.iloc[val_idx].to_numpy(dtype=np.float32)
        y_va = y.iloc[val_idx].to_numpy(dtype=np.int32)

        spw = _calc_scale_pos_weight(y_tr)
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_weight=2,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=spw,
            early_stopping_rounds=20,
        )
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        preds = model.predict(x_va)
        probas = model.predict_proba(x_va)[:, 1]
        m = _compute_metrics(y_va, preds, probas)
        cv_scores.append(m["auc"])
        cv_metrics.append(m)
        all_y_va.append(y_va)
        all_probas.append(probas)

    mean_metrics = {
        k: round(sum(m[k] for m in cv_metrics) / len(cv_metrics), 4)
        for k in ("accuracy", "precision", "recall", "f1", "auc")
    }
    mean_auc = mean_metrics["auc"]

    # 5. Load previous AUC for comparison
    old_auc = 0.0
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text())
            old_auc = float(old_meta.get("metrics", {}).get("auc", old_meta.get("auc", 0.0)))
        except Exception:
            pass

    logger.info(
        "Retrain %s (vibe=%s): old_auc=%.4f new_auc=%.4f acc=%.4f f1=%.4f (splits=%d)",
        symbol,
        use_vibe,
        old_auc,
        mean_auc,
        mean_metrics["accuracy"],
        mean_metrics["f1"],
        len(cv_scores),
    )

    # 6. Quality Gate
    if mean_auc < min_auc:
        logger.warning(
            "Model NOT saved for %s - mean AUC %.4f below min_auc %.4f.",
            symbol,
            mean_auc,
            min_auc,
        )
        return False

    if old_auc > 0.0 and mean_auc <= old_auc + 0.005:
        logger.info(
            "Model NOT saved for %s - no meaningful improvement over old_auc %.4f.",
            symbol,
            old_auc,
        )
        return False

    # 7. Final retrain on all data and save
    x_all = X.to_numpy(dtype=np.float32)
    y_all = y.to_numpy(dtype=np.int32)
    spw_all = _calc_scale_pos_weight(y_all)
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=2,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=spw_all,
    )
    final_model.fit(x_all, y_all)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.get_booster().feature_names = list(feature_cols)
    final_model.save_model(str(model_path))

    importance = final_model.feature_importances_
    feat_imp = {
        col: float(importance[i])
        for i, col in enumerate(feature_cols)
    }

    meta = {
        "metrics": mean_metrics,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": symbol,
        "samples": len(y_all),
        "splits": len(cv_scores),
        "features": feature_cols,
        "vibe_enabled": use_vibe,
        "feature_importance": dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Model UPDATED for %s → %s (AUC %.4f)", symbol, model_path, mean_auc)

    # 8. Fit isotonic calibration from accumulated hold-out predictions
    try:
        from strategy.prob_calibration import fit_and_save_calibration as _fit_cal
        cal_path = _fit_cal(
            symbol,
            np.concatenate(all_y_va),
            np.concatenate(all_probas),
        )
        logger.info("Calibration saved → %s", cal_path)
    except Exception as exc:
        logger.warning("Calibration fitting failed (non-fatal): %s", exc)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain ClawdBot XGBoost models from OHLCV history (v2)")
    parser.add_argument(
        "--symbols",
        default=os.environ.get("WATCHLIST", "BTC/USDT,ETH/USDT"),
        help="Comma-separated list of symbols to train",
    )
    parser.add_argument("--days", type=int, default=90, help="Lookback days for OHLCV")
    parser.add_argument("--min-auc", type=float, default=0.55, help="Minimum mean AUC to accept model")
    parser.add_argument("--vibe", action="store_true", help="Train with 28 features (24 base + 4 VIBE) and save as _v3.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols provided.")
        return 1

    all_ok = True
    for symbol in symbols:
        safe_name = symbol.replace("/", "_")
        suffix = "_v3" if args.vibe else "_v2"
        model_path = _MODELS_DIR / f"{safe_name}{suffix}.json"
        ok = retrain_and_validate(
            symbol=symbol,
            model_path=model_path,
            lookback_days=args.days,
            min_auc=args.min_auc,
            use_vibe=args.vibe,
        )
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
