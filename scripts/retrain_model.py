"""XGBoost model retraining script with market-data features and VIBE injection.

Usage (legacy 12 features):
    python scripts/retrain_model.py --symbols BTC/USDT ETH/USDT --days 90

Usage (VIBE-aware 16 features, saves as _v2.json):
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
    QUANT_FEATURE_COLS,
    VIBE_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
    DEFAULT_LABEL_ROUND_TRIP,
)
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
    """Append 4 VIBE columns with neutral values + optional Gaussian noise.

    Noise prevents zero-variance collapse in XGBoost when training on
    historical data that predates VIBE-Trading deployment.
    """
    neutral = np.array(VIBE_FEATURE_NEUTRAL, dtype=float)
    n = len(df)
    for i, col in enumerate(VIBE_FEATURE_COLS):
        base = np.full(n, neutral[i], dtype=float)
        if add_noise:
            base += np.random.normal(0.0, _VIBE_NEUTRAL_NOISE_STD, size=n)
        df[col] = base
    return df


def _compute_auc(y_true: Any, y_prob: Any) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.0


def retrain_and_validate(
    symbol: str,
    model_path: Path,
    lookback_days: int = 90,
    min_auc: float = 0.52,
    use_vibe: bool = False,
) -> bool:
    """Retrain XGBoost with TimeSeriesSplit and AUC quality gate.

    Parameters
    ----------
    symbol: Trading pair to train on.
    model_path: Where to save the XGB JSON model.
    lookback_days: Days of OHLCV history to load.
    min_auc: Minimum mean ROC-AUC across CV splits to accept the model.
    use_vibe: If True, inject 4 VIBE features (16-dim vector) and save as _v2.json.

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

    # ------------------------------------------------------------------
    # 1. Load market data
    # ------------------------------------------------------------------
    df = asyncio.run(_fetch_ohlcv(symbol, lookback_days))
    if df is None or len(df) < 200:
        logger.warning(
            "Not enough OHLCV data for %s (need ≥200 rows, got %s).",
            symbol,
            len(df) if df is not None else 0,
        )
        return False

    # ------------------------------------------------------------------
    # 2. Compute quant features
    # ------------------------------------------------------------------
    feat = add_quant_features(df)
    feat["label"] = forward_return_label(feat["close"], horizon=5, round_trip_cost=DEFAULT_LABEL_ROUND_TRIP)

    # ------------------------------------------------------------------
    # 3. Inject VIBE features if requested
    # ------------------------------------------------------------------
    feature_cols = list(QUANT_FEATURE_COLS)
    if use_vibe:
        feat = _add_vibe_features(feat)
        feature_cols.extend(VIBE_FEATURE_COLS)

    # Drop rows with NaN in features or label
    feat = feat.dropna(subset=feature_cols + ["label"])
    if len(feat) < 50:
        logger.warning("Not enough labelled samples for %s (need ≥50, got %d).", symbol, len(feat))
        return False

    X = feat[feature_cols]
    y = feat["label"]

    # ------------------------------------------------------------------
    # 4. TimeSeriesSplit cross-validation
    # ------------------------------------------------------------------
    n_splits = min(5, max(2, len(X) // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probas = model.predict_proba(X.iloc[val_idx])[:, 1]
        auc = _compute_auc(y.iloc[val_idx], probas)
        scores.append(auc)

    mean_auc = sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # 5. Load previous AUC for comparison
    # ------------------------------------------------------------------
    old_auc = 0.0
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            old_auc = float(json.loads(meta_path.read_text()).get("auc", 0.0))
        except Exception:
            pass

    logger.info(
        "Retrain %s (vibe=%s): old_auc=%.4f new_auc=%.4f (splits=%d)",
        symbol,
        use_vibe,
        old_auc,
        mean_auc,
        len(scores),
    )

    # ------------------------------------------------------------------
    # 6. Quality Gate
    # ------------------------------------------------------------------
    if mean_auc < min_auc:
        logger.warning(
            "Model NOT saved for %s — mean AUC %.4f below min_auc %.4f.",
            symbol,
            mean_auc,
            min_auc,
        )
        return False

    # Optional: require improvement over previous model
    if old_auc > 0.0 and mean_auc <= old_auc + 0.005:
        logger.info(
            "Model NOT saved for %s — no meaningful improvement over old_auc %.4f.",
            symbol,
            old_auc,
        )
        return False

    # ------------------------------------------------------------------
    # 7. Final retrain on all data and save
    # ------------------------------------------------------------------
    final_model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    final_model.fit(X, y)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))

    meta = {
        "auc": round(mean_auc, 4),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": symbol,
        "samples": len(y),
        "splits": len(scores),
        "features": feature_cols,
        "vibe_enabled": use_vibe,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Model UPDATED for %s → %s (AUC %.4f)", symbol, model_path, mean_auc)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain ClawdBot XGBoost models from OHLCV history")
    parser.add_argument(
        "--symbols",
        default=os.environ.get("WATCHLIST", "BTC/USDT,ETH/USDT"),
        help="Comma-separated list of symbols to train",
    )
    parser.add_argument("--days", type=int, default=90, help="Lookback days for OHLCV")
    parser.add_argument("--min-auc", type=float, default=0.52, help="Minimum mean AUC to accept model")
    parser.add_argument("--vibe", action="store_true", help="Train with 16 features (12 base + 4 VIBE) and save as _v2.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols provided.")
        return 1

    all_ok = True
    for symbol in symbols:
        safe_name = symbol.replace("/", "_")
        suffix = "_v2" if args.vibe else "_v1"
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
