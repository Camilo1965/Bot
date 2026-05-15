"""Background weekly XGBoost re-training (Sunday 00:00 UTC).

Each Sunday the retrainer:
1. Fetches the latest OHLCV data per symbol (up to ``_RETRAINER_DATA_LIMIT`` rows).
2. Retrains the XGBoost model on that data.
3. **Validates the AUC** of the new model.  If AUC < ``RETRAINER_MIN_AUC``
   (default 0.52, configurable via env), the new model is **discarded** and the
   previous model stays in production.  This prevents silently deploying a degraded
   model when market data is noisy or insufficient.
4. Backs up the previous model to ``<model>.<timestamp>.bak`` before overwriting.
5. Hot-reloads the predictor so the next signal cycle uses the fresh weights.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from strategy.ml_predictor import MLPredictor
from strategy.quant_features import (
    QUANT_FEATURE_COLS,
    add_quant_features,
    forward_return_label,
    DEFAULT_LABEL_ROUND_TRIP,
)

_RETRAINER_DATA_LIMIT: int = 10_000
# Minimum AUC on the train set to accept the new model.  Purposely low (0.52)
# because train-set AUC of a freshly warm-started model is a weak signal —
# we just want to catch obviously broken models (AUC ≈ 0.50 = random).
_DEFAULT_MIN_AUC: float = 0.52


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _evaluate_auc(model, X: pd.DataFrame, y: pd.Series) -> float:
    """Return ROC-AUC on the provided (X, y) split.  Returns 0.0 on any error."""
    try:
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        proba = model.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:  # noqa: BLE001
        return 0.0


async def weekly_retrainer(
    predictor: MLPredictor,
    watchlist: list[str],
    model_path: Path,
    data_limit: int = _RETRAINER_DATA_LIMIT,
) -> None:
    """[PRO] Background task — re-trains the ML model every Sunday at 00:00 UTC.

    Guards against deploying degraded models by requiring a minimum AUC threshold.
    Set ``RETRAINER_MIN_AUC`` in .env to override (default 0.52).
    """
    log = logging.getLogger("clawdbot.retrainer")
    min_auc: float = _float_env("RETRAINER_MIN_AUC", _DEFAULT_MIN_AUC)

    def _seconds_until_next_sunday_midnight() -> float:
        """Return the number of seconds until the next Sunday 00:00 UTC."""
        now = datetime.now(tz=timezone.utc)
        days_until_sunday = (7 - now.isoweekday()) % 7
        candidate = (now + timedelta(days=days_until_sunday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if candidate <= now:
            candidate += timedelta(weeks=1)
        return (candidate - now).total_seconds()

    while True:
        wait_secs = _seconds_until_next_sunday_midnight()
        log.info(
            "[PRO] Weekly Re-trainer sleeping %.1f hours until Sunday 00:00 UTC.",
            wait_secs / 3600,
        )
        await asyncio.sleep(wait_secs)

        log.info("[PRO] Weekly Re-training started — fetching latest market data.")
        all_prices: list[float] = []
        all_highs: list[float] = []
        all_lows: list[float] = []
        all_volumes: list[float] = []

        for sym in watchlist:
            try:
                # Local import to pick up any reconnected pool.
                from database.db_manager import db as _db  # noqa: PLC0415
                rows = await _db.fetch_market_data_ohlcv(symbol=sym, limit=data_limit)
                if rows:
                    all_prices.extend(float(r["close"]) for r in rows)
                    all_highs.extend(float(r.get("high") or r["close"]) for r in rows)
                    all_lows.extend(float(r.get("low") or r["close"]) for r in rows)
                    all_volumes.extend(float(r.get("volume") or 0.0) for r in rows)
                    log.info("[PRO] Fetched %d OHLCV rows for %s.", len(rows), sym)
                else:
                    log.warning("[PRO] No OHLCV data for %s — skipping.", sym)
            except Exception as exc:  # noqa: BLE001
                log.warning("[PRO] Could not fetch data for %s: %s", sym, exc)

        if len(all_prices) < 50:
            log.warning(
                "[PRO] Re-training skipped — only %d price samples (need >= 50).",
                len(all_prices),
            )
            continue

        # ── Build feature matrix ───────────────────────────────────────────────
        log.info("[PRO] Building feature matrix from %d price samples.", len(all_prices))
        try:
            n = len(all_prices)
            highs_use = all_highs if len(all_highs) == n else list(all_prices)
            lows_use  = all_lows  if len(all_lows)  == n else list(all_prices)
            vols_use  = all_volumes if len(all_volumes) == n else [0.0] * n

            df = pd.DataFrame({
                "open":   all_prices,
                "high":   highs_use,
                "low":    lows_use,
                "close":  all_prices,
                "volume": vols_use,
            })
            feat = add_quant_features(df)
            feat["label"] = forward_return_label(
                feat["close"],
                horizon=5,
                round_trip_cost=DEFAULT_LABEL_ROUND_TRIP,
            )
            feat = feat.dropna(subset=QUANT_FEATURE_COLS + ["label"])

            if len(feat) < 20:
                log.warning(
                    "[PRO] Not enough labelled rows (%d) after feature engineering — skipping.",
                    len(feat),
                )
                continue

            X = feat[QUANT_FEATURE_COLS]
            y = feat["label"]
        except Exception as exc:  # noqa: BLE001
            log.warning("[PRO] Feature engineering failed: %s — skipping retrain.", exc)
            continue

        # ── Retrain model ──────────────────────────────────────────────────────
        log.info("[PRO] Re-training XGBoost on %d labelled samples.", len(X))
        success = predictor.warm_start(
            prices=all_prices,
            highs=highs_use,
            lows=lows_use,
            volumes=vols_use,
        )
        if not success:
            log.warning("[PRO] warm_start returned False — keeping current model.")
            continue

        # ── AUC validation guard ───────────────────────────────────────────────
        # Evaluate train-set AUC as a sanity check.  Values < 0.52 indicate the
        # model is effectively random (e.g. noisy or corrupted data).
        try:
            new_auc = _evaluate_auc(predictor._model, X, y)
        except Exception as exc:  # noqa: BLE001
            log.warning("[PRO] AUC evaluation failed: %s — treating as 0.0.", exc)
            new_auc = 0.0

        log.info(
            "[PRO] Retrained model train-set AUC = %.4f (threshold = %.4f)",
            new_auc,
            min_auc,
        )

        if new_auc < min_auc:
            log.warning(
                "[PRO] ⚠️  New model AUC %.4f < threshold %.4f — DISCARDING. "
                "Keeping current production model in %s.",
                new_auc,
                min_auc,
                model_path,
            )
            # Restore the previous model so the predictor stays consistent.
            if model_path.is_file():
                predictor.load_model(model_path)
                log.info("[PRO] Previous model restored from %s.", model_path)
            continue

        # ── Back up previous model before overwriting ──────────────────────────
        if model_path.is_file():
            ts_tag = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = model_path.with_suffix(f".{ts_tag}.bak")
            try:
                shutil.copy2(model_path, backup_path)
                log.info("[PRO] Previous model backed up to %s.", backup_path.name)
            except Exception as exc:  # noqa: BLE001
                log.warning("[PRO] Could not back up previous model: %s — continuing.", exc)

        # ── Save and hot-reload ────────────────────────────────────────────────
        saved = predictor.save_model(model_path)
        if not saved:
            log.warning("[PRO] Model could not be saved to %s.", model_path)
            continue

        reloaded = predictor.load_model(model_path)
        if reloaded:
            log.info(
                "[PRO] ✅ Weekly Re-training complete — AUC=%.4f — hot-reloaded from %s.",
                new_auc,
                model_path,
            )
        else:
            log.warning(
                "[PRO] Re-training complete (AUC=%.4f) but hot-reload from %s failed.",
                new_auc,
                model_path,
            )
