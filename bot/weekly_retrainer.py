"""Background weekly XGBoost re-training (Sunday 00:00 UTC)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database.db_manager import db
from strategy.ml_predictor import MLPredictor

_RETRAINER_DATA_LIMIT: int = 10_000


async def weekly_retrainer(
    predictor: MLPredictor,
    watchlist: list[str],
    model_path: Path,
    data_limit: int = _RETRAINER_DATA_LIMIT,
) -> None:
    """[PRO] Background task - re-trains the ML model every Sunday at 00:00 UTC."""
    log = logging.getLogger("clawdbot.retrainer")

    def _seconds_until_next_sunday_midnight() -> float:
        """Return the number of seconds until the next Sunday 00:00 UTC."""
        now = datetime.now(tz=timezone.utc)
        days_until_sunday = (7 - now.isoweekday()) % 7
        candidate = (now + timedelta(days=days_until_sunday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if candidate < now:
            candidate += timedelta(weeks=1)
        return (candidate - now).total_seconds()

    while True:
        wait_secs = _seconds_until_next_sunday_midnight()
        log.info(
            "[PRO] Weekly Re-trainer sleeping %.1f hours until Sunday 00:00 UTC.",
            wait_secs / 3600,
        )
        await asyncio.sleep(wait_secs)

        log.info("[PRO] Weekly Re-training started - fetching latest market data.")
        all_prices: list[float] = []
        for sym in watchlist:
            try:
                prices = await db.fetch_market_data(symbol=sym, limit=data_limit)
                all_prices.extend(prices)
                log.info("[PRO] Fetched %d prices for %s.", len(prices), sym)
            except Exception as exc:  # noqa: BLE001
                log.warning("[PRO] Could not fetch data for %s: %s", sym, exc)

        if len(all_prices) < 50:
            log.warning(
                "[PRO] Re-training skipped - only %d price samples available (need >= 50).",
                len(all_prices),
            )
            continue

        log.info("[PRO] Re-training model on %d total price samples.", len(all_prices))
        success = predictor.warm_start(prices=all_prices)
        if not success:
            log.warning("[PRO] Re-training failed (warm_start returned False).")
            continue

        saved = predictor.save_model(model_path)
        if not saved:
            log.warning("[PRO] Re-training complete but model could not be saved to %s.", model_path)
            continue

        reloaded = predictor.load_model(model_path)
        if reloaded:
            log.info("[PRO] Weekly Re-training complete - model hot-reloaded from %s.", model_path)
        else:
            log.warning(
                "[PRO] Re-training complete but hot-reload from %s failed.", model_path
            )
