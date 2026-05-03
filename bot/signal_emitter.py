"""Periodic ML signal evaluation and trade/open/exit orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

from database.db_manager import db

from bot.constants import (
    DEBUG_LOG_HINT,
    NEWS_FILTER_HOLD_MINUTES,
    NEWS_FILTER_VOLATILITY_THRESHOLD,
)

from execution.paper_executor import PaperExecutor
from strategy.feature_engineer import FeatureEngineer
from strategy.ml_predictor import BUY_PROB_THRESHOLD, MLPredictor, compute_htf_trend


async def signal_emitter(
    state: dict[str, Any],
    predictor: MLPredictor,
    paper_executor: PaperExecutor,
    watchlist: list[str],
    interval: int = 15,
) -> None:
    logger = logging.getLogger("clawdbot.signal")
    while True:
        await asyncio.sleep(interval)
        sentiment: float | None = state.get("sentiment")

        # ------------------------------------------------------------------
        # [PRO] Advanced News Filter
        # ------------------------------------------------------------------
        # Record the current sentiment reading with its timestamp.
        now = datetime.now(tz=timezone.utc)
        sentiment_history: deque[tuple[datetime, float]] = state["sentiment_history"]
        # Only append to history once a real Gemini score is available.
        # When sentiment is None (first boot, before first Gemini call) we skip
        # the append so the deque stays empty and no artificial swing is triggered.
        if sentiment is not None:
            sentiment_history.append((now, sentiment))

            # Prune entries older than the 10-minute observation window.
            cutoff = now - timedelta(minutes=10)
            while sentiment_history and sentiment_history[0][0] < cutoff:
                sentiment_history.popleft()

            # Check for high-volatility sentiment fluctuation.
            hold_until: datetime | None = state.get("news_hold_until")
            if len(sentiment_history) >= 2:
                scores = [s for _, s in sentiment_history]
                if max(scores) - min(scores) > NEWS_FILTER_VOLATILITY_THRESHOLD:
                    new_hold_until = now + timedelta(minutes=NEWS_FILTER_HOLD_MINUTES)
                    # Do not keep extending HOLD forever while volatility remains high.
                    # Only open a new HOLD window when there is no active one.
                    if hold_until is None or now >= hold_until:
                        state["news_hold_until"] = new_hold_until
                        logger.info(
                            "[PRO] News Filter triggered – sentiment swing %.4f > %.2f "
                            "in the last 10 min. Global HOLD until %s.",
                            max(scores) - min(scores),
                            NEWS_FILTER_VOLATILITY_THRESHOLD,
                            new_hold_until.isoformat(),
                        )
                    else:
                        logger.debug(
                            "[PRO] News Filter volatility still high (swing=%.4f) but HOLD already active until %s.",
                            max(scores) - min(scores),
                            hold_until.isoformat(),
                        )

        # Honour the active HOLD period: skip all signal evaluation.
        hold_until = state.get("news_hold_until")
        if hold_until is not None and now < hold_until:
            remaining = int((hold_until - now).total_seconds() / 60)
            logger.debug(
                "[PRO] News Filter active – global HOLD in effect (%d min remaining).",
                remaining,
            )
            continue
        elif hold_until is not None and now >= hold_until:
            # Clear the expired HOLD.
            state["news_hold_until"] = None

        for symbol in watchlist:
            prices: list[float] = list(state["prices"].get(symbol, []))

            # Resolve the effective sentiment score for signal generation.
            # Use neutral 0.0 until the first real Gemini reading is available.
            effective_sentiment: float = sentiment if sentiment is not None else 0.0

            if len(prices) < 50:
                logger.debug(
                    "⏳ [AI WARMUP] %s – Recopilando datos... (%d/50 ticks necesarios)",
                    symbol,
                    len(prices),
                )
                continue

            # [ELITE] Gather regime and funding inputs
            highs: list[float] = list(state.get("highs", {}).get(symbol, []))
            lows: list[float] = list(state.get("lows", {}).get(symbol, []))
            obi_ratio: float = state.get("obi_ratios", {}).get(symbol, 1.0)
            funding_rate: float = state.get("funding_rates", {}).get(symbol, 0.0)

            # [ATR] Compute ATR_14 from the 15m OHLCV buffers and cache in state
            current_atr: float | None = FeatureEngineer.compute_atr(highs, lows, prices)
            if current_atr is not None:
                state.setdefault("atrs", {})[symbol] = current_atr
                logger.debug(
                    "📐 [ATR] %s – ATR_14=%.4f", symbol, current_atr
                )
            else:
                current_atr = state.get("atrs", {}).get(symbol)

            # [MTA] Compute higher-timeframe trend statuses
            closes_4h: list[float] = list(
                state.get("htf_closes", {}).get(symbol, {}).get("4h", [])
            )
            opens_4h: list[float] = list(
                state.get("htf_opens", {}).get(symbol, {}).get("4h", [])
            )
            closes_1h: list[float] = list(
                state.get("htf_closes", {}).get(symbol, {}).get("1h", [])
            )
            opens_1h: list[float] = list(
                state.get("htf_opens", {}).get(symbol, {}).get("1h", [])
            )

            trend_4h = compute_htf_trend(closes_4h, opens_4h or None)
            trend_1h = compute_htf_trend(closes_1h, opens_1h or None)
            trend_15m = compute_htf_trend(prices, None)

            # Update shared state and persist to DB
            state.setdefault("htf_trend", {}).setdefault(symbol, {})
            state["htf_trend"][symbol] = {
                "4h": trend_4h,
                "1h": trend_1h,
                "15m": trend_15m,
            }
            for tf, trend in (("4h", trend_4h), ("1h", trend_1h), ("15m", trend_15m)):
                try:
                    await db.upsert_htf_trend(symbol, tf, trend)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[MTA] DB upsert_htf_trend failed for %s/%s: %s", symbol, tf, exc)

            logger.debug(
                "🔭 [MTA RADAR] %s – 4H=%s | 1H=%s | 15M=%s",
                symbol, trend_4h, trend_1h, trend_15m,
            )

            signal = predictor.generate_signal(
                prices,
                effective_sentiment,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
                funding_rate=funding_rate,
                htf_trend_4h=trend_4h,
                htf_trend_1h=trend_1h,
            )
            win_prob: float = predictor.predict_proba(
                prices,
                effective_sentiment,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
            ) or 0.0

            # Store the latest ML confidence so dashboard_logger can display it.
            state["ml_probs"][symbol] = win_prob

            logger.debug(
                "🧠 [AI THOUGHT] %s – Signal: %s | Confidence: %.2f%% | Prices in buffer: %d | Sentiment: %.4f",
                symbol,
                signal,
                win_prob * 100,
                len(prices),
                effective_sentiment,
            )

            # ------------------------------------------------------------------
            # [SMART EXIT] Layers 1 + 4 – ML Exhaustion & TTL
            # ------------------------------------------------------------------
            # For every open position evaluate whether the current ML signal
            # warrants an early exit (trend reversal) or the TTL has elapsed.
            # This runs in the same signal-emitter cycle (every ~15 s) so it
            # never blocks the event loop with additional network calls.
            if symbol in paper_executor.open_positions and prices:
                q_live = state.get("mt5_last_quote", {}).get(symbol)
                if isinstance(q_live, dict) and float(q_live.get("mid") or 0.0) > 0.0:
                    current_price = float(q_live["mid"])
                else:
                    current_price = prices[-1]
                try:
                    smart_pnl = await paper_executor.check_ml_exit(
                        current_price=current_price,
                        ml_signal=signal,
                        ml_probability=win_prob if win_prob > 0.0 else None,
                        symbol=symbol,
                        min_confidence=BUY_PROB_THRESHOLD,
                    )
                    if smart_pnl is not None:
                        logger.debug(
                            "Smart exit triggered  symbol=%s  pnl=%.4f  total_pnl=%.4f",
                            symbol,
                            smart_pnl,
                            paper_executor.total_pnl,
                        )
                        # Position was closed – skip the BUY entry logic below.
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "⚠️ [ALERTA] check_ml_exit failed for %s: %s %s",
                        symbol,
                        exc,
                        DEBUG_LOG_HINT,
                    )

            if signal == "BUY" and prices:
                q_live = state.get("mt5_last_quote", {}).get(symbol)
                if isinstance(q_live, dict) and float(q_live.get("ask") or 0.0) > 0.0:
                    entry_price = float(q_live["ask"])
                else:
                    entry_price = prices[-1]
                try:
                    opened = await paper_executor.try_open_trade(
                        entry_price=entry_price,
                        win_probability=win_prob,
                        symbol=symbol,
                        sentiment_score=effective_sentiment,
                        current_atr=current_atr,
                    )
                    if not opened:
                        logger.debug(
                            "BUY signal ignored for %s – position already open, "
                            "max positions reached, or insufficient balance.",
                            symbol,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("⚠️ [ALERTA] Paper trade open failed for %s: %s %s", symbol, exc, DEBUG_LOG_HINT)

            elif signal == "SELL" and symbol in paper_executor.open_positions and prices:
                # Long-only bot: SELL never opens a short. Exits are driven by the
                # smart-exit block above (same ML signal passed to check_ml_exit).
                # This branch documents intent and aids debugging when the model
                # flips bearish but the confidence gate leaves the trade open.
                logger.debug(
                    "[LONG_ONLY] Model SELL for %s — exit relies on smart-exit/TTL/SL "
                    "(check_ml_exit already evaluated this cycle).",
                    symbol,
                )
