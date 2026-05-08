"""Periodic ML signal evaluation and trade/open/exit orchestration."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from bot.constants import DEBUG_LOG_HINT
from database.db_manager import db
from execution.paper_executor import PaperExecutor
from strategy.feature_engineer import FeatureEngineer
from strategy.quant_features import MIN_OHLC_ROWS
from strategy.ml_predictor import BUY_PROB_THRESHOLD, MLPredictor, get_symbol_config

# BUY gate + ML-reversal min confidence = BUY_PROB_THRESHOLD (default 0.50 max-performance).

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
        now = datetime.now(tz=timezone.utc)

        for symbol in watchlist:
            prices: list[float] = list(state["prices"].get(symbol, []))

            if len(prices) < MIN_OHLC_ROWS:
                logger.debug(
                    "⏳ [AI WARMUP] %s – Recopilando datos... (%d/%d barras necesarias)",
                    symbol,
                    len(prices),
                    MIN_OHLC_ROWS,
                )
                continue

            # [ELITE] Gather regime and funding inputs
            highs: list[float] = list(state.get("highs", {}).get(symbol, []))
            lows: list[float] = list(state.get("lows", {}).get(symbol, []))
            volumes: list[float] = list(state.get("volumes", {}).get(symbol, []))
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

            # Single-pass prediction
            signal, win_prob = predictor.generate_signal(
                prices,
                0.0,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
                funding_rate=funding_rate,
                volumes=volumes or None,
                symbol=symbol,
            )
            state["ml_signals"][symbol] = signal
            # Store the latest ML confidence so dashboard_logger can display it.
            state["ml_probs"][symbol] = win_prob

            # [VIBE] Log pattern recognition overlay (INFORMATIONAL — does NOT gate entries)
            vibe_pattern = state.get("vibe_patterns", {}).get(symbol)
            if vibe_pattern:
                logger.info(
                    "🔍 [VIBE] %s pattern overlay: %s",
                    symbol,
                    str(vibe_pattern)[:120],
                )

            logger.debug(
                "🧠 [AI THOUGHT] %s – Signal: %s | Confidence: %.2f%% | Prices in buffer: %d",
                symbol,
                signal,
                win_prob * 100,
                len(prices),
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
                        min_confidence=float(get_symbol_config(symbol)["prob_threshold"]),
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
                        sentiment_score=0.0,
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
