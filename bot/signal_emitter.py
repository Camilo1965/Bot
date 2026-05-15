"""Periodic ML signal evaluation and trade/open/exit orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

import aiohttp

import aiohttp

from bot.constants import DEBUG_LOG_HINT
from database.db_manager import db
from execution.paper_executor import PaperExecutor
from strategy.feature_engineer import FeatureEngineer
from strategy.quant_features import MIN_OHLC_ROWS
from strategy.ml_predictor import BUY_PROB_THRESHOLD, MLPredictor, get_symbol_config, model_json_path_for_symbol
from strategy.regime_predictor import predict_regime
from bot.observability import SignalMetrics, TradeMetrics, ExitMetrics, get_buffer
from vibe.decision_bridge import (
    check_vibe_force_exit,
    compute_entry_score,
    compute_quality_multiplier,
    is_extreme_macro_veto,
)
from vibe.feature_bridge import extract_vibe_features

_VIBE_MCP_URL: str = os.environ.get("VIBE_MCP_URL", "http://localhost:5000/predict")
_VIBE_MCP_TIMEOUT: float = float(os.environ.get("VIBE_MCP_TIMEOUT_S", "30"))
_VIBE_MCP_ENABLED: bool = os.environ.get("VIBE_MCP_ENABLED", "1").strip().lower() in ("1", "true", "yes")

async def _fetch_vibe_mcp_decision(
    session: aiohttp.ClientSession,
    features: list[float],
    ohlcv: list[dict[str, Any]],
    symbol: str,
) -> dict[str, Any] | None:
    payload = {
        "features": features,
        "ohlcv": ohlcv,
        "symbol": symbol,
    }
    try:
        async with session.post(
            _VIBE_MCP_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=_VIBE_MCP_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            logging.getLogger("clawdbot.signal").warning(
                "[VIBE MCP] HTTP %d: %s", resp.status, text[:200]
            )
            return None
    except asyncio.TimeoutError:
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Timeout after %.0fs", _VIBE_MCP_TIMEOUT
        )
        return None
    except Exception as exc:
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Request failed: %s", exc
        )
        return None


_VIBE_MCP_URL: str = os.environ.get("VIBE_MCP_URL", "http://localhost:5000/predict")
_VIBE_MCP_TIMEOUT: float = float(os.environ.get("VIBE_MCP_TIMEOUT_S", "30"))
_VIBE_MCP_ENABLED: bool = os.environ.get("VIBE_MCP_ENABLED", "1").strip().lower() in ("1", "true", "yes")

async def _fetch_vibe_mcp_decision(
    session: aiohttp.ClientSession,
    features: list[float],
    ohlcv: list[dict[str, Any]],
    symbol: str,
) -> dict[str, Any] | None:
    """Send features + OHLCV to the VIBE MCP server and return the JSON decision."""
    payload = {
        "features": features,
        "ohlcv": ohlcv,
        "symbol": symbol,
    }
    try:
        async with session.post(
            _VIBE_MCP_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=_VIBE_MCP_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            logging.getLogger("clawdbot.signal").warning(
                "[VIBE MCP] HTTP %d: %s", resp.status, text[:200]
            )
            return None
    except asyncio.TimeoutError:
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Timeout after %.0fs", _VIBE_MCP_TIMEOUT
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Request failed: %s", exc
        )
        return None

_VIBE_MCP_URL: str = os.environ.get("VIBE_MCP_URL", "http://localhost:5000/predict")
_VIBE_MCP_TIMEOUT: float = float(os.environ.get("VIBE_MCP_TIMEOUT_S", "30"))
_VIBE_MCP_ENABLED: bool = os.environ.get("VIBE_MCP_ENABLED", "0").strip().lower() in ("1", "true", "yes")


async def _fetch_vibe_mcp_decision(
    session: aiohttp.ClientSession,
    features: list[float],
    ohlcv: list[dict[str, Any]],
    symbol: str,
) -> dict[str, Any] | None:
    """Send features + OHLCV to the VIBE MCP server and return the JSON decision."""
    payload = {
        "features": features,
        "ohlcv": ohlcv,
        "symbol": symbol,
    }
    try:
        async with session.post(
            _VIBE_MCP_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=_VIBE_MCP_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            logging.getLogger("clawdbot.signal").warning(
                "[VIBE MCP] HTTP %d: %s", resp.status, text[:200]
            )
            return None
    except asyncio.TimeoutError:
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Timeout after %.0fs", _VIBE_MCP_TIMEOUT
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Request failed: %s", exc
        )
        return None


def _extract_vibe_pattern_text(data: Any) -> str:
    """Extract readable text from a Vibe pattern recognition result."""
    try:
        if isinstance(data, dict) and "content" in data:
            parts = []
            for item in data["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
            text = " ".join(parts).strip()
            return text[:200] if text else ""
        return str(data)[:200]
    except Exception:
        return str(data)[:200]

def _swarm_adjusted_threshold(base_threshold: float, state: dict[str, Any], symbol: str) -> float:
    """Adjust probability threshold based on swarm recommendation (Fase 5).

    * STRONG_BUY  → threshold * 0.9  (easier entry)
    * BUY         → threshold * 0.95 (slightly easier)
    * REDUCE      → threshold * 1.2  (harder entry)
    * EXIT        → threshold * 1.5  (much harder entry)
    * NEUTRAL     → no change

    Bounds: never below 0.50, never above 0.99.
    """
    swarm_rec = state.get("vibe_swarm_recommendation", {}).get(symbol, "NEUTRAL")
    multipliers = {
        "STRONG_BUY": 0.90,
        "BUY": 0.95,
        "NEUTRAL": 1.0,
        "REDUCE": 1.2,
        "EXIT": 1.5,
    }
    mult = multipliers.get(swarm_rec, 1.0)
    adjusted = base_threshold * mult
    return max(0.50, min(0.99, adjusted))


def _apply_vibe_gating(
    state: dict[str, Any],
    symbol: str,
    signal: str,
    win_prob: float,
) -> tuple[str, str | None]:
    """Apply VIBE Phase-1 probabilistic hybrid gating to a BUY signal.

    Returns ``(possibly_modified_signal, reason_or_None)``.  When the
    returned signal is ``"HOLD"`` the caller should skip the trade-open
    path but still let the smart-exit logic run for existing positions.
    """
    if signal != "BUY":
        return signal, None

    _vibe_gate = os.environ.get("VIBE_GATE_TRADES", "0").strip()
    if _vibe_gate not in ("1", "true", "yes"):
        return signal, None

    # a) Extreme macro veto (absolute)
    if is_extreme_macro_veto(state):
        return "HOLD", "VIBE_EXTREME_VETO"

    # b-d) Probabilistic modulation
    vibe_score = compute_entry_score(state, symbol)
    cfg = get_symbol_config(symbol)

    # [VIBE FASE 5] Swarm-adjusted threshold
    base_th = float(cfg["prob_threshold"])
    effective_th = _swarm_adjusted_threshold(base_th, state, symbol)
    effective_prob = win_prob * vibe_score

    if effective_prob < effective_th:
        return "HOLD", f"VIBE_SCORE_{vibe_score:.2f}"

    return signal, None


# BUY gate + ML-reversal min confidence = BUY_PROB_THRESHOLD (default 0.50 max-performance).

async def signal_emitter(
    state: dict[str, Any],
    predictor: MLPredictor,
    paper_executor: PaperExecutor,
    watchlist: list[str],
    interval: int = 15,
) -> None:
    logger = logging.getLogger("clawdbot.signal")
    http_session = aiohttp.ClientSession()
    try:
        while True:
            await asyncio.sleep(interval)
            now = datetime.now(tz=timezone.utc)
    
            for symbol in watchlist:
                prices: list[float] = list(state["prices"].get(symbol, []))

            if len(prices) < MIN_OHLC_ROWS:
                logger.debug(
                    "⏳ [AI WARMUP] %s - Recopilando datos... (%d/%d barras necesarias)",
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
                    "📐 [ATR] %s - ATR_14=%.4f", symbol, current_atr
                )
            else:
                current_atr = state.get("atrs", {}).get(symbol)

            # [VIBE FASE 4] Extraer features numéricas para el modelo XGBoost
            vibe_vec = None
            if os.environ.get("VIBE_FEATURES_ENABLED") == "1":
                vibe_vec = extract_vibe_features(state, symbol)

            # [VIBE MCP SERVER]
            if _VIBE_MCP_ENABLED:
                ohlcv_list = []
                n = min(len(prices), len(highs) if highs else len(prices), len(lows) if lows else len(prices), len(volumes) if volumes else len(prices))
                compress_ohlcv = n > 200
                for i in range(n):
                    o_val = prices[i]
                    v_val = volumes[i] if volumes else 0.0
                    if compress_ohlcv:
                        ohlcv_list.append({"close": o_val, "volume": v_val})
                    else:
                        h_val = highs[i] if highs else o_val
                        l_val = lows[i] if lows else o_val
                        ohlcv_list.append({"open": o_val, "high": h_val, "low": l_val, "close": o_val, "volume": v_val})
                
                q_feat = predictor._compute_features(
                    prices, 0.0, highs=highs or None, lows=lows or None,
                    obi_ratio=obi_ratio, volumes=volumes or None, symbol=symbol
                )
                full_features = (q_feat or []) + (vibe_vec or [0.0, 0.0, 1.0, 0.5])
                
                # Fire-and-forget with safe exception logging
                async def _safe_vibe_fetch() -> None:
                    try:
                        await _fetch_vibe_mcp_decision(http_session, full_features, ohlcv_list, symbol)
                    except Exception as _vibe_exc:
                        logger.debug("[VIBE MCP] background fetch failed: %s", _vibe_exc)

                asyncio.create_task(_safe_vibe_fetch())

            # Single-pass prediction
            _t0_pred = time.time()
            signal, win_prob = predictor.generate_signal(
                prices,
                0.0,
                highs=highs or None,
                lows=lows or None,
                obi_ratio=obi_ratio,
                funding_rate=funding_rate,
                volumes=volumes or None,
                symbol=symbol,
                vibe_features=vibe_vec,
            )
            _pred_latency_ms = (time.time() - _t0_pred) * 1000.0
            state["ml_signals"][symbol] = signal
            # Store the latest ML confidence so dashboard_logger can display it.
            state["ml_probs"][symbol] = win_prob

            # Observability: emit signal metrics
            get_buffer().emit_signal(
                SignalMetrics(
                    symbol=symbol,
                    timestamp=time.time(),
                    latency_ms=round(_pred_latency_ms, 2),
                    probability=round(win_prob, 4),
                    signal=signal,
                    features_valid=True,
                    model_used=model_json_path_for_symbol(symbol).name,
                )
            )

            # [VIBE] Log pattern recognition overlay (INFORMATIONAL - does NOT gate entries)
            vibe_pattern = state.get("vibe_patterns", {}).get(symbol)
            if vibe_pattern:
                _pdesc = _extract_vibe_pattern_text(vibe_pattern)
                if _pdesc:
                    logger.warning("🔍 [VIBE] %s pattern: %s", symbol, _pdesc)

            logger.debug(
                "🧠 [AI THOUGHT] %s - Signal: %s | Confidence: %.2f%% | Prices in buffer: %d",
                symbol,
                signal,
                win_prob * 100,
                len(prices),
            )

            # ------------------------------------------------------------------
            # [SMART EXIT] Layers 1 + 4 - ML Exhaustion & TTL
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

                # ------------------------------------------------------------------
                # [VIBE FASE 3] Exit Augmentation - EMERGENCY (antes que ML exit)
                # ------------------------------------------------------------------
                if os.environ.get("VIBE_FORCE_EXIT") == "1":
                    vibe_reason = check_vibe_force_exit(state, symbol)
                    if vibe_reason:
                        logger.warning(
                            "[VIBE] EXIT FORZADO en %s por patrón crítico: %s",
                            symbol,
                            vibe_reason,
                        )
                        forced_pnl = await paper_executor._close_position(
                            symbol,
                            current_price,
                            f"VIBE_EXIT_{vibe_reason}",
                        )
                        logger.info(
                            "[VIBE] Posición %s cerrada forzosamente. PnL=%.4f",
                            symbol,
                            forced_pnl if forced_pnl is not None else 0.0,
                        )
                        # Skip ML exit and BUY entry logic for this cycle
                        continue

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
                        # Position was closed - skip the BUY entry logic below.
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "⚠️ [ALERTA] check_ml_exit failed for %s: %s %s",
                        symbol,
                        exc,
                        DEBUG_LOG_HINT,
                    )

                # [REGIME] Close open positions if market enters ranging regime
                if os.environ.get("REGIME_FILTER_ENABLED", "1").strip() in ("1", "true", "yes"):
                    regime_result = predict_regime(symbol, prices, highs=highs, lows=lows, volumes=volumes)
                    if regime_result is not None:
                        regime, regime_prob = regime_result
                        if regime == "RANGING":
                            logger.warning(
                                "[REGIME] Closing %s position: market entered RANGING (prob=%.2f)",
                                symbol, regime_prob,
                            )
                            close_pnl = await paper_executor._close_position(
                                symbol, current_price, "REGIME_RANGING"
                            )
                            logger.info(
                                "[REGIME] Position %s closed. PnL=%.4f", symbol, close_pnl or 0.0
                            )
                            continue

            # ------------------------------------------------------------------
            # [REGIME] Filter new entries by market regime
            # ------------------------------------------------------------------
            regime_gate_passed = True
            regime_str = "UNKNOWN"
            regime_prob = 0.0
            if os.environ.get("REGIME_FILTER_ENABLED", "1").strip() in ("1", "true", "yes"):
                regime_result = predict_regime(symbol, prices, highs=highs, lows=lows, volumes=volumes)
                if regime_result is not None:
                    regime_str, regime_prob = regime_result
                    if regime_str == "RANGING":
                        regime_gate_passed = False
                        logger.info(
                            "[REGIME] Entry blocked for %s: market is RANGING (prob=%.2f)",
                            symbol, regime_prob,
                        )
                else:
                    logger.debug("[REGIME] No regime model available for %s", symbol)

            # ------------------------------------------------------------------
            # [VIBE FASE 1] Gating de entradas - Veto Probabilístico Híbrido
            # ------------------------------------------------------------------
            # Se evalúa DESPUÉS del smart-exit para que posiciones abiertas
            # sigan gestionándose independientemente del gating de nuevas entradas.
            gated_signal, gate_reason = _apply_vibe_gating(state, symbol, signal, win_prob)
            if gated_signal != signal:
                if gate_reason == "VIBE_EXTREME_VETO":
                    logger.warning(
                        "[VIBE] VETO ABSOLUTO: Riesgo macro extremo detectado. %s bloqueado.",
                        symbol,
                    )
                else:
                    logger.info(
                        "[VIBE] Entrada bloqueada para %s: prob %.2f reducida por "
                        "contexto macro/técnico (%s)",
                        symbol,
                        win_prob,
                        gate_reason,
                    )
                signal = gated_signal

            if signal == "BUY" and prices and regime_gate_passed:
                # [VIBE FASE 2] Sizing dinámico por calidad del setup
                _vibe_adj = os.environ.get("VIBE_ADJUST_SIZE", "1").strip()
                if _vibe_adj not in ("1", "true", "yes"):
                    vibe_quality = 1.0
                else:
                    vibe_quality = compute_quality_multiplier(state, symbol)
                    if vibe_quality != 1.0:
                        logger.info(
                            "[VIBE] Sizing ajustado para %s: quality=%.2f",
                            symbol, vibe_quality,
                        )

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
                        vibe_quality=vibe_quality,
                    )
                    if not opened:
                        logger.debug(
                            "BUY signal ignored for %s - position already open, "
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
                    "[LONG_ONLY] Model SELL for %s - exit relies on smart-exit/TTL/SL "
                    "(check_ml_exit already evaluated this cycle).",
                    symbol,
                )
    finally:
        await http_session.close()
