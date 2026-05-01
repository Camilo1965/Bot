"""Background asyncio loops: dashboard refresh, sync, health, reconciler."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from collections import deque
from zoneinfo import ZoneInfo

from rich.live import Live

from database.db_manager import db

from bot.constants import DEBUG_LOG_HINT
from bot.dashboard import generate_dashboard
from bot.dashboard_helpers import mt5_dashboard_mark

from execution.mt5_executor import MT5Executor, fetch_mt5_wallet_snapshot
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager

from utils.telegram_notifier import (
    send_priority_telegram_alert,
)


async def dashboard_logger(
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    state: dict[str, Any],
    live: Live,
    watchlist: list[str],
    interval: int = 1,
) -> None:
    """Refresh the Rich Live dashboard every *interval* seconds.

    On each tick the function:

    * Calls :func:`generate_dashboard` to build a fresh :class:`~rich.table.Table`
      and pushes it to the running :class:`~rich.live.Live` context so the
      terminal display is updated in-place without any scrolling.
    * Writes per-position risk telemetry (trailing stop, ATR, high-watermark)
      exclusively to ``audit.log`` via the ``clawdbot.audit`` logger – this
      output never reaches the console.
    * Emits a DEBUG heartbeat to ``bot_debug.log`` for post-session analysis.

    Important log events (errors, trade opens/closes) continue to reach the
    console via the :class:`~rich.logging.RichHandler` that is attached to the
    root logger in ``main``, which renders them above the live table.
    """
    logger_dash = logging.getLogger("clawdbot.dashboard")
    audit = logging.getLogger("clawdbot.audit")
    while True:
        await asyncio.sleep(interval)

        if isinstance(paper_executor, MT5Executor) and paper_executor._live:
            snap = fetch_mt5_wallet_snapshot()
            if snap:
                state["mt5_wallet"] = snap

        # ── Refresh the live TUI table ────────────────────────────────────────
        live.update(generate_dashboard(state, paper_executor, risk_manager, watchlist))

        # ── Detailed heartbeat → bot_debug.log only (DEBUG) ──────────────────
        n_positions = len(paper_executor.open_positions)
        ml_probs: dict[str, float] = state.get("ml_probs", {})
        w = state.get("mt5_wallet")
        bal_log = float(w["balance"]) if isinstance(w, dict) and "balance" in w else risk_manager.balance
        logger_dash.debug(
            "📊 [DASHBOARD] Total PnL: %.4f USDT  |  Balance: %.2f USDT  |  "
            "Open positions: %d/%d  |  Symbols: %s",
            paper_executor.total_pnl,
            bal_log,
            n_positions,
            risk_manager.max_positions,
            list(paper_executor.open_positions.keys()) or "none",
        )

        # ── Audit telemetry: per-position risk data → audit.log only ─────────
        hora = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        for sym, pos in paper_executor.open_positions.items():
            prices_buf = state["prices"].get(sym, [])
            if not prices_buf:
                continue
            mark_price = mt5_dashboard_mark(state, sym, float(prices_buf[-1])) or float(prices_buf[-1])
            if pos.entry_price == 0.0:
                continue
            # Use the live ratcheted stop stored on the position object – this
            # is updated by check_and_close() on every price tick, so it always
            # reflects the current in-memory value rather than a recomputed
            # snapshot that could diverge from what the exit logic actually uses.
            stop_calculated: float = pos.current_stop_loss
            atr_val: float | None = state.get("atrs", {}).get(sym)
            # DISTANCIA_% = ((PRECIO_ACTUAL - STOP_CALCULADO) / PRECIO_ACTUAL) * 100
            distancia_pct = (
                (mark_price - stop_calculated) / mark_price * 100
                if mark_price > 0
                else 0.0
            )
            atr_str = f"{atr_val:.4f}" if atr_val is not None else "N/A"
            xgb_conf: float = ml_probs.get(sym, 0.0)
            audit.info(
                "%s | %-10s | %13.4f | %11.4f | %8s | %14.4f | %8.4f | %11.2f%%",
                hora,
                sym,
                mark_price,
                pos.peak_price,
                atr_str,
                stop_calculated,
                xgb_conf,
                distancia_pct,
            )

            # ── Track max drawdown across all open positions ──────────────────
            qty = pos.position_size / pos.entry_price
            unrealized = (mark_price - pos.entry_price) * qty
            current_dd = state.get("max_drawdown", 0.0)
            if unrealized < current_dd:
                state["max_drawdown"] = unrealized


async def position_sync_loop(
    paper_executor: PaperExecutor,
    interval: int = 60,
) -> None:
    """Periodically reconcile local position state with the live exchange.

    On each tick the coroutine:

    1. Logs ``[SISTEMA] 🔄 Sincronizando posiciones…``
    2. Calls :meth:`~execution.paper_executor.PaperExecutor.sync_positions_with_exchange`
       which removes any *ghost* positions (in memory but gone on the exchange) and
       cancels their orphan orders.
    3. Logs ``[SISTEMA] ✅ Sincronización completa. Posiciones reales: {count}``.

    When the executor is running in pure paper-trading mode (no live exchange
    client and not a live MT5Executor) the coroutine exits immediately so it
    never occupies a task slot needlessly.
    """
    is_live_mt5 = isinstance(paper_executor, MT5Executor) and paper_executor._live
    if paper_executor._exchange is None and not is_live_mt5:
        return  # Nothing to sync in pure paper-trading mode.

    logger = logging.getLogger("clawdbot.sync")
    while True:
        await asyncio.sleep(interval)
        logger.debug("[SISTEMA] 🔄 Sincronizando posiciones con el exchange...")
        try:
            real_count = await paper_executor.sync_positions_with_exchange()
            log_fn = logger.debug if real_count == 0 else logger.info
            log_fn(
                "[SISTEMA] ✅ Sincronización completa. Posiciones reales: %d.",
                real_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "⚠️ [ALERTA] Position sync failed: %s %s",
                exc,
                DEBUG_LOG_HINT,
            )


async def health_monitor_loop(
    state: dict[str, Any],
    market_queue: asyncio.Queue[dict[str, Any]],
    paper_executor: PaperExecutor,
    interval: int = 30,
) -> None:
    """Periodic health checks for feed freshness and processing backlog."""
    logger = logging.getLogger("clawdbot.health")
    stale_alert_sent = False
    backlog_alert_sent = False
    pending_alert_streak = 0
    while True:
        await asyncio.sleep(interval)
        now = datetime.now(tz=timezone.utc)
        last_msg: datetime | None = state.get("last_market_message_at")
        queue_size = market_queue.qsize()
        open_positions = len(paper_executor.open_positions)
        stale_seconds: float | None = None
        if last_msg is not None:
            stale_seconds = (now - last_msg).total_seconds()

        logger.debug(
            "[HEALTH] queue=%d open_positions=%d stale_seconds=%s",
            queue_size,
            open_positions,
            f"{stale_seconds:.1f}" if stale_seconds is not None else "n/a",
        )

        if queue_size > 2000:
            logger.warning(
                "[HEALTH] market queue backlog is high (%d messages). "
                "Consumer may be lagging.",
                queue_size,
            )
            if not backlog_alert_sent:
                backlog_alert_sent = True
                try:
                    await db.insert_health_event(
                        level="WARNING",
                        component="health_monitor",
                        event_code="QUEUE_BACKLOG_HIGH",
                        message=f"Queue backlog high: {queue_size}",
                        payload_json=json.dumps({"queue_size": queue_size}),
                    )
                except Exception:
                    pass
        else:
            backlog_alert_sent = False
        if stale_seconds is not None and stale_seconds > 120:
            logger.warning(
                "[HEALTH] market feed appears stale (%.1fs without messages).",
                stale_seconds,
            )
            if not stale_alert_sent:
                stale_alert_sent = True
                try:
                    await db.insert_health_event(
                        level="WARNING",
                        component="health_monitor",
                        event_code="MARKET_FEED_STALE",
                        message=f"Market feed stale for {stale_seconds:.1f}s",
                        payload_json=json.dumps({"stale_seconds": stale_seconds}),
                    )
                except Exception:
                    pass
                asyncio.create_task(
                    send_priority_telegram_alert(
                        "⚠️ *HEALTH ALERT* Feed de mercado sin mensajes recientes (>120s)."
                    , priority="critical", dedup_key="health:market_feed_stale")
                )
        else:
            stale_alert_sent = False

        pending_count = sum(
            1 for p in paper_executor.open_positions.values() if getattr(p, "close_pending", False)
        )
        if pending_count > 0:
            pending_alert_streak += 1
            if pending_alert_streak >= 3:
                pending_alert_streak = 0
                details = []
                for sym, pos in paper_executor.open_positions.items():
                    if getattr(pos, "close_pending", False):
                        details.append(f"{sym}:{getattr(pos, 'last_close_error', 'unknown')}")
                try:
                    await db.insert_health_event(
                        level="ERROR",
                        component="reconciler",
                        event_code="PENDING_CLOSE_PERSISTENT",
                        message=f"Pending closes persist: {pending_count}",
                        payload_json=json.dumps({"pending": details}),
                    )
                except Exception:
                    pass
                asyncio.create_task(
                    send_priority_telegram_alert(
                        "🚨 *ALERTA DEL SISTEMA: ÓRDENES DE CIERRE RECHAZADAS*\n\n"
                        f"El bot ha intentado cerrar *{pending_count} posición(es)* repetidamente, pero el bróker (MT5) está rechazando las órdenes.\n\n"
                        "🛑 *¿Qué significa esto?*\n"
                        "El bot ya decidió que la operación debe cerrarse (por toma de ganancias o stop loss), pero MT5 no procesa la solicitud.\n\n"
                        "⚠️ *Acción recomendada:*\n"
                        "Revisa la terminal de MT5 manualmente para confirmar si la orden sigue abierta o si ya se cerró. Si sigue abierta, es posible que debas cerrarla a mano.\n\n"
                        "*Detalles técnicos (Símbolo : Error):*\n"
                        + "\n".join(f"• `{d}`" for d in details[:5]),
                        priority="critical",
                        dedup_key="health:pending_close_persistent",
                    )
                )
        else:
            pending_alert_streak = 0


async def close_pending_reconciler_loop(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    interval: int = 20,
) -> None:
    """Retry loop for positions marked as close_pending."""
    logger = logging.getLogger("clawdbot.reconciler")
    while True:
        await asyncio.sleep(interval)
        pending = [
            sym
            for sym, pos in paper_executor.open_positions.items()
            if getattr(pos, "close_pending", False)
        ]
        if not pending:
            continue
        latest_prices: dict[str, float] = {}
        prices_state: dict[str, deque[float]] = state.get("prices", {})
        for sym in pending:
            buf = prices_state.get(sym, deque())
            if buf:
                latest_prices[sym] = (
                    mt5_dashboard_mark(state, sym, float(buf[-1])) or float(buf[-1])
                )
        if not latest_prices:
            logger.debug("[RECONCILER] pending closes exist but no latest prices available.")
            continue
        try:
            closed = await paper_executor.retry_close_pending_positions(latest_prices)
            if closed > 0:
                logger.info(
                    "[RECONCILER] Successfully closed %d pending position(s).",
                    closed,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RECONCILER] retry loop failed: %s", exc)


def _report_tz() -> ZoneInfo:
    tz_name = os.environ.get("REPORT_TIMEZONE", "America/Bogota").strip() or "America/Bogota"
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("America/Bogota")


def _next_run_local(target_weekday: int | None, target_hour: int, target_minute: int = 0) -> datetime:
    """Compute next run datetime in UTC from local schedule."""
    tz = _report_tz()
    now_local = datetime.now(tz=tz)
    run_local = now_local.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target_weekday is not None:
        days_ahead = (target_weekday - now_local.weekday()) % 7
        run_local = run_local + timedelta(days=days_ahead)
        if run_local <= now_local:
            run_local = run_local + timedelta(days=7)
    else:
        if run_local <= now_local:
            if now_local.month == 12:
                run_local = run_local.replace(year=now_local.year + 1, month=1, day=1)
            else:
                run_local = run_local.replace(month=now_local.month + 1, day=1)
        else:
            run_local = run_local.replace(day=1)
    return run_local.astimezone(timezone.utc)


async def weekly_report_loop(interval_floor: int = 30) -> None:
    """Send automatic weekly Telegram summary in configured timezone."""
    logger = logging.getLogger("clawdbot.reports.weekly")
    from utils.telegram_notifier import _build_period_report  # local import to avoid circular load

    weekly_hour = int(os.environ.get("REPORT_WEEKLY_HOUR_LOCAL", "21"))
    weekly_minute = int(os.environ.get("REPORT_WEEKLY_MINUTE_LOCAL", "0"))
    weekly_weekday = int(os.environ.get("REPORT_WEEKLY_DAY_LOCAL", "6"))  # 0=Mon ... 6=Sun
    while True:
        next_run_utc = _next_run_local(
            target_weekday=weekly_weekday,
            target_hour=weekly_hour,
            target_minute=weekly_minute,
        )
        sleep_s = max(interval_floor, int((next_run_utc - datetime.now(tz=timezone.utc)).total_seconds()))
        logger.info("Weekly report scheduled for %s UTC (%ds).", next_run_utc.isoformat(), sleep_s)
        await asyncio.sleep(sleep_s)
        try:
            tz_name = str(_report_tz())
            report = await _build_period_report("week", tz_name)
            await send_priority_telegram_alert(
                report,
                priority="summary",
                dedup_key=f"auto:weekly:{datetime.now(tz=_report_tz()).strftime('%Y-%W')}",
                force=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Weekly report failed: %s", exc)
        await asyncio.sleep(interval_floor)


async def monthly_report_loop(interval_floor: int = 30) -> None:
    """Send automatic monthly Telegram summary in configured timezone."""
    logger = logging.getLogger("clawdbot.reports.monthly")
    from utils.telegram_notifier import _build_period_report  # local import to avoid circular load

    monthly_hour = int(os.environ.get("REPORT_MONTHLY_HOUR_LOCAL", "21"))
    monthly_minute = int(os.environ.get("REPORT_MONTHLY_MINUTE_LOCAL", "0"))
    while True:
        next_run_utc = _next_run_local(
            target_weekday=None,
            target_hour=monthly_hour,
            target_minute=monthly_minute,
        )
        sleep_s = max(interval_floor, int((next_run_utc - datetime.now(tz=timezone.utc)).total_seconds()))
        logger.info("Monthly report scheduled for %s UTC (%ds).", next_run_utc.isoformat(), sleep_s)
        await asyncio.sleep(sleep_s)
        try:
            tz_name = str(_report_tz())
            report = await _build_period_report("month", tz_name)
            await send_priority_telegram_alert(
                report,
                priority="summary",
                dedup_key=f"auto:monthly:{datetime.now(tz=_report_tz()).strftime('%Y-%m')}",
                force=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monthly report failed: %s", exc)
        await asyncio.sleep(interval_floor)
