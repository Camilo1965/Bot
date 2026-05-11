"""Tests for bot.monitoring alert and health functions."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.monitoring import (
    check_daily_pnl_alert,
    check_stale_feed_alert,
    check_vibe_down_alert,
    check_open_positions_alert,
    get_health_metrics,
    _should_alert,
    _last_alert_at,
)
from execution.paper_executor import PaperExecutor, OpenPosition
from risk.risk_manager import RiskManager


@pytest.fixture(autouse=True)
def _reset_alert_state():
    """Clear deduplication state before every test."""
    _last_alert_at.clear()
    yield


def _make_state(last_msg_at: datetime | None = None) -> dict:
    return {
        "last_market_message_at": last_msg_at,
        "vibe_client": None,
    }


class TestShouldAlert:
    def test_first_alert_allowed(self):
        assert _should_alert("test_key", cooldown=1) is True

    def test_duplicate_within_cooldown_blocked(self):
        _should_alert("test_key2", cooldown=60)
        assert _should_alert("test_key2", cooldown=60) is False

    def test_after_cooldown_allowed(self):
        _should_alert("test_key3", cooldown=0.01)
        time.sleep(0.02)
        assert _should_alert("test_key3", cooldown=0.01) is True


class TestStaleFeedAlert:
    @pytest.mark.asyncio
    async def test_alert_when_stale(self):
        stale = datetime.now(tz=timezone.utc) - __import__("datetime").timedelta(seconds=200)
        state = _make_state(last_msg_at=stale)
        with patch("asyncio.create_task") as mock_task:
            await check_stale_feed_alert(state, threshold_s=90)
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_when_fresh(self):
        fresh = datetime.now(tz=timezone.utc) - __import__("datetime").timedelta(seconds=10)
        state = _make_state(last_msg_at=fresh)
        with patch("asyncio.create_task") as mock_task:
            await check_stale_feed_alert(state, threshold_s=90)
            mock_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_no_last_msg(self):
        state = _make_state(last_msg_at=None)
        with patch("asyncio.create_task") as mock_task:
            await check_stale_feed_alert(state, threshold_s=90)
            mock_task.assert_not_called()


class TestDailyPnlAlert:
    @pytest.mark.asyncio
    async def test_alert_when_loss_threshold_crossed(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        rm._daily_start_balance = 10_000.0
        rm._daily_loss = 200.0  # 2% loss
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        with patch("asyncio.create_task") as mock_task:
            await check_daily_pnl_alert(rm, executor, threshold_pct=0.015)
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_when_loss_below_threshold(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        rm._daily_start_balance = 10_000.0
        rm._daily_loss = 50.0  # 0.5% loss
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        with patch("asyncio.create_task") as mock_task:
            await check_daily_pnl_alert(rm, executor, threshold_pct=0.015)
            mock_task.assert_not_called()


class TestVibeDownAlert:
    @pytest.mark.asyncio
    async def test_alert_when_vibe_unavailable(self):
        client = MagicMock()
        client.available = False
        client.last_error = "timeout"
        state = {"vibe_client": client}
        with patch("asyncio.create_task") as mock_task:
            await check_vibe_down_alert(state)
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_when_vibe_available(self):
        client = MagicMock()
        client.available = True
        state = {"vibe_client": client}
        with patch("asyncio.create_task") as mock_task:
            await check_vibe_down_alert(state)
            mock_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_alert_when_no_vibe_client(self):
        state = {"vibe_client": None}
        with patch("asyncio.create_task") as mock_task:
            await check_vibe_down_alert(state)
            mock_task.assert_not_called()


class TestOpenPositionsAlert:
    @pytest.mark.asyncio
    async def test_alert_when_close_pending(self):
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        pos = OpenPosition(
            trade_id="t1", symbol="BTC/USDT",
            entry_time=datetime.now(tz=timezone.utc),
            entry_price=80000.0, position_size=1000.0,
            sl_price=78000.0, activation_price=81600.0,
            trailing_distance_pct=0.02, peak_price=80000.0,
            ml_confidence=0.7, sl_pct=0.025, activation_pct=0.02,
            stop_loss_price=78000.0, current_stop_loss=78000.0,
            close_pending=True, last_close_error="broker_timeout",
        )
        executor.open_positions["BTC/USDT"] = pos
        with patch("asyncio.create_task") as mock_task:
            await check_open_positions_alert(executor)
            mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_alert_when_no_pending(self):
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        with patch("asyncio.create_task") as mock_task:
            await check_open_positions_alert(executor)
            mock_task.assert_not_called()


class TestGetHealthMetrics:
    def test_returns_ok_when_fresh(self):
        state = {"last_market_message_at": datetime.now(tz=timezone.utc)}
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        class _Q:
            def qsize(self) -> int:
                return 5
        m = get_health_metrics(state, _Q(), executor, rm)
        assert m["status"] == "ok"
        assert m["queue_size"] == 5
        assert m["open_positions"] == 0
        assert m["vibe_available"] is False

    def test_returns_degraded_when_stale(self):
        stale = datetime.now(tz=timezone.utc) - __import__("datetime").timedelta(seconds=200)
        state = {"last_market_message_at": stale}
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        class _Q:
            def qsize(self) -> int:
                return 0
        m = get_health_metrics(state, _Q(), executor, rm)
        assert m["status"] == "degraded"
        assert m["stale_feed_seconds"] is not None
        assert m["stale_feed_seconds"] > 90

    def test_daily_loss_pct_present(self):
        state = {"last_market_message_at": datetime.now(tz=timezone.utc)}
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        rm._daily_loss = 150.0
        rm._daily_start_balance = 10_000.0
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        class _Q:
            def qsize(self) -> int:
                return 0
        m = get_health_metrics(state, _Q(), executor, rm)
        assert m["daily_loss_pct"] == pytest.approx(0.015, abs=0.001)
