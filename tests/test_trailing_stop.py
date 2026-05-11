"""Tests for trailing-stop logic."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.paper_executor import OpenPosition, PaperExecutor
from risk.risk_manager import RiskManager


def _make_pos(entry_price: float = 100.0, activation_pct: float = 0.02) -> OpenPosition:
    return OpenPosition(
        trade_id="t1",
        symbol="BTC/USDT",
        entry_time=datetime.now(tz=timezone.utc),
        entry_price=entry_price,
        position_size=1000.0,
        sl_price=entry_price * (1 - 0.025),
        activation_price=entry_price * (1 + activation_pct),
        trailing_distance_pct=0.02,
        peak_price=entry_price,
        ml_confidence=0.7,
        sl_pct=0.025,
        activation_pct=activation_pct,
        stop_loss_price=entry_price * (1 - 0.025),
        current_stop_loss=entry_price * (1 - 0.025),
    )


class TestTrailingStopLogic:
    @pytest.mark.asyncio
    async def test_trail_activates_at_exact_threshold(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        pos = _make_pos(entry_price=100.0, activation_pct=0.02)
        executor.open_positions["BTC/USDT"] = pos

        # price exactly at activation threshold
        await executor.check_and_close("BTC/USDT", 102.0)
        assert pos.trailing_stop_active is True

    @pytest.mark.asyncio
    async def test_sl_ratchets_up_when_peak_rises(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        pos = _make_pos(entry_price=100.0, activation_pct=0.02)
        executor.open_positions["BTC/USDT"] = pos

        await executor.check_and_close("BTC/USDT", 105.0)
        expected_sl = 105.0 * (1 - 0.02)
        assert pos.sl_price == pytest.approx(expected_sl)
        assert pos.current_stop_loss == pytest.approx(expected_sl)

    @pytest.mark.asyncio
    async def test_sl_never_moves_down(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        pos = _make_pos(entry_price=100.0, activation_pct=0.02)
        executor.open_positions["BTC/USDT"] = pos

        await executor.check_and_close("BTC/USDT", 105.0)
        old_sl = pos.sl_price
        await executor.check_and_close("BTC/USDT", 104.0)
        assert pos.sl_price == old_sl

    @pytest.mark.asyncio
    async def test_trail_false_below_threshold(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        pos = _make_pos(entry_price=100.0, activation_pct=0.02)
        executor.open_positions["BTC/USDT"] = pos

        await executor.check_and_close("BTC/USDT", 101.0)
        assert pos.trailing_stop_active is False


class TestTrailingStopMT5Sync:
    @pytest.mark.asyncio
    async def test_sync_exchange_stops_exists_on_mt5_executor(self):
        """MT5Executor must expose _sync_exchange_stops so market_consumer can call it."""
        from execution.mt5_executor import MT5Executor
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        with patch.dict("os.environ", {"MT5_LOGIN": "1", "MT5_PASSWORD": "x", "MT5_SERVER": "Demo"}):
            executor = MT5Executor(db=MagicMock(), risk_manager=rm, live=False)
            assert hasattr(executor, "_sync_exchange_stops")
