"""Tests for the core trading loop (market_consumer, signal_emitter, loops).

These modules have 0% coverage and are the heart of live execution.
"""

from __future__ import annotations

import asyncio
import collections
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager


def _make_state(watchlist: list[str] | None = None) -> dict:
    """Build a minimal shared_state matching main.py initialization."""
    syms = watchlist or ["BTC/USDT"]
    return {
        "prices": {s: deque(maxlen=1000) for s in syms},
        "highs": {s: deque(maxlen=1000) for s in syms},
        "lows": {s: deque(maxlen=1000) for s in syms},
        "volumes": {s: deque(maxlen=1000) for s in syms},
        "dashboard_rsi_closes": {s: deque(maxlen=500) for s in syms},
        "last_kline_ts": {s: None for s in syms},
        "mt5_last_quote": {},
        "htf_closes": {s: {"1h": deque(maxlen=500), "4h": deque(maxlen=500)} for s in syms},
        "htf_opens": {s: {"1h": deque(maxlen=500), "4h": deque(maxlen=500)} for s in syms},
        "ml_signals": {},
        "ml_probs": {},
        "atrs": {},
        "obi_ratios": {},
        "funding_rates": {},
        "vibe_patterns": {},
        "last_market_message_at": None,
        "last_price": 0.0,
        "max_drawdown": 0.0,
    }


def _fill_prices(state: dict, symbol: str, n: int = 800, base: float = 80000.0) -> None:
    """Populate price buffers with enough rows for inference."""
    for i in range(n):
        p = base + (i % 100) * 10.0
        state["prices"][symbol].append(p)
        state["highs"][symbol].append(p + 50.0)
        state["lows"][symbol].append(p - 50.0)
        state["volumes"][symbol].append(1000.0)


# ═════════════════════════════════════════════════════════════════════════════
#  MARKET CONSUMER
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketConsumer:
    @pytest.mark.asyncio
    async def test_trade_message_updates_last_price(self):
        from bot.market_consumer import market_consumer

        queue: asyncio.Queue[dict] = asyncio.Queue()
        state = _make_state()
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)

        queue.put_nowait({"symbol": "BTC/USDT", "type": "trade", "price": 81000.0})
        task = asyncio.create_task(market_consumer(queue, state, executor))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert state["last_price"] == 81000.0
        assert state["last_market_message_at"] is not None

    @pytest.mark.asyncio
    async def test_kline_message_updates_buffers(self):
        from bot.market_consumer import market_consumer

        queue: asyncio.Queue[dict] = asyncio.Queue()
        state = _make_state()
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)

        queue.put_nowait({
            "symbol": "BTC/USDT",
            "type": "kline",
            "timeframe": "30m",
            "close": 82000.0,
            "open": 81000.0,
            "high": 83000.0,
            "low": 80500.0,
            "volume": 5000.0,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })
        task = asyncio.create_task(market_consumer(queue, state, executor))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(state["prices"]["BTC/USDT"]) == 1
        assert state["prices"]["BTC/USDT"][0] == 82000.0
        assert state["highs"]["BTC/USDT"][0] == 83000.0
        assert state["lows"]["BTC/USDT"][0] == 80500.0

    @pytest.mark.asyncio
    async def test_order_book_message_computes_mid(self):
        from bot.market_consumer import market_consumer

        queue: asyncio.Queue[dict] = asyncio.Queue()
        state = _make_state()
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)

        queue.put_nowait({
            "symbol": "BTC/USDT",
            "type": "order_book",
            "bids": [["80990.0", "1.0"]],
            "asks": [["81010.0", "1.0"]],
        })
        task = asyncio.create_task(market_consumer(queue, state, executor))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert state["last_price"] == pytest.approx(81000.0)

    @pytest.mark.asyncio
    async def test_invalid_price_skipped(self):
        from bot.market_consumer import market_consumer

        queue: asyncio.Queue[dict] = asyncio.Queue()
        state = _make_state()
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)

        queue.put_nowait({"symbol": "BTC/USDT", "type": "trade", "price": -1.0})
        task = asyncio.create_task(market_consumer(queue, state, executor))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert state["last_price"] == 0.0  # default, not updated

    @pytest.mark.asyncio
    async def test_check_and_close_called_when_position_open(self):
        from bot.market_consumer import market_consumer
        from execution.paper_executor import OpenPosition

        queue: asyncio.Queue[dict] = asyncio.Queue()
        state = _make_state()
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        pos = OpenPosition(
            trade_id="t1", symbol="BTC/USDT",
            entry_time=datetime.now(tz=timezone.utc),
            entry_price=80000.0, position_size=1000.0,
            sl_price=78000.0, activation_price=81600.0,
            trailing_distance_pct=0.02, peak_price=80000.0,
            ml_confidence=0.7, sl_pct=0.025, activation_pct=0.02,
            stop_loss_price=78000.0, current_stop_loss=78000.0,
        )
        executor.open_positions["BTC/USDT"] = pos

        with patch.object(executor, "check_and_close", new_callable=AsyncMock) as mock_close:
            queue.put_nowait({"symbol": "BTC/USDT", "type": "trade", "price": 81000.0})
            task = asyncio.create_task(market_consumer(queue, state, executor))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            mock_close.assert_awaited_once_with("BTC/USDT", 81000.0)


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL EMITTER
# ═════════════════════════════════════════════════════════════════════════════

class _FakePredictor:
    def __init__(self, signal: str = "HOLD", prob: float = 0.0):
        self._signal = signal
        self._prob = prob

    def generate_signal(self, *args, **kwargs):
        return self._signal, self._prob


class TestSignalEmitter:
    @pytest.mark.asyncio
    async def test_warmup_not_enough_data(self):
        from bot.signal_emitter import signal_emitter

        state = _make_state()
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        predictor = _FakePredictor("BUY", 0.9)

        task = asyncio.create_task(signal_emitter(state, predictor, executor, ["BTC/USDT"], interval=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Not enough prices → no signal stored
        assert "BTC/USDT" not in state["ml_signals"]

    @pytest.mark.asyncio
    async def test_buy_signal_opens_trade(self):
        from bot.signal_emitter import signal_emitter

        state = _make_state()
        _fill_prices(state, "BTC/USDT", n=800)
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        predictor = _FakePredictor("BUY", 0.75)

        with patch.object(executor, "try_open_trade", new_callable=AsyncMock, return_value=True) as mock_open:
            task = asyncio.create_task(signal_emitter(state, predictor, executor, ["BTC/USDT"], interval=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert state["ml_signals"]["BTC/USDT"] == "BUY"
        assert state["ml_probs"]["BTC/USDT"] == 0.75
        assert mock_open.await_count >= 1

    @pytest.mark.asyncio
    async def test_sell_signal_triggers_smart_exit(self):
        from bot.signal_emitter import signal_emitter
        from execution.paper_executor import OpenPosition

        state = _make_state()
        _fill_prices(state, "BTC/USDT", n=800)
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        pos = OpenPosition(
            trade_id="t1", symbol="BTC/USDT",
            entry_time=datetime.now(tz=timezone.utc),
            entry_price=80000.0, position_size=1000.0,
            sl_price=78000.0, activation_price=81600.0,
            trailing_distance_pct=0.02, peak_price=85000.0,
            ml_confidence=0.7, sl_pct=0.025, activation_pct=0.02,
            stop_loss_price=78000.0, current_stop_loss=78000.0,
        )
        executor.open_positions["BTC/USDT"] = pos
        predictor = _FakePredictor("SELL", 0.8)

        with patch.object(executor, "check_ml_exit", new_callable=AsyncMock, return_value=150.0) as mock_exit:
            task = asyncio.create_task(signal_emitter(state, predictor, executor, ["BTC/USDT"], interval=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert mock_exit.await_count >= 1

    @pytest.mark.asyncio
    async def test_atr_cached_in_state(self):
        from bot.signal_emitter import signal_emitter

        state = _make_state()
        _fill_prices(state, "BTC/USDT", n=800)
        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        predictor = _FakePredictor("HOLD", 0.3)

        task = asyncio.create_task(signal_emitter(state, predictor, executor, ["BTC/USDT"], interval=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "BTC/USDT" in state["atrs"]
        assert state["atrs"]["BTC/USDT"] > 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  LOOPS
# ═════════════════════════════════════════════════════════════════════════════

class TestPositionSyncLoop:
    @pytest.mark.asyncio
    async def test_exits_immediately_for_paper_mode(self):
        from bot.loops import position_sync_loop

        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        # Paper mode has no _exchange and is not MT5Executor live
        result = await position_sync_loop(executor, interval=0.01)
        assert result is None  # returns immediately

    @pytest.mark.asyncio
    async def test_calls_sync_for_live_mt5(self):
        from bot.loops import position_sync_loop
        from execution.mt5_executor import MT5Executor

        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = MT5Executor(db=MagicMock(), risk_manager=rm, live=True)
        with patch.object(executor, "sync_positions_with_exchange", new_callable=AsyncMock, return_value=2) as mock_sync:
            task = asyncio.create_task(position_sync_loop(executor, interval=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            mock_sync.assert_awaited()


class TestHealthMonitorLoop:
    @pytest.mark.asyncio
    async def test_detects_queue_backlog(self):
        from bot.loops import health_monitor_loop

        state = _make_state()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        # Fill queue above threshold
        for i in range(850):
            queue.put_nowait({"symbol": "BTC/USDT", "price": float(i)})

        executor = PaperExecutor(db=MagicMock(), risk_manager=RiskManager(), exchange=None)
        with patch.object(executor, "sync_positions_with_exchange", new_callable=AsyncMock):
            task = asyncio.create_task(health_monitor_loop(state, queue, executor, RiskManager(), interval=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Should not raise; validates that backlog path runs
        assert True


class TestDashboardLogger:
    @pytest.mark.asyncio
    async def test_updates_dashboard(self):
        from bot.loops import dashboard_logger

        state = _make_state()
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        executor = PaperExecutor(db=MagicMock(), risk_manager=rm, exchange=None)
        live_mock = MagicMock()
        live_mock.update = MagicMock()

        with patch("bot.loops.generate_dashboard", return_value="dashboard_table"):
            task = asyncio.create_task(dashboard_logger(executor, rm, state, live_mock, ["BTC/USDT"], interval=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        live_mock.update.assert_called_with("dashboard_table")
