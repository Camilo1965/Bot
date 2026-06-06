"""Property-based regression tests for critical trading invariants.

Uses Hypothesis to fuzz inputs and verify that core safety properties
hold across a wide range of values.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from execution.paper_executor import OpenPosition, PaperExecutor
from risk.risk_manager import RiskManager, MAX_PORTFOLIO_RISK_PCT, LEVERAGE


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_check_and_close(executor: PaperExecutor, symbol: str, price: float) -> OpenPosition | None:
    await executor.check_and_close(symbol, price)
    return executor.open_positions.get(symbol)


def _run_async(coro) -> None:
    """Run a coroutine in an isolated event loop, cancelling pending tasks before close.

    Using asyncio.run() directly in a hypothesis test can leave AsyncMock coroutines
    pending; cancelling them prevents the 'coroutine ignored GeneratorExit' warning
    that pytest-asyncio turns into a test failure.
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def _make_executor_with_pos(
    entry_price: float = 100.0,
    activation_pct: float = 0.02,
    trailing_distance_pct: float = 0.02,
    sl_pct: float = 0.025,
) -> PaperExecutor:
    rm = RiskManager(initial_balance=10_000.0, max_positions=3)
    db_mock = AsyncMock()
    executor = PaperExecutor(db=db_mock, risk_manager=rm, exchange=None)
    pos = OpenPosition(
        trade_id="t1",
        symbol="BTC/USDT",
        entry_time=datetime.now(tz=timezone.utc),
        entry_price=entry_price,
        position_size=1000.0,
        sl_price=entry_price * (1 - sl_pct),
        activation_price=entry_price * (1 + activation_pct),
        trailing_distance_pct=trailing_distance_pct,
        peak_price=entry_price,
        ml_confidence=0.7,
        sl_pct=sl_pct,
        activation_pct=activation_pct,
        stop_loss_price=entry_price * (1 - sl_pct),
        current_stop_loss=entry_price * (1 - sl_pct),
    )
    executor.open_positions["BTC/USDT"] = pos
    executor._append_to_journal = lambda *args, **kwargs: None
    return executor


# ── Trailing-stop properties ──────────────────────────────────────────────────

class TestTrailingStopProperties:
    @settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much], deadline=None)
    @given(
        entry_price=st.floats(min_value=1.0, max_value=1_000_000.0),
        price_sequence=st.lists(
            st.floats(min_value=0.01, max_value=2_000_000.0),
            min_size=1, max_size=20,
        ),
        activation_pct=st.floats(min_value=0.001, max_value=0.50),
        trailing_distance_pct=st.floats(min_value=0.001, max_value=0.50),
        sl_pct=st.floats(min_value=0.001, max_value=0.50),
    )
    def test_sl_price_never_decreases(self, entry_price, price_sequence, activation_pct, trailing_distance_pct, sl_pct):
        """Once the SL is ratcheted up it must never move back down."""
        executor = _make_executor_with_pos(entry_price, activation_pct, trailing_distance_pct, sl_pct)
        prev_sl = executor.open_positions["BTC/USDT"].sl_price

        for price in price_sequence:
            # skip non-finite values that hypothesis might generate at boundaries
            if price != price or price <= 0.0:
                continue
            _run_async(_run_check_and_close(executor, "BTC/USDT", price))
            pos = executor.open_positions.get("BTC/USDT")
            if pos is None:
                break
            assert pos.sl_price >= prev_sl * 0.99999999  # allow tiny fp jitter
            prev_sl = pos.sl_price

    @settings(max_examples=200, deadline=None)
    @given(
        entry_price=st.floats(min_value=10.0, max_value=100_000.0),
        activation_pct=st.floats(min_value=0.001, max_value=0.20),
    )
    def test_trail_activates_when_price_crosses_threshold(self, entry_price, activation_pct):
        """Trailing stop must become active as soon as price >= activation_price."""
        executor = _make_executor_with_pos(entry_price, activation_pct)
        activation_price = entry_price * (1 + activation_pct)

        # Price exactly at activation threshold
        asyncio.run(_run_check_and_close(executor, "BTC/USDT", activation_price))
        pos = executor.open_positions["BTC/USDT"]
        assert pos.trailing_stop_active is True

    @settings(max_examples=100, deadline=None)
    @given(
        entry_price=st.floats(min_value=10.0, max_value=100_000.0),
        current_price=st.floats(min_value=0.01, max_value=2_000_000.0),
        sl_pct=st.floats(min_value=0.001, max_value=0.50),
    )
    def test_position_is_closed_if_price_hits_sl(self, entry_price, current_price, sl_pct):
        """If current_price <= sl_price the position must be closed."""
        executor = _make_executor_with_pos(entry_price, sl_pct=sl_pct)
        sl_price = entry_price * (1 - sl_pct)

        # Drive price to SL
        asyncio.run(_run_check_and_close(executor, "BTC/USDT", sl_price))
        assert "BTC/USDT" not in executor.open_positions

    @settings(max_examples=200, deadline=None)
    @given(
        entry_price=st.floats(min_value=10.0, max_value=100_000.0),
        peak_multiplier=st.floats(min_value=1.0, max_value=5.0),
        trailing_distance_pct=st.floats(min_value=0.001, max_value=0.50),
    )
    def test_trailing_sl_is_at_least_peak_minus_distance(self, entry_price, peak_multiplier, trailing_distance_pct):
        """When active, SL >= peak * (1 - trailing_distance_pct)."""
        executor = _make_executor_with_pos(entry_price, trailing_distance_pct=trailing_distance_pct)
        peak = entry_price * peak_multiplier
        asyncio.run(_run_check_and_close(executor, "BTC/USDT", peak))
        pos = executor.open_positions.get("BTC/USDT")
        if pos is None:
            pytest.skip("Position closed before check")
        if not pos.trailing_stop_active:
            pytest.skip("Trailing stop not yet activated")
        expected_min_sl = peak * (1.0 - trailing_distance_pct)
        assert pos.sl_price >= expected_min_sl * 0.99999999


# ── RiskManager properties ────────────────────────────────────────────────────

class TestRiskManagerProperties:
    @settings(max_examples=300, deadline=None)
    @given(
        balance=st.floats(min_value=100.0, max_value=10_000_000.0),
        win_prob=st.floats(min_value=0.0, max_value=1.0),
        sl_distance_pct=st.floats(min_value=0.001, max_value=0.50),
        max_positions=st.integers(min_value=1, max_value=20),
        vibe_quality=st.floats(min_value=0.0, max_value=2.0),
    )
    def test_position_size_is_non_negative(self, balance, win_prob, sl_distance_pct, max_positions, vibe_quality):
        rm = RiskManager(initial_balance=balance, max_positions=max_positions)
        size = rm.calculate_position_size(
            win_probability=win_prob,
            sl_distance_pct=sl_distance_pct,
            vibe_quality=vibe_quality,
        )
        assert size >= 0.0

    @settings(max_examples=300, deadline=None)
    @given(
        balance=st.floats(min_value=100.0, max_value=10_000_000.0),
        win_prob=st.floats(min_value=0.0, max_value=1.0),
        sl_distance_pct=st.floats(min_value=0.001, max_value=0.50),
        max_positions=st.integers(min_value=1, max_value=20),
    )
    def test_position_size_does_not_exceed_max_allocation(self, balance, win_prob, sl_distance_pct, max_positions):
        rm = RiskManager(initial_balance=balance, max_positions=max_positions)
        size = rm.calculate_position_size(
            win_probability=win_prob,
            sl_distance_pct=sl_distance_pct,
        )
        max_allocation = (balance / max_positions) * LEVERAGE
        assert size <= max_allocation * 1.00000001

    @settings(max_examples=200, deadline=None)
    @given(
        open_count=st.integers(min_value=0, max_value=50),
        max_positions=st.integers(min_value=1, max_value=20),
    )
    def test_can_open_respects_max_positions(self, open_count, max_positions):
        rm = RiskManager(initial_balance=10_000.0, max_positions=max_positions)
        rm.sync_open_count(open_count)
        if open_count >= max_positions:
            assert rm.can_open_position() is False
        else:
            assert rm.can_open_position() is True

    @settings(max_examples=200, deadline=None)
    @given(
        initial_balance=st.floats(min_value=100.0, max_value=10_000_000.0),
        dd_multiplier=st.floats(min_value=0.0, max_value=0.5),
    )
    def test_portfolio_dd_exceeded_when_balance_below_floor(self, initial_balance, dd_multiplier):
        rm = RiskManager(initial_balance=initial_balance)
        floor = initial_balance * (1 - 0.15)
        rm.balance = floor * (1 - dd_multiplier)
        if rm.balance < floor:
            assert rm.is_portfolio_dd_exceeded() is True
        else:
            assert rm.is_portfolio_dd_exceeded() is False

    @settings(max_examples=200, deadline=None)
    @given(
        loss=st.floats(min_value=0.0, max_value=1_000_000.0),
        daily_start=st.floats(min_value=100.0, max_value=10_000_000.0),
    )
    def test_trading_halted_after_daily_loss_threshold(self, loss, daily_start):
        rm = RiskManager(initial_balance=daily_start)
        rm._daily_start_balance = daily_start
        rm.record_daily_loss(loss)
        threshold = daily_start * 0.03
        if loss >= threshold:
            assert rm.is_trading_halted() is True
        else:
            assert rm.is_trading_halted() is False

    @settings(max_examples=200, deadline=None)
    @given(
        balance=st.floats(min_value=100.0, max_value=10_000_000.0),
        total_risk_usd=st.floats(min_value=0.0, max_value=1_000_000_000.0),
    )
    def test_current_risk_exposure_non_negative(self, balance, total_risk_usd):
        rm = RiskManager(initial_balance=balance)
        rm._total_risk_usd = total_risk_usd
        exposure = rm.get_current_risk_exposure()
        assert exposure >= 0.0

    @settings(max_examples=200, deadline=None)
    @given(
        balance=st.floats(min_value=100.0, max_value=10_000_000.0),
        win_prob=st.floats(min_value=0.0, max_value=1.0),
        sl_distance_pct=st.floats(min_value=0.001, max_value=0.50),
    )
    def test_position_size_zero_when_risk_budget_full(self, balance, win_prob, sl_distance_pct):
        rm = RiskManager(initial_balance=balance, max_positions=1)
        # Fill the risk budget artificially
        rm._total_risk_usd = balance * MAX_PORTFOLIO_RISK_PCT
        size = rm.calculate_position_size(
            win_probability=win_prob,
            sl_distance_pct=sl_distance_pct,
        )
        assert size == 0.0

    @settings(max_examples=200, deadline=None)
    @given(
        balance=st.floats(min_value=100.0, max_value=10_000_000.0),
        risk_usd=st.floats(min_value=0.0, max_value=1_000_000.0),
    )
    def test_register_open_close_balances_risk(self, balance, risk_usd):
        rm = RiskManager(initial_balance=balance)
        initial_exposure = rm.get_current_risk_exposure()
        rm.register_open(risk_usd)
        assert rm.get_current_risk_exposure() >= initial_exposure
        rm.register_close(risk_usd)
        assert rm.get_current_risk_exposure() <= initial_exposure + 1e-9
        assert rm._total_risk_usd >= 0.0
