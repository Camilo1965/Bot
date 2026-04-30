"""Tests for :class:`~risk.risk_manager.RiskManager`."""

from __future__ import annotations

from risk.risk_manager import (
    MAX_PORTFOLIO_DD_PCT,
    RiskManager,
    RISK_PER_TRADE,
    LEVERAGE,
    MAX_POSITIONS,
)


def test_calculate_position_size_uses_fixed_fraction_and_cap() -> None:
    rm = RiskManager(initial_balance=10_000.0, max_positions=3)
    raw = 10_000.0 * RISK_PER_TRADE * LEVERAGE
    cap = 10_000.0 / 3
    assert rm.calculate_position_size(0.99) == min(raw, cap)


def test_calculate_position_size_ignores_win_probability() -> None:
    rm = RiskManager(initial_balance=1000.0, max_positions=MAX_POSITIONS)
    a = rm.calculate_position_size(0.1)
    b = rm.calculate_position_size(0.99)
    assert a == b


def test_can_open_position_respects_max() -> None:
    rm = RiskManager(max_positions=2)
    rm.sync_open_count(2)
    assert rm.can_open_position() is False
    rm.sync_open_count(1)
    assert rm.can_open_position() is True


def test_portfolio_drawdown_exceeded() -> None:
    rm = RiskManager(initial_balance=1000.0)
    floor = 1000.0 * (1.0 - MAX_PORTFOLIO_DD_PCT)
    rm.balance = floor - 1.0
    assert rm.is_portfolio_dd_exceeded() is True


def test_daily_loss_triggers_halt() -> None:
    rm = RiskManager(initial_balance=1000.0)
    rm.record_daily_loss(40.0)
    assert rm.is_trading_halted() is True
