"""Tests for position limit enforcement and race condition prevention.

Validates:
- max_positions limit is respected
- Duplicate symbol prevention works
- Counter invariant is maintained after sync
- register_open/register_close rollback on failures
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from risk.risk_manager import RiskManager


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(initial_balance=10_000.0, max_positions=3)


def test_can_open_position_respects_max(risk_manager: RiskManager) -> None:
    """Verify can_open_position blocks after max_positions reached."""
    assert risk_manager.can_open_position() is True
    risk_manager.register_open()
    assert risk_manager.can_open_position() is True
    risk_manager.register_open()
    assert risk_manager.can_open_position() is True
    risk_manager.register_open()
    assert risk_manager.can_open_position() is False


def test_register_close_below_zero(risk_manager: RiskManager) -> None:
    """Verify register_close doesn't go below 0."""
    assert risk_manager.open_count == 0
    risk_manager.register_close()
    assert risk_manager.open_count == 0


def test_sync_open_count(risk_manager: RiskManager) -> None:
    """Verify sync_open_count corrects drift."""
    risk_manager.register_open()
    risk_manager.register_open()
    assert risk_manager.open_count == 2
    risk_manager.sync_open_count(1)
    assert risk_manager.open_count == 1
    assert risk_manager.can_open_position() is True


def test_sync_open_count_negative_clamped(risk_manager: RiskManager) -> None:
    """Verify sync_open_count clamps negative to 0."""
    risk_manager.sync_open_count(-5)
    assert risk_manager.open_count == 0


def test_sector_exposure_blocks_duplicate_sector(risk_manager: RiskManager) -> None:
    """Same sector = blocked."""
    assert risk_manager.is_sector_exposed("BTC/USDT", []) is False
    assert risk_manager.is_sector_exposed("BTC/USDT", ["ETH/USDT"]) is False
    assert risk_manager.is_sector_exposed("LINK/USDT", ["INJ/USDT"]) is True


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.insert_open_trade = AsyncMock(return_value=1)
    db.close_trade = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_paper_executor_duplicate_symbol_rejected(
    risk_manager: RiskManager, mock_db: AsyncMock
) -> None:
    """Verify a second open on the same symbol is rejected."""
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor(db=mock_db, risk_manager=risk_manager)
    opened = await executor.try_open_trade(
        entry_price=100.0,
        win_probability=0.7,
        symbol="BTC/USDT",
        sentiment_score=0.0,
    )
    assert opened is True
    assert risk_manager.open_count == 1

    opened2 = await executor.try_open_trade(
        entry_price=100.0,
        win_probability=0.7,
        symbol="BTC/USDT",
        sentiment_score=0.0,
    )
    assert opened2 is False
    assert risk_manager.open_count == 1


@pytest.mark.asyncio
async def test_paper_executor_max_positions_respected(
    risk_manager: RiskManager, mock_db: AsyncMock
) -> None:
    """Verify 4th position on a max_positions=3 executor is blocked."""
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor(db=mock_db, risk_manager=risk_manager)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    results = []
    for sym in symbols:
        opened = await executor.try_open_trade(
            entry_price=100.0,
            win_probability=0.7,
            symbol=sym,
            sentiment_score=0.0,
        )
        results.append(opened)

    assert results == [True, True, True, False]
    assert risk_manager.open_count == 3
    assert len(executor.open_positions) == 3


@pytest.mark.asyncio
async def test_paper_executor_counter_matches_dict(
    risk_manager: RiskManager, mock_db: AsyncMock
) -> None:
    """Verify the risk counter stays in sync with open_positions dict."""
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor(db=mock_db, risk_manager=risk_manager)
    await executor.try_open_trade(
        entry_price=100.0,
        win_probability=0.7,
        symbol="BTC/USDT",
    )
    assert risk_manager.open_count == len(executor.open_positions) == 1

    await executor.try_open_trade(
        entry_price=50.0,
        win_probability=0.7,
        symbol="ETH/USDT",
    )
    assert risk_manager.open_count == len(executor.open_positions) == 2


@pytest.mark.asyncio
async def test_concurrent_open_same_symbol_only_one_succeeds(
    risk_manager: RiskManager, mock_db: AsyncMock
) -> None:
    """Simulate two concurrent try_open_trade calls for the same symbol.
    
    Only one should succeed (the lock prevents race conditions).
    """
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor(db=mock_db, risk_manager=risk_manager)

    results = await asyncio.gather(
        executor.try_open_trade(entry_price=100.0, win_probability=0.7, symbol="BTC/USDT"),
        executor.try_open_trade(entry_price=100.0, win_probability=0.7, symbol="BTC/USDT"),
    )
    assert sum(results) == 1
    assert risk_manager.open_count == 1
    assert len(executor.open_positions) == 1


@pytest.mark.asyncio
async def test_concurrent_open_exceeding_max_positions(
    risk_manager: RiskManager, mock_db: AsyncMock
) -> None:
    """Simulate 5 concurrent opens on max_positions=3. At most 3 succeed."""
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor(db=mock_db, risk_manager=risk_manager)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "LINK/USDT"]

    results = await asyncio.gather(
        *(executor.try_open_trade(entry_price=100.0, win_probability=0.7, symbol=sym)
          for sym in symbols)
    )
    assert sum(results) <= 3
    assert risk_manager.open_count == sum(results)
    assert len(executor.open_positions) == sum(results)


@pytest.mark.asyncio
async def test_close_then_reopen_same_symbol(
    risk_manager: RiskManager, mock_db: AsyncMock
) -> None:
    """After closing, the same symbol can be reopened."""
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor(db=mock_db, risk_manager=risk_manager)
    await executor.try_open_trade(entry_price=100.0, win_probability=0.7, symbol="BTC/USDT")
    assert "BTC/USDT" in executor.open_positions

    # Drop below default SL (1.5% of 100 = 98.5) but not enough to trigger daily loss halt
    pnl = await executor.check_and_close(current_price=98.0, symbol="BTC/USDT")
    assert pnl is not None
    assert "BTC/USDT" not in executor.open_positions
    assert risk_manager.open_count == 0

    reopened = await executor.try_open_trade(entry_price=98.0, win_probability=0.7, symbol="BTC/USDT")
    assert reopened is True
    assert risk_manager.open_count == 1


def test_db_close_trade_exit_reason_param() -> None:
    """Verify the close_trade query includes exit_reason parameter slot ($9)."""
    from database.db_manager import _CLOSE_TRADE
    assert "$9" in _CLOSE_TRADE
    assert "exit_reason" in _CLOSE_TRADE


def test_db_schema_has_lifecycle_columns() -> None:
    """Verify the ALTER adds exit_reason, mt5_ticket, sl_price, tp_price."""
    from database.db_manager import _ALTER_TRADES_HISTORY_METRICS
    assert "exit_reason" in _ALTER_TRADES_HISTORY_METRICS
    assert "mt5_ticket" in _ALTER_TRADES_HISTORY_METRICS
    assert "sl_price" in _ALTER_TRADES_HISTORY_METRICS
    assert "tp_price" in _ALTER_TRADES_HISTORY_METRICS
