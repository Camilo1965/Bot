"""Critical-path regression tests: in-flight open reservation, limits, sector."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager


@pytest.mark.asyncio
async def test_same_symbol_blocked_while_db_insert_in_flight() -> None:
    """First open holds _pending_symbols until DB returns; second must not pass."""
    risk = RiskManager(initial_balance=10_000.0, max_positions=3)
    db = AsyncMock()
    gate = asyncio.Event()

    async def slow_insert(*_a: object, **_kw: object) -> int:
        await gate.wait()
        return 42

    db.insert_open_trade = slow_insert

    ex = PaperExecutor(db=db, risk_manager=risk)
    first = asyncio.create_task(
        ex.try_open_trade(entry_price=100.0, win_probability=0.7, symbol="BTC/USDT")
    )
    await asyncio.sleep(0)
    second = await ex.try_open_trade(entry_price=101.0, win_probability=0.7, symbol="BTC/USDT")
    assert second is False
    gate.set()
    assert await first is True
    assert risk.open_count == 1
    assert len(ex.open_positions) == 1


@pytest.mark.asyncio
async def test_max_positions_enforced_with_pending_risk_slots() -> None:
    """Three in-flight opens reserve slots so a fourth concurrent attempt fails."""
    risk = RiskManager(initial_balance=10_000.0, max_positions=3)
    db = AsyncMock()
    gate = asyncio.Event()

    async def slow_insert(*_a: object, **_kw: object) -> int:
        await gate.wait()
        return 1

    db.insert_open_trade = slow_insert

    ex = PaperExecutor(db=db, risk_manager=risk)
    syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    tasks = [
        asyncio.create_task(
            ex.try_open_trade(100.0, 0.7, symbol=s),
        )
        for s in syms
    ]
    await asyncio.sleep(0)
    fourth = await ex.try_open_trade(100.0, 0.7, symbol="LINK/USDT")
    assert fourth is False
    gate.set()
    results = await asyncio.gather(*tasks)
    assert all(results)
    assert risk.open_count == 3


@pytest.mark.asyncio
async def test_open_count_matches_positions_after_concurrent_burst() -> None:
    risk = RiskManager(initial_balance=100_000.0, max_positions=3)
    db = AsyncMock()
    db.insert_open_trade = AsyncMock(return_value=1)

    ex = PaperExecutor(db=db, risk_manager=risk)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    results = await asyncio.gather(
        *(ex.try_open_trade(100.0, 0.7, symbol=s) for s in symbols)
    )
    assert sum(results) == 3
    assert risk.open_count == 3 == len(ex.open_positions)
