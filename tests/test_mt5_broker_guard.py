"""MT5 broker occupation guard (foreign magic / duplicate long)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from database.db_manager import DatabaseManager
from execution.mt5_executor import MT5Executor
from risk.risk_manager import RiskManager


@pytest.fixture
def mock_db() -> MagicMock:
    db = MagicMock(spec=DatabaseManager)
    return db


def test_broker_blocks_foreign_long(mock_db: MagicMock) -> None:
    """Manual / other-EA long on symbol blocks bot open when overlap disallowed."""
    risk = RiskManager(initial_balance=10_000.0, max_positions=3)
    ex = MT5Executor(db=mock_db, risk_manager=risk, live=False, magic=20240101)
    foreign_pos = MagicMock()
    foreign_pos.type = 0  # BUY
    foreign_pos.magic = 999999

    with (
        patch("execution.mt5_executor._MT5_AVAILABLE", True),
        patch("execution.mt5_executor.mt5") as m,
    ):
        m.POSITION_TYPE_BUY = 0
        m.positions_get = MagicMock(return_value=[foreign_pos])
        block, reason = ex._broker_blocks_new_long_on_symbol("BTCUSD-T")
    assert block is True
    assert "foreign" in reason


def test_broker_allows_foreign_when_env_set(mock_db: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MT5_ALLOW_FOREIGN_SYMBOL_OVERLAP", "1")
    risk = RiskManager(initial_balance=10_000.0, max_positions=3)
    ex = MT5Executor(db=mock_db, risk_manager=risk, live=False, magic=20240101)
    foreign_pos = MagicMock()
    foreign_pos.type = 0
    foreign_pos.magic = 999999

    with (
        patch("execution.mt5_executor._MT5_AVAILABLE", True),
        patch("execution.mt5_executor.mt5") as m,
    ):
        m.POSITION_TYPE_BUY = 0
        m.positions_get = MagicMock(return_value=[foreign_pos])
        block, _reason = ex._broker_blocks_new_long_on_symbol("BTCUSD-T")
    assert block is False


def test_broker_blocks_second_bot_long(mock_db: MagicMock) -> None:
    """Broker already shows our magic — treat as occupied (sync lag safety)."""
    risk = RiskManager(initial_balance=10_000.0, max_positions=3)
    magic = 20240101
    ex = MT5Executor(db=mock_db, risk_manager=risk, live=False, magic=magic)
    ours = MagicMock()
    ours.type = 0
    ours.magic = magic

    with (
        patch("execution.mt5_executor._MT5_AVAILABLE", True),
        patch("execution.mt5_executor.mt5") as m,
    ):
        m.POSITION_TYPE_BUY = 0
        m.positions_get = MagicMock(return_value=[ours])
        block, reason = ex._broker_blocks_new_long_on_symbol("BTCUSD-T")
    assert block is True
    assert "bot_long" in reason
