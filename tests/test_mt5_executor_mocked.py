"""Heavy mocked tests for execution.mt5_executor to lift coverage without Windows MT5."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.mt5_executor import MT5Executor
from risk.risk_manager import RiskManager


class _FakeMT5:
    """Minimal mock of the MetaTrader5 module."""
    TRADE_RETCODE_DONE = 10009
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    POSITION_TYPE_BUY = 0
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 1
    ORDER_FILLING_IOC = 1

    @staticmethod
    def terminal_info():
        return SimpleNamespace(connected=True)

    @staticmethod
    def account_info():
        return SimpleNamespace(balance=10000.0, equity=10000.0, margin_free=8000.0, margin=2000.0)

    @staticmethod
    def symbol_info(sym: str):
        return SimpleNamespace(
            digits=5, point=0.00001, trade_tick_size=0.00001,
            trade_contract_size=1.0, visible=True,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            margin_initial=50000.0,
        )

    @staticmethod
    def symbol_info_tick(sym: str):
        return SimpleNamespace(ask=81000.0, bid=80990.0, time=datetime.now().timestamp())

    @staticmethod
    def positions_get(**kwargs):
        # Return empty by default; tests can patch this.
        return ()

    @staticmethod
    def order_send(request: dict):
        return SimpleNamespace(retcode=_FakeMT5.TRADE_RETCODE_DONE, comment="Done", order=12345, position=999)

    @staticmethod
    def initialize(**kwargs):
        return True

    @staticmethod
    def shutdown():
        pass


@pytest.fixture(autouse=True)
def _patch_mt5_module():
    with patch.dict("sys.modules", {"MetaTrader5": _FakeMT5()}):
        # Force re-import so the module-level `import MetaTrader5 as mt5` picks up our fake
        import execution.mt5_executor as _mte
        _mte.mt5 = _FakeMT5
        _mte._MT5_AVAILABLE = True
        yield


@pytest.fixture
def mt5_executor():
    rm = RiskManager(initial_balance=10_000.0, max_positions=3)
    ex = MT5Executor(db=MagicMock(), risk_manager=rm, live=True, magic=20240101)
    return ex


class TestMT5OpenTradeMocked:
    @pytest.mark.asyncio
    async def test_try_open_trade_live_success(self, mt5_executor):
        with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
            with patch.object(mt5_executor, "_mt5_terminal_connected", return_value=True):
                with patch.object(mt5_executor, "_validate_tick_freshness", return_value=True):
                    with patch.object(mt5_executor, "_broker_blocks_new_long_on_symbol", return_value=(False, "")):
                        with patch.object(mt5_executor, "_check_margin_available", return_value=True):
                            with patch.object(mt5_executor, "_send_order_with_retry", new_callable=AsyncMock, return_value=SimpleNamespace(retcode=_FakeMT5.TRADE_RETCODE_DONE, position=999)):
                                with patch.object(mt5_executor, "_resolve_position_ticket_after_buy", new_callable=AsyncMock, return_value=999):
                                    with patch.object(mt5_executor._db, "insert_open_trade", new_callable=AsyncMock, return_value="uuid-123"):
                                        with patch.object(mt5_executor._db, "update_trade_exit", new_callable=AsyncMock):
                                            result = await mt5_executor.try_open_trade(
                                                entry_price=80000.0,
                                                win_probability=0.7,
                                                symbol="BTC/USDT",
                                            )
                                            assert result is True
                                            assert "BTC/USDT" in mt5_executor.open_positions

    @pytest.mark.asyncio
    async def test_try_open_trade_live_rejected(self, mt5_executor):
        with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
            with patch.object(mt5_executor, "_mt5_terminal_connected", return_value=True):
                with patch.object(mt5_executor, "_validate_tick_freshness", return_value=True):
                    with patch.object(mt5_executor, "_broker_blocks_new_long_on_symbol", return_value=(False, "")):
                        with patch.object(mt5_executor, "_check_margin_available", return_value=True):
                            with patch.object(mt5_executor, "_send_order_with_retry", new_callable=AsyncMock, return_value=None):
                                with patch.object(mt5_executor._db, "update_trade_exit", new_callable=AsyncMock):
                                    result = await mt5_executor.try_open_trade(
                                        entry_price=80000.0,
                                        win_probability=0.7,
                                        symbol="BTC/USDT",
                                    )
                                    assert result is False


class TestMT5CloseMocked:
    @pytest.mark.asyncio
    async def test_close_position_live(self, mt5_executor):
        from execution.paper_executor import OpenPosition
        pos = OpenPosition(
            trade_id="t1", symbol="BTC/USDT",
            entry_time=datetime.now(tz=timezone.utc),
            entry_price=80000.0, position_size=1000.0,
            sl_price=78000.0, activation_price=81600.0,
            trailing_distance_pct=0.02, peak_price=80000.0,
            ml_confidence=0.7, sl_pct=0.025, activation_pct=0.02,
            stop_loss_price=78000.0, current_stop_loss=78000.0,
            mt5_position_ticket=999,
        )
        mt5_executor.open_positions["BTC/USDT"] = pos
        fake_mt5_pos = SimpleNamespace(ticket=999, magic=mt5_executor._magic, type=_FakeMT5.POSITION_TYPE_BUY, volume=0.1)
        with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
            with patch.object(_FakeMT5, "positions_get", return_value=(fake_mt5_pos,)):
                with patch.object(mt5_executor, "_send_order_with_retry", new_callable=AsyncMock, return_value=SimpleNamespace(retcode=_FakeMT5.TRADE_RETCODE_DONE)):
                    with patch.object(mt5_executor._db, "update_trade_exit", new_callable=AsyncMock):
                        pnl = await mt5_executor._close_position("BTC/USDT", 82000.0, "take_profit")
                        assert "BTC/USDT" not in mt5_executor.open_positions


class TestMT5SyncMocked:
    @pytest.mark.asyncio
    async def test_sync_positions_adopts_orphan(self, mt5_executor):
        fake_pos = SimpleNamespace(
            magic=mt5_executor._magic, type=_FakeMT5.POSITION_TYPE_BUY,
            symbol="BTCUSD", ticket=111, price_open=80000.0,
            sl=78000.0, volume=0.1, tp=0.0,
            price_current=80500.0, time=datetime.now().timestamp(),
        )
        with patch.object(_FakeMT5, "positions_get", return_value=(fake_pos,)):
            with patch.object(mt5_executor, "_local_symbol_from_broker", return_value="BTC/USDT"):
                with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
                    with patch.object(mt5_executor, "_mt5_terminal_connected", return_value=True):
                        with patch.object(mt5_executor._db, "insert_open_trade", new_callable=AsyncMock, return_value="adopt-uuid"):
                            count = await mt5_executor.sync_positions_with_exchange()
                            assert count == 1
                            assert "BTC/USDT" in mt5_executor.open_positions

    @pytest.mark.asyncio
    async def test_sync_positions_no_drift_when_empty(self, mt5_executor):
        with patch.object(_FakeMT5, "positions_get", return_value=()):
            with patch.object(mt5_executor, "_mt5_terminal_connected", return_value=True):
                count = await mt5_executor.sync_positions_with_exchange()
                assert count == 0


class TestMT5ModifyMocked:
    @pytest.mark.asyncio
    async def test_modify_position_success(self, mt5_executor):
        with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
            with patch.object(mt5_executor, "_send_order_with_retry", new_callable=AsyncMock, return_value=SimpleNamespace(retcode=_FakeMT5.TRADE_RETCODE_DONE)):
                ok = await mt5_executor.modify_position(999, new_sl=79000.0, symbol="BTC/USDT")
                assert ok is True

    @pytest.mark.asyncio
    async def test_sync_exchange_stops_ratchets(self, mt5_executor):
        from execution.paper_executor import OpenPosition
        pos = OpenPosition(
            trade_id="t1", symbol="BTC/USDT",
            entry_time=datetime.now(tz=timezone.utc),
            entry_price=80000.0, position_size=1000.0,
            sl_price=78000.0, activation_price=81600.0,
            trailing_distance_pct=0.02, peak_price=85000.0,
            ml_confidence=0.7, sl_pct=0.025, activation_pct=0.02,
            stop_loss_price=78000.0, current_stop_loss=78000.0,
            mt5_position_ticket=999, last_mt5_modify_mono=0.0,
        )
        mt5_executor.open_positions["BTC/USDT"] = pos
        fake_broker_pos = SimpleNamespace(sl=78000.0, tp=0.0)
        with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
            with patch.object(mt5_executor, "_find_magic_long_ticket", return_value=999):
                with patch.object(_FakeMT5, "positions_get", return_value=(fake_broker_pos,)):
                    with patch.object(mt5_executor, "_validate_stops", return_value=True):
                        with patch.object(mt5_executor, "_clamp_stop_loss_buy", return_value=(83300.0, False)):
                            with patch.object(mt5_executor, "modify_position", new_callable=AsyncMock, return_value=True) as mock_mod:
                                await mt5_executor._sync_exchange_stops("BTC/USDT", pos, 85000.0, None)
                                # Verify modify_position was called with a ratcheted SL
                                assert mock_mod.called
                                args = mock_mod.call_args
                                assert args.kwargs["new_sl"] > 78000.0


class TestMT5EffectiveTrailing:
    def test_effective_trailing_includes_spread(self, mt5_executor):
        with patch.object(mt5_executor, "_resolve_symbol", return_value="BTCUSD"):
            eff = mt5_executor._effective_trailing_distance("BTC/USDT", 80000.0, 0.02)
            assert eff >= 0.02


class TestMT5RecoveryMocked:
    @pytest.mark.asyncio
    async def test_recover_positions_on_startup(self, mt5_executor):
        fake_pos = SimpleNamespace(
            magic=mt5_executor._magic, type=_FakeMT5.POSITION_TYPE_BUY,
            symbol="BTCUSD", ticket=222, price_open=80500.0,
            sl=80000.0, volume=0.05,
        )
        with patch.object(_FakeMT5, "positions_get", return_value=(fake_pos,)):
            with patch.object(mt5_executor, "_local_symbol_from_broker", return_value="BTC/USDT"):
                n = await mt5_executor.recover_positions_on_startup()
                assert n == 1
                assert "BTC/USDT" in mt5_executor.open_positions
                assert mt5_executor.open_positions["BTC/USDT"].mt5_position_ticket == 222
