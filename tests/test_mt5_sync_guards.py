import unittest
from unittest.mock import MagicMock, patch

from execution.mt5_executor import SYMBOL_MAP, MT5Executor


class _FakeRiskManager:
    def __init__(self) -> None:
        self.balance = 10_000.0
        self.max_positions = 3


class _FakeDB:
    async def insert_open_trade(self, **kwargs) -> int:  # noqa: ARG002
        return 1


class MT5SyncGuardsTests(unittest.TestCase):
    def test_ghost_confirmation_threshold(self) -> None:
        ex = MT5Executor(db=_FakeDB(), risk_manager=_FakeRiskManager(), live=False)
        ex._ghost_min_confirmations = 3
        self.assertFalse(ex._ghost_mark_missing("ticket:1"))
        self.assertFalse(ex._ghost_mark_missing("ticket:1"))
        self.assertTrue(ex._ghost_mark_missing("ticket:1"))
        ex._ghost_reset("ticket:1")
        self.assertFalse(ex._ghost_mark_missing("ticket:1"))

    @patch("execution.mt5_executor.mt5")
    @patch("execution.mt5_executor._MT5_AVAILABLE", True)
    def test_validate_symbol_mapping_auto_heals_variant(self, mock_mt5) -> None:
        original = SYMBOL_MAP["BTC/USDT"]
        SYMBOL_MAP["BTC/USDT"] = "BTCUSD-T"

        def _fake_symbol_info(sym: str):
            if sym == "BTCUSD-T":
                return None
            if sym == "BTCUSD":
                obj = MagicMock()
                obj.visible = True
                return obj
            return None

        mock_mt5.symbol_info.side_effect = _fake_symbol_info
        ex = MT5Executor(db=_FakeDB(), risk_manager=_FakeRiskManager(), live=False)
        unresolved = ex.validate_symbol_mapping(["BTC/USDT"])
        self.assertEqual(unresolved, [])
        self.assertEqual(SYMBOL_MAP["BTC/USDT"], "BTCUSD")
        SYMBOL_MAP["BTC/USDT"] = original

