import unittest
from datetime import datetime, timedelta, timezone
import sys
import types

# Lightweight ccxt stubs so tests run even when ccxt is not installed globally.
if "ccxt" not in sys.modules:
    ccxt_mod = types.ModuleType("ccxt")
    ccxt_async_mod = types.ModuleType("ccxt.async_support")
    ccxt_base_mod = types.ModuleType("ccxt.base")
    ccxt_errors_mod = types.ModuleType("ccxt.base.errors")

    class _NetworkError(Exception):
        pass

    class _AuthenticationError(Exception):
        pass

    class _ExchangeError(Exception):
        pass

    class _InsufficientFunds(Exception):
        pass

    class _NotSupported(Exception):
        pass

    ccxt_errors_mod.NetworkError = _NetworkError
    ccxt_errors_mod.AuthenticationError = _AuthenticationError
    ccxt_errors_mod.ExchangeError = _ExchangeError
    ccxt_errors_mod.InsufficientFunds = _InsufficientFunds
    ccxt_errors_mod.NotSupported = _NotSupported

    ccxt_mod.async_support = ccxt_async_mod
    ccxt_mod.base = ccxt_base_mod
    ccxt_base_mod.errors = ccxt_errors_mod

    sys.modules["ccxt"] = ccxt_mod
    sys.modules["ccxt.async_support"] = ccxt_async_mod
    sys.modules["ccxt.base"] = ccxt_base_mod
    sys.modules["ccxt.base.errors"] = ccxt_errors_mod

if "aiohttp" not in sys.modules:
    aiohttp_mod = types.ModuleType("aiohttp")

    class _ClientError(Exception):
        pass

    class _ClientTimeout:
        def __init__(self, total: float | None = None) -> None:  # noqa: ARG002
            return

    class _ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    aiohttp_mod.ClientError = _ClientError
    aiohttp_mod.ClientTimeout = _ClientTimeout
    aiohttp_mod.ClientSession = _ClientSession
    sys.modules["aiohttp"] = aiohttp_mod

from ccxt.base.errors import NetworkError as CcxtNetworkError

from execution.paper_executor import PaperExecutor


class _FakeRiskManager:
    def __init__(self) -> None:
        self.balance = 10_000.0
        self.max_positions = 3
        self._open = 0

    def is_trading_halted(self) -> bool:
        return False

    def is_portfolio_dd_exceeded(self) -> bool:
        return False

    def can_open_position(self) -> bool:
        return self._open < self.max_positions

    def is_sector_exposed(self, symbol: str, open_symbols: list[str]) -> bool:  # noqa: ARG002
        return False

    def calculate_position_size(self, win_probability: float) -> float:  # noqa: ARG002
        return 1000.0

    def has_sufficient_balance(self, position_size: float) -> bool:
        return self.balance >= position_size

    def deduct(self, amount: float) -> None:
        self.balance -= amount

    def credit(self, amount: float) -> None:
        self.balance += amount

    def register_open(self) -> None:
        self._open += 1

    def register_close(self) -> None:
        self._open = max(0, self._open - 1)

    def record_daily_loss(self, loss: float) -> None:  # noqa: ARG002
        return


class _FakeDB:
    async def insert_open_trade(self, **kwargs) -> int:  # noqa: ARG002
        return 1

    async def close_trade(self, **kwargs) -> None:  # noqa: ARG002
        return


class _ExchangeBuyOkSellFail:
    async def set_leverage(self, leverage: int, symbol: str) -> None:  # noqa: ARG002
        return

    async def create_market_buy_order(self, symbol: str, amount: float, params: dict) -> dict:  # noqa: ARG002
        return {"id": "buy-ok"}

    async def create_market_sell_order(self, symbol: str, amount: float, params: dict) -> dict:  # noqa: ARG002
        raise CcxtNetworkError("simulated sell failure")


class ExecutorSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_live_close_sets_close_pending(self) -> None:
        rm = _FakeRiskManager()
        db = _FakeDB()
        ex = _ExchangeBuyOkSellFail()
        executor = PaperExecutor(db=db, risk_manager=rm, exchange=ex)

        opened = await executor.try_open_trade(
            entry_price=100.0,
            win_probability=0.8,
            symbol="BTC/USDT",
            sentiment_score=0.1,
        )
        self.assertTrue(opened)
        self.assertIn("BTC/USDT", executor.open_positions)

        # Force SL hit.
        result = await executor.check_and_close(current_price=95.0, symbol="BTC/USDT")
        self.assertIsNone(result)
        self.assertIn("BTC/USDT", executor.open_positions)
        self.assertTrue(executor.open_positions["BTC/USDT"].close_pending)

    async def test_ttl_exit_closes_position_in_paper_mode(self) -> None:
        rm = _FakeRiskManager()
        db = _FakeDB()
        executor = PaperExecutor(db=db, risk_manager=rm, exchange=None)

        opened = await executor.try_open_trade(
            entry_price=100.0,
            win_probability=0.8,
            symbol="ETH/USDT",
            sentiment_score=0.0,
        )
        self.assertTrue(opened)

        pos = executor.open_positions["ETH/USDT"]
        pos.max_ttl_hours = 0.0
        pos.entry_time = datetime.now(tz=timezone.utc) - timedelta(hours=1)

        result = await executor.check_ml_exit(
            current_price=101.0,
            ml_signal="HOLD",
            ml_probability=0.6,
            symbol="ETH/USDT",
        )
        self.assertIsNotNone(result)
        self.assertNotIn("ETH/USDT", executor.open_positions)

    async def test_retry_close_pending_positions_closes_when_price_available(self) -> None:
        rm = _FakeRiskManager()
        db = _FakeDB()
        ex = _ExchangeBuyOkSellFail()
        executor = PaperExecutor(db=db, risk_manager=rm, exchange=ex)

        opened = await executor.try_open_trade(
            entry_price=100.0,
            win_probability=0.8,
            symbol="SOL/USDT",
            sentiment_score=0.0,
        )
        self.assertTrue(opened)

        # First close attempt fails and marks position close_pending.
        await executor.check_and_close(current_price=95.0, symbol="SOL/USDT")
        self.assertIn("SOL/USDT", executor.open_positions)
        self.assertTrue(executor.open_positions["SOL/USDT"].close_pending)

        class _ExchangeSellOk(_ExchangeBuyOkSellFail):
            async def create_market_sell_order(self, symbol: str, amount: float, params: dict) -> dict:  # noqa: ARG002
                return {"id": "sell-ok"}

        executor._exchange = _ExchangeSellOk()
        closed = await executor.retry_close_pending_positions({"SOL/USDT": 96.0})
        self.assertEqual(closed, 1)
        self.assertNotIn("SOL/USDT", executor.open_positions)

