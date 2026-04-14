import unittest
import sys
import types

# Lightweight ccxt stubs so importing executors works without ccxt installed.
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

from execution.mt5_executor import MT5Executor


class _FakeRiskManager:
    def __init__(self) -> None:
        self.balance = 1000.0
        self.max_positions = 3


class _FakeDB:
    async def insert_open_trade(self, **kwargs) -> int:  # noqa: ARG002
        return 1


class MT5MappingTests(unittest.TestCase):
    def test_local_symbol_from_broker(self) -> None:
        self.assertEqual(MT5Executor._local_symbol_from_broker("BTCUSD-T"), "BTC/USDT")
        self.assertEqual(MT5Executor._local_symbol_from_broker("UNKNOWN"), "UNKNOWN")

    def test_resolve_accepts_direct_broker_symbol(self) -> None:
        executor = MT5Executor(
            db=_FakeDB(),
            risk_manager=_FakeRiskManager(),
            live=False,
        )
        resolved = executor._resolve_symbol("BTCUSD-T")
        self.assertEqual(resolved, "BTCUSD-T")

