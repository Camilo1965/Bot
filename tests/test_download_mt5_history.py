"""
tests.test_download_mt5_history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Smoke tests for ``scripts/download_mt5_history``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.download_mt5_history import _mt5_tf, _resolve_symbol


class TestTimeframeMapping:
    def test_15m_returns_15(self) -> None:
        assert _mt5_tf("15m") == 15

    def test_1h_returns_16385(self) -> None:
        assert _mt5_tf("1h") == 16385

    def test_1d_returns_16408(self) -> None:
        assert _mt5_tf("1d") == 16408

    def test_default_on_unknown(self) -> None:
        assert _mt5_tf("unknown") == 15


class TestSymbolResolution:
    def test_btc_usdt(self) -> None:
        result = _resolve_symbol("BTC/USDT")
        assert result is not None
        assert "BTC" in result

    def test_eth_usdt(self) -> None:
        result = _resolve_symbol("ETH/USDT")
        assert result is not None
        assert "ETH" in result

    def test_unknown_returns_none(self) -> None:
        result = _resolve_symbol("ZZZ/XYZ")
        assert result is None
