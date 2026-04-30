"""Tests for :mod:`bot.dashboard_helpers`."""

from __future__ import annotations

from bot.dashboard_helpers import compute_rsi, mt5_dashboard_mark


def test_mt5_dashboard_mark_prefers_mid() -> None:
    state = {"mt5_last_quote": {"BTC/USDT": {"bid": 100.0, "ask": 102.0, "mid": 101.0}}}
    assert mt5_dashboard_mark(state, "BTC/USDT", 50.0) == 101.0


def test_mt5_dashboard_mark_respects_bid_env(monkeypatch) -> None:
    monkeypatch.setenv("MT5_DASHBOARD_PRICE", "bid")
    state = {"mt5_last_quote": {"ETH/USDT": {"bid": 99.5, "ask": 100.5, "mid": 100.0}}}
    assert mt5_dashboard_mark(state, "ETH/USDT", 1.0) == 99.5


def test_mt5_dashboard_mark_fallback_to_candle() -> None:
    assert mt5_dashboard_mark({}, "BTC/USDT", 42.0) == 42.0


def test_compute_rsi_insufficient_bars() -> None:
    assert compute_rsi([100.0, 101.0], period=14) is None


def test_compute_rsi_flat_series_high_rsi() -> None:
    flat = [100.0] * 20
    rsi = compute_rsi(flat)
    assert rsi is not None
    assert rsi == 100.0
