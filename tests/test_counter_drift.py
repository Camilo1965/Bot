"""Tests for RiskManager counter-drift fix."""

from __future__ import annotations

import pytest

from risk.risk_manager import RiskManager


class TestRiskManagerCounter:
    def test_open_count_reads_from_internal_when_no_fn(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        rm.register_open(risk_usd=100.0)
        assert rm.open_count == 1

    def test_open_count_reads_from_external_fn(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3, position_counter_fn=lambda: 5)
        assert rm.open_count == 5

    def test_open_count_correct_after_external_close(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3, position_counter_fn=lambda: 0)
        rm.register_open(risk_usd=100.0)
        # External fn says 0, so property auto-syncs
        assert rm.open_count == 0

    def test_no_drift_after_10_open_close_cycles(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        for _ in range(10):
            rm.register_open(risk_usd=50.0)
            assert rm.open_count == 1
            rm.register_close(risk_usd=50.0)
            assert rm.open_count == 0

    def test_set_position_counter_fn_updates_source(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=3)
        rm.register_open(risk_usd=100.0)
        assert rm.open_count == 1
        rm.set_position_counter_fn(lambda: 0)
        assert rm.open_count == 0

    def test_can_open_position_uses_live_count(self):
        rm = RiskManager(initial_balance=10_000.0, max_positions=2, position_counter_fn=lambda: 2)
        assert not rm.can_open_position()
        rm.set_position_counter_fn(lambda: 1)
        assert rm.can_open_position()
