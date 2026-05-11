"""Tests for scripts/monthly_reset.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.monthly_reset import (
    calculate_stats,
    can_reset_safely,
    generate_html_report,
    verify_clean_state,
)


class TestMonthlyReset:
    def test_reset_blocked_with_open_positions(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"positions": {"BTC/USDT": {}}}))
        with patch("scripts.monthly_reset._LOGS_DIR", tmp_path):
            assert not can_reset_safely()

    def test_reset_allowed_when_empty(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"positions": {}}))
        with patch("scripts.monthly_reset._LOGS_DIR", tmp_path):
            assert can_reset_safely()

    def test_stats_pnl_winrate_correct(self, tmp_path):
        journal = tmp_path / "trade_journal.csv"
        now = datetime.now(tz=timezone.utc).isoformat()
        journal.write_text(
            "trade_id,symbol,entry_time,exit_time,entry_price,exit_price,position_size,gross_pnl,net_pnl,exit_reason,ml_confidence,duration_minutes\n"
            f"1,BTC/USDT,{now},{now},100,110,1000,100,100,tp,0.7,10\n"
            f"2,ETH/USDT,{now},{now},200,190,1000,-50,-50,sl,0.6,5\n"
        )
        month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
        with patch("scripts.monthly_reset._LOGS_DIR", tmp_path):
            stats = calculate_stats(month)
        assert stats["total_pnl"] == 50.0
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["winrate"] == 50.0

    def test_html_report_generated(self, tmp_path):
        stats = {
            "month": "2026-05",
            "total_pnl": 100.0,
            "winrate": 50.0,
            "wins": 1,
            "losses": 1,
            "total": 2,
            "by_symbol": {"BTC/USDT": {"pnl": 100.0, "count": 1}},
            "best_trade": 100.0,
            "worst_trade": -50.0,
        }
        out = tmp_path / "report.html"
        generate_html_report(stats, "2026-05", out)
        assert out.exists()
        content = out.read_text()
        assert "2026-05" in content
        assert "BTC/USDT" in content

    def test_vibe_state_cleared_on_reset(self, tmp_path):
        vibe_state = tmp_path / "vibe_state.json"
        vibe_state.write_text('{"last_run_pattern": "2026-05-01"}')
        from scripts.monthly_reset import reset_vibe_state
        with patch("scripts.monthly_reset._LOGS_DIR", tmp_path):
            reset_vibe_state()
        assert not vibe_state.exists()

    def test_verify_clean_state(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"positions": {}}))
        with patch("scripts.monthly_reset._LOGS_DIR", tmp_path):
            assert verify_clean_state()
