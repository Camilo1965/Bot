"""Tests for vibe.mcp_client and vibe.scheduled_tasks."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibe.mcp_client import VibeMCPClient


@pytest.fixture
def client():
    return VibeMCPClient()


class TestVibeMCPClient:
    def test_initial_state(self, client):
        assert not client.available
        assert client._proc is None

    @pytest.mark.asyncio
    async def test_start_binary_not_found(self, client):
        with patch("shutil.which", return_value=None):
            result = await client.start()
            assert result is False
            assert not client.available

    @pytest.mark.asyncio
    async def test_call_tool_when_unavailable(self, client):
        result = await client.call_tool("backtest", {"run_dir": "/tmp/test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, client):
        await client.stop()
        assert not client.available

    @pytest.mark.asyncio
    async def test_backtest_when_unavailable(self, client):
        result = await client.backtest("/tmp/test_run")
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_trade_journal_when_unavailable(self, client):
        result = await client.analyze_trade_journal("/tmp/fake.csv")
        assert result is None

    @pytest.mark.asyncio
    async def test_pattern_recognition_when_unavailable(self, client):
        result = await client.pattern_recognition("/tmp/test_run")
        assert result is None

    @pytest.mark.asyncio
    async def test_factor_analysis_when_unavailable(self, client):
        result = await client.factor_analysis(
            codes=["BTC/USDT"], factor_name="close",
            start_date="2025-01-01", end_date="2025-04-01",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_shadow_strategy_when_unavailable(self, client):
        result = await client.extract_shadow_strategy("/tmp/fake.csv")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_shadow_backtest_when_unavailable(self, client):
        result = await client.run_shadow_backtest("shd_test_123")
        assert result is None

    @pytest.mark.asyncio
    async def test_render_shadow_report_when_unavailable(self, client):
        result = await client.render_shadow_report("shd_test_123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_data_when_unavailable(self, client):
        result = await client.get_market_data(
            codes=["BTC/USDT"], start_date="2025-01-01", end_date="2025-04-01",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_list_skills_when_unavailable(self, client):
        result = await client.list_skills()
        assert result is None


class TestJournalAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_journal_no_file(self):
        from vibe.journal_analyzer import analyze_journal

        client = VibeMCPClient()
        result = await analyze_journal(client, Path("/nonexistent/path.csv"))
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_journal_unavailable_client(self, tmp_path):
        from vibe.journal_analyzer import analyze_journal

        csv_file = tmp_path / "trade_journal.csv"
        csv_file.write_text("trade_id,symbol,entry_time\n1,BTC/USDT,2024-01-01\n")
        client = VibeMCPClient()
        result = await analyze_journal(client, csv_file)
        assert result is None


class TestBacktest:
    @pytest.mark.asyncio
    async def test_run_backtest_unavailable(self):
        from vibe.backtest import run_backtest

        client = VibeMCPClient()
        result = await run_backtest(client, "BTC/USDT")
        assert result is None


class TestPatternRecognition:
    @pytest.mark.asyncio
    async def test_detect_patterns_unavailable(self):
        from vibe.pattern_recognition import detect_patterns

        client = VibeMCPClient()
        result = await detect_patterns(client, "BTC/USDT")
        assert result is None


class TestFactorResearch:
    @pytest.mark.asyncio
    async def test_analyze_factors_unavailable(self):
        from vibe.factor_research import analyze_factors

        client = VibeMCPClient()
        result = await analyze_factors(client, symbols=["BTC/USDT"])
        assert result is None


class TestShadowAccount:
    @pytest.mark.asyncio
    async def test_extract_and_backtest_shadow_no_file(self):
        from vibe.shadow_account import extract_and_backtest_shadow

        client = VibeMCPClient()
        result = await extract_and_backtest_shadow(client, Path("/nonexistent.csv"))
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_and_backtest_shadow_unavailable(self, tmp_path):
        from vibe.shadow_account import extract_and_backtest_shadow

        csv_file = tmp_path / "trade_journal.csv"
        csv_file.write_text("trade_id,symbol,entry_time\n1,BTC/USDT,2024-01-01\n")
        client = VibeMCPClient()
        result = await extract_and_backtest_shadow(client, csv_file)
        assert result is None


class TestScheduledTasks:
    @pytest.mark.asyncio
    async def test_journal_analysis_loop_skips_when_unavailable(self):
        from vibe.scheduled_tasks import journal_analysis_loop

        client = VibeMCPClient()
        state: dict = {}
        await journal_analysis_loop(client, state, interval_s=0.01)
        assert "vibe_journal_analysis" not in state

    @pytest.mark.asyncio
    async def test_pattern_detection_loop_skips_when_unavailable(self):
        from vibe.scheduled_tasks import pattern_detection_loop

        client = VibeMCPClient()
        state: dict = {}
        await pattern_detection_loop(client, state, ["BTC/USDT"], interval_s=0.01)
        assert "vibe_patterns" not in state

    @pytest.mark.asyncio
    async def test_factor_analysis_loop_skips_when_unavailable(self):
        from vibe.scheduled_tasks import factor_analysis_loop

        client = VibeMCPClient()
        state: dict = {}
        await factor_analysis_loop(client, state, ["BTC/USDT"], interval_s=0.01)
        assert "vibe_factors" not in state

    @pytest.mark.asyncio
    async def test_weekly_backtest_loop_skips_when_unavailable(self):
        from vibe.scheduled_tasks import weekly_backtest_loop

        client = VibeMCPClient()
        state: dict = {}
        await weekly_backtest_loop(client, state, ["BTC/USDT"], interval_s=0.01)
        assert "vibe_backtest" not in state

    @pytest.mark.asyncio
    async def test_shadow_account_loop_skips_when_unavailable(self):
        from vibe.scheduled_tasks import shadow_account_loop

        client = VibeMCPClient()
        state: dict = {}
        await shadow_account_loop(client, state, interval_s=0.01)
        assert "vibe_shadow_report" not in state


class TestVibeStatePersistence:
    def test_initial_sleep_respects_last_run(self, tmp_path):
        from vibe.scheduled_tasks import _initial_sleep_with_state, _save_vibe_state

        state_file = tmp_path / "vibe_state.json"
        with patch("vibe.scheduled_tasks._VIBE_STATE_FILE", state_file):
            now = datetime.now(tz=timezone.utc)
            _save_vibe_state({"last_run_pattern": now.isoformat()})
            sleep = _initial_sleep_with_state("pattern", 120, 900)
            assert 0 <= sleep <= 900

    def test_record_run_persists_timestamp(self, tmp_path):
        from vibe.scheduled_tasks import _record_run, _load_vibe_state

        state_file = tmp_path / "vibe_state.json"
        with patch("vibe.scheduled_tasks._VIBE_STATE_FILE", state_file):
            _record_run("journal")
            state = _load_vibe_state()
            assert "last_run_journal" in state


class TestVibeMCPResilience:
    @pytest.mark.asyncio
    async def test_mcp_timeout_doesnt_crash_bot(self, client):
        """Timeout in _call must return None, not raise."""
        with patch.object(client, "_call", new_callable=AsyncMock, return_value=None):
            result = await client.call_tool("pattern_recognition", {"run_dir": "/tmp"})
            assert result is None

    @pytest.mark.asyncio
    async def test_mcp_retries_with_backoff(self, client):
        """Failed calls should retry up to max_retries times."""
        client._available = True
        client._proc = MagicMock()
        client._proc.poll = MagicMock(return_value=None)
        with patch.object(client, "_call", new_callable=AsyncMock, side_effect=[None, None, {"ok": True}]) as mock_call:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.call_tool("backtest", {"run_dir": "/tmp"}, max_retries=2)
                assert result == {"ok": True}
                assert mock_call.call_count == 3

    @pytest.mark.asyncio
    async def test_pattern_recognition_limits_ohlcv_rows(self):
        """Pattern recognition must request ≤ 1000 OHLCV rows."""
        from vibe.pattern_recognition import _MAX_OHLCV_ROWS
        assert _MAX_OHLCV_ROWS == 1000