"""Tests for vibe.mcp_client and vibe.scheduled_tasks."""

from __future__ import annotations

import asyncio
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
        result = await client.call_tool("backtest", {"prompt": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, client):
        await client.stop()
        assert not client.available

    @pytest.mark.asyncio
    async def test_backtest_when_unavailable(self, client):
        result = await client.backtest("test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_trade_journal_when_unavailable(self, client):
        result = await client.analyze_trade_journal("/tmp/fake.csv")
        assert result is None

    @pytest.mark.asyncio
    async def test_pattern_recognition_when_unavailable(self, client):
        result = await client.pattern_recognition("BTC/USDT", "test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_factor_analysis_when_unavailable(self, client):
        result = await client.factor_analysis("test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_shadow_strategy_when_unavailable(self, client):
        result = await client.extract_shadow_strategy("/tmp/fake.csv")
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
        result = await run_backtest(client, "test prompt")
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
        result = await analyze_factors(client, "BTC/USDT")
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