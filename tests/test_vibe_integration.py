"""
Tests de integracion completa de Vibe-Trading en ClawdBot.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────
# Test MCPClient robustness
# ─────────────────────────────────────────────────────────
class TestMCPClientRobust:
    def test_tool_timeout_uses_per_tool_config(self):
        """pattern_recognition usa 120s, get_market_data usa 30s."""
        from vibe.mcp_client import _TOOL_TIMEOUTS, _REQUEST_TIMEOUT
        assert _TOOL_TIMEOUTS["pattern_recognition"] == 120.0
        assert _TOOL_TIMEOUTS["backtest"] == 180.0
        assert _TOOL_TIMEOUTS.get("get_market_data", 30.0) <= 30.0
        assert _TOOL_TIMEOUTS.get("list_skills", _REQUEST_TIMEOUT) == 15.0

    @pytest.mark.asyncio
    async def test_unavailable_client_returns_none(self):
        """Si MCP no esta disponible, call_tool retorna None sin crashear."""
        from vibe.mcp_client import VibeMCPClient
        client = VibeMCPClient()
        result = await client.call_tool("pattern_recognition", {"symbol": "BTC/USDT"})
        assert result is None

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """shutdown() puede llamarse multiples veces sin error."""
        from vibe.mcp_client import VibeMCPClient
        client = VibeMCPClient()
        await client.stop()
        await client.stop()

    def test_client_has_tool_timeouts_dict(self):
        """El cliente tiene un dict de timeouts por tool con al menos 10 entradas."""
        from vibe.mcp_client import _TOOL_TIMEOUTS
        assert len(_TOOL_TIMEOUTS) >= 10
        for tool, timeout in _TOOL_TIMEOUTS.items():
            assert timeout > 0
            assert isinstance(tool, str)


# ─────────────────────────────────────────────────────────
# Test pattern recognition limits
# ─────────────────────────────────────────────────────────
class TestPatternRecognition:
    def test_max_ohlcv_rows_constant(self):
        """pattern_recognition nunca exporta mas de _MAX_OHLCV_ROWS filas."""
        from vibe.pattern_recognition import _MAX_OHLCV_ROWS
        assert _MAX_OHLCV_ROWS <= 1000
        assert _MAX_OHLCV_ROWS > 0


# ─────────────────────────────────────────────────────────
# Test scheduled tasks state persistence
# ─────────────────────────────────────────────────────────
class TestScheduledTasks:
    def test_vibe_state_persists_last_run(self, tmp_path, monkeypatch):
        """last_run se guarda en disco y se recupera entre reinicios."""
        from vibe import scheduled_tasks
        monkeypatch.setattr(scheduled_tasks, "_VIBE_STATE_FILE", tmp_path / "vibe_state.json")

        scheduled_tasks._save_vibe_state({"last_run_pattern": "2026-05-10T10:00:00+00:00"})
        loaded = scheduled_tasks._load_vibe_state()
        assert "last_run_pattern" in loaded

    def test_initial_sleep_respects_last_run(self, tmp_path, monkeypatch):
        """_initial_sleep_with_state respeta last_run para no duplicar tareas."""
        from vibe.scheduled_tasks import _initial_sleep_with_state
        from datetime import datetime, timezone

        state_file = tmp_path / "vibe_state.json"
        monkeypatch.setattr(
            "vibe.scheduled_tasks._VIBE_STATE_FILE", state_file
        )

        recent = datetime.now(timezone.utc).isoformat()
        state_file.write_text(json.dumps({"last_run_test": recent}), encoding="utf-8")

        sleep = _initial_sleep_with_state("test", default_sleep=60.0, interval_s=3600.0)
        assert sleep >= 5.0


# ─────────────────────────────────────────────────────────
# Test decision bridge safe-calls
# ─────────────────────────────────────────────────────────
class TestDecisionBridge:
    def test_compute_entry_score_returns_neutral_when_no_vibe_data(self):
        """Sin datos VIBE, compute_entry_score retorna neutral (1.0)."""
        from vibe.decision_bridge import compute_entry_score
        score = compute_entry_score({}, "BTC/USDT")
        assert score == 1.0

    def test_compute_entry_score_bounded(self):
        """compute_entry_score siempre esta en [0.0, 1.0]."""
        from vibe.decision_bridge import compute_entry_score
        state = {
            "vibe_patterns": {"BTC/USDT": {"text": "bullish breakout strong buy"}},
            "vibe_factors": {"text": "strong predictive high ic 0.25"},
            "vibe_journal_analysis": {"text": "disciplined consistent edge confirmed"},
            "vibe_backtest": {"BTC/USDT": {"text": "sharpe ratio 1.8 profitable"}},
        }
        score = compute_entry_score(state, "BTC/USDT")
        assert 0.0 <= score <= 1.0

    def test_compute_quality_multiplier_bounded(self):
        """compute_quality_multiplier esta siempre en [VIBE_QUALITY_MIN, VIBE_QUALITY_MAX]."""
        from vibe.decision_bridge import compute_quality_multiplier, _VIBE_QUALITY_MIN, _VIBE_QUALITY_MAX
        state = {
            "vibe_factors": {"text": "strong predictive"},
            "vibe_backtest": {"BTC/USDT": {"text": "sharpe ratio 2.0"}},
        }
        mult = compute_quality_multiplier(state, "BTC/USDT")
        assert _VIBE_QUALITY_MIN <= mult <= _VIBE_QUALITY_MAX

    def test_is_extreme_macro_veto_false_when_no_data(self):
        """Sin datos VIBE, veto retorna False."""
        from vibe.decision_bridge import is_extreme_macro_veto
        assert is_extreme_macro_veto({}) is False

    def test_check_vibe_force_exit_none_when_no_data(self):
        """Sin datos VIBE, force_exit retorna None."""
        from vibe.decision_bridge import check_vibe_force_exit
        assert check_vibe_force_exit({}, "BTC/USDT") is None


# ─────────────────────────────────────────────────────────
# Test feature bridge safe-calls
# ─────────────────────────────────────────────────────────
class TestFeatureBridge:
    def test_extract_vibe_features_returns_neutral_when_no_state(self):
        """Sin estado VIBE, retorna vector neutral de 4 features."""
        from vibe.feature_bridge import extract_vibe_features, VIBE_FEATURE_NEUTRAL
        features = extract_vibe_features(None, "BTC/USDT")
        assert features == list(VIBE_FEATURE_NEUTRAL)
        assert len(features) == 4

    def test_extract_vibe_features_bounded(self):
        """Todas las features retornadas estan en rangos seguros para XGBoost."""
        from vibe.feature_bridge import extract_vibe_features
        state = {
            "vibe_patterns": {"BTC/USDT": {"text": "bullish"}},
            "vibe_factors": {"text": "ic 0.15"},
            "vibe_journal_analysis": {"text": "disciplined"},
            "vibe_backtest": {"BTC/USDT": {"text": "sharpe 1.5"}},
        }
        features = extract_vibe_features(state, "BTC/USDT")
        assert len(features) == 4
        for f in features:
            assert -1.0 <= f <= 1.0


# ─────────────────────────────────────────────────────────
# Test swarm loops
# ─────────────────────────────────────────────────────────
class TestSwarmLoops:
    def test_parse_swarm_recommendation_neutral_when_none(self):
        """Sin resultado, parse_swarm_recommendation retorna NEUTRAL."""
        from vibe.swarm_loops import parse_swarm_recommendation
        assert parse_swarm_recommendation(None) == "NEUTRAL"
        assert parse_swarm_recommendation({}) == "NEUTRAL"

    def test_parse_swarm_recommendation_detects_buy(self):
        """Texto con 'buy' retorna BUY o STRONG_BUY."""
        from vibe.swarm_loops import parse_swarm_recommendation
        result = {"content": [{"type": "text", "text": "The signal is a strong buy."}]}
        rec = parse_swarm_recommendation(result)
        assert rec in ("BUY", "STRONG_BUY")

    def test_parse_swarm_recommendation_detects_exit(self):
        """Texto con 'exit' retorna EXIT o REDUCE."""
        from vibe.swarm_loops import parse_swarm_recommendation
        result = {"content": [{"type": "text", "text": "We should exit this position now."}]}
        rec = parse_swarm_recommendation(result)
        assert rec in ("EXIT", "REDUCE")


# ─────────────────────────────────────────────────────────
# Test journal analyzer
# ─────────────────────────────────────────────────────────
class TestJournalAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_journal_skips_when_no_file(self):
        """Si el journal no existe, analyze_journal retorna None sin crashear."""
        from vibe.journal_analyzer import analyze_journal
        from vibe.mcp_client import VibeMCPClient
        client = VibeMCPClient()
        result = await analyze_journal(client, journal_path=Path("/nonexistent/journal.csv"))
        assert result is None


# ─────────────────────────────────────────────────────────
# Test shadow account
# ─────────────────────────────────────────────────────────
class TestShadowAccount:
    def test_extract_shadow_id_parses_correctly(self):
        """_extract_shadow_id extrae shadow_id de respuestas tipicas."""
        from vibe.shadow_account import _extract_shadow_id
        result = {"shadow_id": "shd_abc123"}
        assert _extract_shadow_id(result) == "shd_abc123"

    def test_extract_shadow_id_handles_text_content(self):
        """_extract_shadow_id maneja respuestas con content/text."""
        from vibe.shadow_account import _extract_shadow_id
        result = {"content": [{"type": "text", "text": '{"shadow_id": "shd_xyz789"}'}]}
        assert _extract_shadow_id(result) == "shd_xyz789"

    def test_extract_shadow_id_returns_none_when_missing(self):
        """_extract_shadow_id retorna None cuando no hay shadow_id."""
        from vibe.shadow_account import _extract_shadow_id
        assert _extract_shadow_id({}) is None
        assert _extract_shadow_id(None) is None
