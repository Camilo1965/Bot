"""
tests.test_retrain_model
~~~~~~~~~~~~~~~~~~~~~~~~

Regresión para ``scripts/retrain_model``.

Mockea la carga de datos de DB y valida el pipeline de entrenamiento
y el Quality Gate de AUC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.retrain_model import _add_vibe_features, retrain_and_validate


@pytest.fixture
def dummy_ohlcv() -> pd.DataFrame:
    """Generate 300 rows of deterministic OHLCV data."""
    n = 300
    t = pd.date_range(end="2026-01-01", periods=n, freq="15min", tz="UTC")
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.normal(0, 0.5, size=n))
    high = close + np.abs(np.random.normal(0, 0.3, size=n))
    low = close - np.abs(np.random.normal(0, 0.3, size=n))
    open_p = close + np.random.normal(0, 0.1, size=n)
    volume = np.abs(np.random.normal(1000, 200, size=n))
    return pd.DataFrame({
        "timestamp": t,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestAddVibeFeatures:
    def test_adds_four_columns(self, dummy_ohlcv: pd.DataFrame) -> None:
        from strategy.quant_features import VIBE_FEATURE_COLS
        df = _add_vibe_features(dummy_ohlcv.copy())
        for col in VIBE_FEATURE_COLS:
            assert col in df.columns

    def test_neutral_fallback_present(self, dummy_ohlcv: pd.DataFrame) -> None:
        from vibe.feature_bridge import VIBE_FEATURE_NEUTRAL
        df = _add_vibe_features(dummy_ohlcv.copy(), add_noise=False)
        # First row should match neutral values exactly
        assert df["vibe_pattern_score"].iloc[0] == pytest.approx(VIBE_FEATURE_NEUTRAL[0])
        assert df["vibe_factor_ic"].iloc[0] == pytest.approx(VIBE_FEATURE_NEUTRAL[1])
        assert df["vibe_journal_health"].iloc[0] == pytest.approx(VIBE_FEATURE_NEUTRAL[2])
        assert df["vibe_backtest_sharpe"].iloc[0] == pytest.approx(VIBE_FEATURE_NEUTRAL[3])

    def test_noise_introduces_variance(self, dummy_ohlcv: pd.DataFrame) -> None:
        df = _add_vibe_features(dummy_ohlcv.copy(), add_noise=True)
        # With noise, std should be > 0
        assert df["vibe_pattern_score"].std() > 0.0


class TestRetrainPipeline:
    @patch("scripts.retrain_model._fetch_ohlcv")
    def test_pipeline_saves_model_when_auc_high(self, mock_fetch: Any, dummy_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        """Test 1: AUC alto → modelo guardado."""
        mock_fetch.return_value = dummy_ohlcv
        model_path = tmp_path / "BTC_USDT_v1.json"

        ok = retrain_and_validate(
            symbol="BTC/USDT",
            model_path=model_path,
            lookback_days=90,
            min_auc=0.50,  # Low bar so dummy data passes
            use_vibe=False,
        )
        assert ok is True
        assert model_path.exists()
        assert model_path.with_suffix(".meta.json").exists()

    @patch("scripts.retrain_model._fetch_ohlcv")
    def test_pipeline_rejects_low_auc(self, mock_fetch: Any, dummy_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        """Test 2: AUC bajo → modelo rechazado (Quality Gate)."""
        mock_fetch.return_value = dummy_ohlcv
        model_path = tmp_path / "BTC_USDT_v1.json"

        ok = retrain_and_validate(
            symbol="BTC/USDT",
            model_path=model_path,
            lookback_days=90,
            min_auc=0.99,  # Impossibly high bar
            use_vibe=False,
        )
        assert ok is False
        assert not model_path.exists()

    @patch("scripts.retrain_model._fetch_ohlcv")
    def test_vibe_mode_saves_as_v2(self, mock_fetch: Any, dummy_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        """Test 3: --vibe guarda con 16 features y path _v2."""
        mock_fetch.return_value = dummy_ohlcv
        model_path = tmp_path / "BTC_USDT_v2.json"

        ok = retrain_and_validate(
            symbol="BTC/USDT",
            model_path=model_path,
            lookback_days=90,
            min_auc=0.50,
            use_vibe=True,
        )
        assert ok is True
        meta = json.loads(model_path.with_suffix(".meta.json").read_text())
        assert meta["vibe_enabled"] is True
        assert len(meta["features"]) == 16  # 12 base + 4 VIBE

    @patch("scripts.retrain_model._fetch_ohlcv")
    def test_not_enough_data_returns_false(self, mock_fetch: Any, tmp_path: Path) -> None:
        """Test 4: Datos insuficientes → aborta gracefully."""
        mock_fetch.return_value = pd.DataFrame({
            "timestamp": pd.date_range(end="2026-01-01", periods=10, freq="15min", tz="UTC"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        })
        model_path = tmp_path / "BTC_USDT_v1.json"

        ok = retrain_and_validate(
            symbol="BTC/USDT",
            model_path=model_path,
            lookback_days=90,
            min_auc=0.50,
            use_vibe=False,
        )
        assert ok is False

    @patch("scripts.retrain_model._fetch_ohlcv")
    def test_meta_contains_expected_keys(self, mock_fetch: Any, dummy_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
        """Test 5: El archivo .meta.json contiene las claves esperadas."""
        mock_fetch.return_value = dummy_ohlcv
        model_path = tmp_path / "BTC_USDT_v1.json"

        retrain_and_validate(
            symbol="BTC/USDT",
            model_path=model_path,
            lookback_days=90,
            min_auc=0.50,
            use_vibe=False,
        )

        meta_path = model_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        assert "auc" in meta
        assert "trained_at" in meta
        assert "symbol" in meta
        assert "samples" in meta
        assert "splits" in meta
        assert "features" in meta
        assert "vibe_enabled" in meta
        assert meta["symbol"] == "BTC/USDT"
