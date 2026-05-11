"""
scripts/bootstrap_models
~~~~~~~~~~~~~~~~~~~~~~~~

Generates synthetic OHLCV data and trains v1/v2 models locally when
TimescaleDB is unavailable (e.g. CI or local dev without Docker).

Usage:
    python scripts/bootstrap_models.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.retrain_model import retrain_and_validate

logger = logging.getLogger("clawdbot.bootstrap")


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    t = pd.date_range(end="2026-01-01", periods=n, freq="15min", tz="UTC")
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


def main() -> int:
    from unittest.mock import patch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["BTC/USDT", "ETH/USDT"]
    dummy_df = _make_ohlcv(n=400, seed=42)

    results: dict[str, dict] = {}

    for symbol in symbols:
        safe = symbol.replace("/", "_")
        for use_vibe, suffix in ((False, "_v1"), (True, "_v2")):
            model_path = models_dir / f"{safe}{suffix}.json"
            with patch("scripts.retrain_model._fetch_ohlcv", return_value=dummy_df):
                ok = retrain_and_validate(
                    symbol=symbol,
                    model_path=model_path,
                    lookback_days=90,
                    min_auc=0.50,
                    use_vibe=use_vibe,
                )
            meta_path = model_path.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                results[f"{safe}{suffix}"] = {
                    "saved": ok,
                    "auc": meta.get("auc"),
                    "samples": meta.get("samples"),
                    "features": meta.get("features"),
                    "vibe_enabled": meta.get("vibe_enabled"),
                }

    print("\n" + "=" * 60)
    print("BOOTSTRAP RESULTS")
    print("=" * 60)
    for name, data in results.items():
        status = "SAVED" if data["saved"] else "REJECTED"
        print(f"{name}: {status} | AUC={data['auc']:.4f} | samples={data['samples']} | features={len(data['features'])}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
