"""
risk.drift_detector
~~~~~~~~~~~~~~~~~~~

Two-sample KS test between live raw_prob observations and the OOS
reference distribution captured at last retrain time.

If KS-stat > 0.15 and p-value < 0.01 → data drift detected:
  1. Log to logs/drift_events.log (JSONL)
  2. Auto-demote symbol via KillSwitch for 24h
  3. Notify via vibe.notifier (non-blocking)

Reference distribution file: models/{SYM}_prob_ref.json
Created by scripts/retrain_model.py after each retrain.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO / "models"
_DRIFT_LOG = _REPO / "logs" / "drift_events.log"


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(event: dict[str, Any]) -> None:
    try:
        _DRIFT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DRIFT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({**event, "ts": _iso()}) + "\n")
    except Exception as exc:
        logger.error("DriftDetector: event log write failed: %s", exc)


# Pure Python KS-test (no scipy dependency) ──────────────────────────────────
def _ks_2sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns (ks_stat, p_value approx)."""
    a_s = np.sort(a)
    b_s = np.sort(b)
    n1, n2 = len(a_s), len(b_s)
    combined = np.concatenate([a_s, b_s])
    combined.sort()

    cdf1 = np.searchsorted(a_s, combined, side="right") / n1
    cdf2 = np.searchsorted(b_s, combined, side="right") / n2
    ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic p-value (Kolmogorov distribution)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lambda_ = (en + 0.12 + 0.11 / en) * ks_stat
    # P(D > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k-1} exp(-2 k^2 lambda^2)
    p = 0.0
    for k in range(1, 50):
        p += ((-1) ** (k - 1)) * math.exp(-2 * (k ** 2) * (lambda_ ** 2))
    p_value = max(0.0, min(1.0, 2 * p))
    return ks_stat, p_value


@dataclass
class DriftResult:
    symbol: str
    ks_stat: float
    p_value: float
    n_live: int
    n_ref: int
    drift_detected: bool
    reason: str = ""


@dataclass
class DriftDetector:
    symbol: str
    window_size: int = 200
    ks_threshold: float = 0.15
    p_threshold: float = 0.01
    _observations: list[float] = field(default_factory=list, repr=False)

    def add_observation(self, raw_prob: float) -> None:
        self._observations.append(float(raw_prob))
        if len(self._observations) > self.window_size:
            self._observations.pop(0)

    def reference_dist(self) -> np.ndarray | None:
        key = self.symbol.replace("/", "_")
        ref_path = MODELS_DIR / f"{key}_prob_ref.json"
        if not ref_path.is_file():
            return None
        try:
            data = json.loads(ref_path.read_text(encoding="utf-8"))
            probs = data.get("raw_probs", [])
            if len(probs) < 30:
                return None
            return np.array(probs, dtype=np.float64)
        except Exception as exc:
            logger.warning("[DriftDetector] Failed to load ref dist for %s: %s", self.symbol, exc)
            return None

    def check(self) -> DriftResult:
        n_live = len(self._observations)
        ref = self.reference_dist()

        if n_live < 50:
            return DriftResult(self.symbol, 0.0, 1.0, n_live, 0, False, "insufficient_live_obs")
        if ref is None:
            return DriftResult(self.symbol, 0.0, 1.0, n_live, 0, False, "no_reference_dist")

        live = np.array(self._observations, dtype=np.float64)
        ks_stat, p_value = _ks_2sample(live, ref)

        drift = ks_stat > self.ks_threshold and p_value < self.p_threshold
        reason = f"ks={ks_stat:.3f} p={p_value:.4f}" if drift else "ok"

        result = DriftResult(
            symbol=self.symbol,
            ks_stat=round(ks_stat, 4),
            p_value=round(p_value, 6),
            n_live=n_live,
            n_ref=int(len(ref)),
            drift_detected=drift,
            reason=reason,
        )

        if drift:
            logger.warning(
                "[DRIFT] %s detected — ks=%.3f p=%.4f (n_live=%d n_ref=%d)",
                self.symbol, ks_stat, p_value, n_live, len(ref),
            )
            _log_event({
                "event": "drift_detected",
                "symbol": self.symbol,
                "ks_stat": round(ks_stat, 4),
                "p_value": round(p_value, 6),
                "n_live": n_live,
                "n_ref": int(len(ref)),
            })
            self._on_drift(ks_stat, p_value)
        else:
            logger.debug("[DriftDetector] %s OK (ks=%.3f p=%.4f)", self.symbol, ks_stat, p_value)

        return result

    def _on_drift(self, ks_stat: float, p_value: float) -> None:
        try:
            from risk.kill_switch import get_kill_switch
            ks = get_kill_switch()
            ks.demote_symbol(self.symbol, threshold=0.95)
        except Exception as exc:
            logger.error("[DriftDetector] KillSwitch demote failed: %s", exc)

        try:
            import asyncio
            from vibe.notifier import notify_drift
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(notify_drift(self.symbol, ks_stat, p_value))
            except RuntimeError:
                asyncio.run(notify_drift(self.symbol, ks_stat, p_value))
        except Exception as exc:
            logger.debug("[DriftDetector] Telegram notify failed (non-fatal): %s", exc)


# ── Registry ──────────────────────────────────────────────────────────────────
_detectors: dict[str, DriftDetector] = {}


def get_detector(symbol: str, window_size: int = 200) -> DriftDetector:
    if symbol not in _detectors:
        _detectors[symbol] = DriftDetector(symbol=symbol, window_size=window_size)
    return _detectors[symbol]


def save_reference_dist(symbol: str, raw_probs: np.ndarray) -> Path:
    """Save OOS raw probs as the reference distribution for drift detection.
    Called by scripts/retrain_model.py after each successful retrain.
    """
    key = symbol.replace("/", "_")
    path = MODELS_DIR / f"{key}_prob_ref.json"
    payload = {
        "symbol": symbol,
        "saved_at": _iso(),
        "n_samples": int(len(raw_probs)),
        "mean": float(np.mean(raw_probs)),
        "std": float(np.std(raw_probs)),
        "raw_probs": [round(float(p), 6) for p in raw_probs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("[DriftDetector] Reference dist saved → %s (%d samples)", path, len(raw_probs))
    return path
