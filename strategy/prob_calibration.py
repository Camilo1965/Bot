"""
strategy.prob_calibration
~~~~~~~~~~~~~~~~~~~~~~~~~

Post-hoc probability calibration for XGBoost classifiers.

XGBoost with high ``scale_pos_weight`` produces raw probabilities that are
compressed toward 0 — the model's AUC may be good but ``predict_proba``
outputs do not reflect true win rates.  Isotonic regression (piecewise
monotone function) corrects this without any retraining of the base model.

Usage (inference):
    from strategy.prob_calibration import calibrate_probability
    cal_prob = calibrate_probability(raw_prob, symbol="ETH/USDT")

Offline fitting (run once after every model retrain):
    from strategy.prob_calibration import fit_and_save_calibration
    fit_and_save_calibration("ETH/USDT", y_true, y_prob)

Calibration files live at ``models/{SYMBOL}_calibration.json`` as simple
lists of ``(x, y)`` breakpoints.  Missing file → identity transform.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"

# In-memory cache: symbol → (x_breakpoints, y_breakpoints)
_calibration_cache: dict[str, tuple[list[float], list[float]] | None] = {}


def _calibration_path(symbol: str) -> Path:
    return MODELS_DIR / f"{symbol.replace('/', '_')}_calibration.json"


def load_calibration(symbol: str) -> tuple[list[float], list[float]] | None:
    """Return ``(x_points, y_points)`` isotonic breakpoints or ``None``."""
    if symbol in _calibration_cache:
        return _calibration_cache[symbol]
    path = _calibration_path(symbol)
    if not path.is_file():
        _calibration_cache[symbol] = None
        return None
    try:
        data = json.loads(path.read_text())
        xs = [float(v) for v in data["x"]]
        ys = [float(v) for v in data["y"]]
        _calibration_cache[symbol] = (xs, ys)
        logger.info("[CALIB] Loaded calibration for %s (%d breakpoints)", symbol, len(xs))
        return xs, ys
    except Exception as exc:
        logger.warning("[CALIB] Failed to load calibration for %s: %s", symbol, exc)
        _calibration_cache[symbol] = None
        return None


def calibrate_probability(raw_prob: float, symbol: str) -> float:
    """Map *raw_prob* through the isotonic calibration for *symbol*.

    Returns *raw_prob* unchanged when no calibration file exists.
    """
    cal = load_calibration(symbol)
    if cal is None:
        return raw_prob
    xs, ys = cal
    if len(xs) < 2:
        return raw_prob
    # Linear interpolation between breakpoints (clamp to [0, 1])
    calibrated = float(np.interp(raw_prob, xs, ys))
    return max(0.0, min(1.0, calibrated))


def fit_and_save_calibration(
    symbol: str,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 30,
) -> Path:
    """Fit isotonic calibration and save breakpoints to disk.

    Parameters
    ----------
    symbol:  Trading symbol, e.g. ``"ETH/USDT"``.
    y_true:  Ground-truth binary labels (0 or 1).
    y_prob:  Raw predicted probabilities from the XGBoost model.
    n_bins:  Resolution of the output calibration curve (default 30).

    Returns
    -------
    Path to the saved ``.json`` calibration file.

    Notes
    -----
    Requires ``scikit-learn`` (``sklearn.isotonic.IsotonicRegression``).
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as exc:
        raise ImportError("scikit-learn is required to fit calibration: pip install scikit-learn") from exc

    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(y_prob_arr, y_true_arr)

    # Sample the fitted curve at evenly-spaced points for compact storage
    x_grid = np.linspace(0.0, 1.0, n_bins)
    y_grid = iso.transform(x_grid)

    path = _calibration_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"x": x_grid.tolist(), "y": y_grid.tolist()}, indent=2))

    # Invalidate cache
    _calibration_cache.pop(symbol, None)

    brier_raw = float(np.mean((y_prob_arr - y_true_arr) ** 2))
    brier_cal = float(np.mean((iso.transform(y_prob_arr) - y_true_arr) ** 2))
    logger.info(
        "[CALIB] Saved calibration for %s → %s  Brier: %.4f → %.4f (n=%d)",
        symbol, path.name, brier_raw, brier_cal, len(y_true_arr),
    )
    return path
