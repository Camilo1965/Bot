#!/usr/bin/env python3
"""
scripts/compare_retrain_diff.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare pre-retrain vs post-retrain `backtest_disk_loaded` JSON outputs.
If the portfolio composite score regresses by more than `--threshold-pct`,
restore the archived models from `--rollback-from` and exit code 2.

Used by `scripts/weekly_retrain.ps1` to gate every retrain cycle.

Usage:
    python scripts/compare_retrain_diff.py --pre logs/weekly_retrain/<ts>/pre.json \
                                           --post logs/weekly_retrain/<ts>/post.json \
                                           --threshold-pct -30 \
                                           --rollback-from models/archive/<ts>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.param_sweep import score as composite_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compare_retrain_diff")

MODELS_DIR = _REPO / "models"


def _portfolio_score(report: dict) -> float:
    """Sum per-symbol composite scores. Mirrors what Phase F evaluates."""
    total = 0.0
    for r in report.get("results", []):
        m = r.get("metrics")
        if not isinstance(m, dict):
            continue
        total += composite_score(m)
    return total


def _rollback(archive_dir: Path) -> int:
    """Copy archived models back to models/. Returns count restored."""
    if not archive_dir.is_dir():
        logger.error("Archive dir not found: %s", archive_dir)
        return 0
    count = 0
    for src in archive_dir.glob("*.json"):
        dst = MODELS_DIR / src.name
        try:
            shutil.copy2(src, dst)
            logger.info("Restored %s", src.name)
            count += 1
        except Exception as exc:
            logger.warning("Failed to restore %s: %s", src.name, exc)
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--threshold-pct", type=float, default=-30.0,
                        help="Negative %% delta in composite score that triggers rollback. Default -30.")
    parser.add_argument("--rollback-from", type=Path, required=True,
                        help="Archive directory created by weekly_retrain.ps1.")
    parser.add_argument("--no-rollback", action="store_true",
                        help="Report only; do not restore archive on regression.")
    args = parser.parse_args()

    if not args.pre.is_file() or not args.post.is_file():
        logger.error("Missing input: pre=%s post=%s", args.pre.is_file(), args.post.is_file())
        return 1

    pre_data = json.loads(args.pre.read_text(encoding="utf-8"))
    post_data = json.loads(args.post.read_text(encoding="utf-8"))

    pre_score = _portfolio_score(pre_data)
    post_score = _portfolio_score(post_data)
    delta_abs = post_score - pre_score
    delta_pct = (delta_abs / abs(pre_score) * 100.0) if pre_score != 0 else 0.0

    logger.info("=" * 70)
    logger.info("Retrain comparison")
    logger.info("=" * 70)
    logger.info("Pre  portfolio composite : %.2f", pre_score)
    logger.info("Post portfolio composite : %.2f", post_score)
    logger.info("Delta                    : %+.2f  (%.1f%%)", delta_abs, delta_pct)
    logger.info("Threshold (regression)   : %.1f%%", args.threshold_pct)

    if delta_pct < args.threshold_pct:
        logger.error("REGRESSION DETECTED. Delta %.1f%% < threshold %.1f%%.", delta_pct, args.threshold_pct)
        if args.no_rollback:
            logger.warning("--no-rollback set; not restoring archive.")
            return 2
        restored = _rollback(args.rollback_from)
        logger.error("Rollback complete: %d files restored from %s.", restored, args.rollback_from)
        return 2

    logger.info("OK — no regression beyond threshold. Keeping new models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
