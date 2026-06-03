#!/usr/bin/env python3
"""
scripts/retrain_all.py
~~~~~~~~~~~~~~~~~~~~~~

One-shot retraining pipeline for all 5 symbols.
Downloads data from Binance via CCXT (no API key required), trains
direction + regime models, runs walk-forward validation, and prints a
summary with recommended prob_threshold per symbol.

Usage:
    python scripts/retrain_all.py
    python scripts/retrain_all.py --days 360 --min-auc 0.55

Requires network access to api.binance.com (run locally, not in
sandboxed cloud environments).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("retrain_all")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT"]


def _run(cmd: list[str]) -> int:
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        logger.error("Command failed with code %d", result.returncode)
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrain all 5 symbol models end-to-end.")
    ap.add_argument("--days", type=int, default=360, help="Lookback days (default: 360)")
    ap.add_argument("--min-auc", type=float, default=0.55, help="Minimum AUC to accept model (default: 0.55)")
    ap.add_argument("--limit", type=int, default=50_000, help="Max candles for regime model (default: 50000)")
    ap.add_argument("--skip-wfv", action="store_true", help="Skip walk-forward validation")
    args = ap.parse_args()

    symbols_csv = ",".join(SYMBOLS)
    failures: list[str] = []

    # ── Step 1: Train direction models (v2) ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1/3 — Training direction models (v2)")
    logger.info("=" * 60)
    for sym in SYMBOLS:
        rc = _run([
            sys.executable, "scripts/retrain_model.py",
            "--symbols", sym,
            "--days", str(args.days),
            "--min-auc", str(args.min_auc),
        ])
        if rc != 0:
            failures.append(f"direction:{sym}")

    # ── Step 2: Train regime models ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/3 — Training regime models")
    logger.info("=" * 60)
    rc = _run([
        sys.executable, "scripts/train_regime_models.py",
        "--symbols", symbols_csv,
        "--limit", str(args.limit),
    ])
    if rc != 0:
        failures.append("regime:all")

    # ── Step 3: Walk-forward validation ───────────────────────────────────────
    if not args.skip_wfv:
        logger.info("=" * 60)
        logger.info("STEP 3/3 — Walk-forward validation (5 folds)")
        logger.info("=" * 60)
        for sym in SYMBOLS:
            rc = _run([
                sys.executable, "scripts/walk_forward_validate.py",
                "--symbol", sym,
                "--days", str(args.days),
            ])
            if rc != 0:
                failures.append(f"wfv:{sym}")
    else:
        logger.info("STEP 3/3 — Walk-forward validation SKIPPED (--skip-wfv)")

    # ── Check which regime models are now reliable (AUC >= 0.80) ─────────────
    logger.info("=" * 60)
    logger.info("REGIME MODEL QUALITY SUMMARY")
    logger.info("=" * 60)
    logger.info("%-20s | %-6s | %-12s", "Symbol", "AUC", "skip_regime?")
    for sym in SYMBOLS:
        meta_path = _ROOT / "models" / f"{sym.replace('/', '_')}_regime.meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                auc = meta.get("metrics", {}).get("auc", 0.0)
                should_skip = auc < 0.80
                action = "keep skip_regime=True" if should_skip else "✅ set skip_regime=False"
                logger.info("%-20s | %.4f | %s", sym, auc, action)
            except Exception:
                logger.info("%-20s | N/A   | meta unreadable", sym)
        else:
            logger.info("%-20s | N/A   | no model trained", sym)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    if failures:
        logger.warning("FAILURES: %s", failures)
        logger.warning("Re-run failed steps manually. Bot will continue with existing models.")
    else:
        logger.info("All training steps completed successfully.")
    logger.info("")
    logger.info("Next step: restart the bot to load new models.")
    logger.info("If regime AUC >= 0.80 for XRP/SOL/DOGE: remove skip_regime from SYMBOL_CONFIG.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
