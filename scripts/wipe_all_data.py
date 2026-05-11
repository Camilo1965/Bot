#!/usr/bin/env python3
"""
scripts/wipe_all_data.py
~~~~~~~~~~~~~~~~~~~~~~~~

Factory-reset script for ClawdBot.

Wipes:
  • PostgreSQL/TimescaleDB tables (market_data, trades_history, and all
    bot-related telemetry tables)
  • Local model files (models/*.json, models/*.meta.json)
  • Trade journal (trade_journal.csv)
  • Runtime metrics (runtime_metrics.jsonl)
  • Log directory contents (logs/*)
  • Local state.json

Usage:
    python scripts/wipe_all_data.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import asyncpg

logger = logging.getLogger("clawdbot.wipe")

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def _load_dotenv() -> None:
    """Inject .env keys into os.environ when present."""
    if not ENV_PATH.exists():
        return
    with ENV_PATH.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _db_dsn() -> str:
    """Build DSN from environment / .env variables."""
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "postgres")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "postgres")
    return f"postgres://{user}:{password}@{host}:{port}/{dbname}"


# ---------------------------------------------------------------------------
# Tables we want to wipe (only those that actually exist are truncated)
# ---------------------------------------------------------------------------

_TARGET_TABLES = frozenset({
    "market_data",
    "trades_history",
    "trades",          # legacy alias, if present
    "positions",       # legacy alias, if present
    "ml_predictions",
    "htf_trend",
    "htf_trend_status",
    "news_sentiment",
    "commands",
    "health_events",
})


async def _fetch_existing_tables(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    return [str(r["table_name"]) for r in rows]


async def wipe_database(dsn: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        existing = set(await _fetch_existing_tables(conn))
        to_truncate = sorted(existing & _TARGET_TABLES)

        if not to_truncate:
            logger.info("No target tables found in database; nothing to truncate.")
            return

        logger.info("Tables to truncate: %s", ", ".join(to_truncate))

        # Build a single TRUNCATE statement with RESTART IDENTITY + CASCADE
        tables_sql = ", ".join(to_truncate)
        sql = f"TRUNCATE TABLE {tables_sql} RESTART IDENTITY CASCADE;"
        await conn.execute(sql)
        logger.info("Truncated %d table(s) successfully.", len(to_truncate))
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Local filesystem cleanup
# ---------------------------------------------------------------------------

def wipe_local_files() -> None:
    models_dir = ROOT / "models"
    logs_dir = ROOT / "logs"

    # 1. Delete JSON models
    if models_dir.is_dir():
        for path in models_dir.iterdir():
            if path.is_file() and path.suffix in (".json",) and path.name != ".gitkeep":
                # Also catches .meta.json because suffix is .json
                path.unlink()
                logger.info("Deleted model file: %s", path.name)

    # 2. Delete trade journal (root or logs/)
    for rel in ("trade_journal.csv", "logs/trade_journal.csv"):
        p = ROOT / rel
        if p.is_file():
            p.unlink()
            logger.info("Deleted %s", rel)

    # 3. Empty logs/ directory (keep the directory itself)
    if logs_dir.is_dir():
        for path in logs_dir.iterdir():
            try:
                if path.is_file():
                    path.unlink()
                    logger.info("Deleted log file: %s", path.name)
                elif path.is_dir():
                    # Recursively delete subdirectories in logs
                    import shutil
                    shutil.rmtree(path)
                    logger.info("Deleted log directory: %s", path.name)
            except OSError as exc:
                logger.warning("Could not remove %s: %s", path, exc)

    # 4. Delete runtime_metrics.jsonl (root or logs/)
    for rel in ("runtime_metrics.jsonl", "logs/runtime_metrics.jsonl"):
        p = ROOT / rel
        if p.is_file():
            p.unlink()
            logger.info("Deleted %s", rel)

    # 5. Delete local state.json (root or logs/)
    for rel in ("state.json", "logs/state.json"):
        p = ROOT / rel
        if p.is_file():
            p.unlink()
            logger.info("Deleted %s", rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main() -> int:
    parser = argparse.ArgumentParser(description="ClawdBot factory reset")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _load_dotenv()
    dsn = _db_dsn()

    print("=" * 60)
    print("  CLAWDBOT FACTORY RESET")
    print("=" * 60)
    print()
    print("This will IRREVERSIBLY delete:")
    print("  • All bot data in the PostgreSQL database")
    print("  • All trained model files (models/*.json)")
    print("  • Trade journal, runtime metrics, and local state")
    print("  • All contents of the logs/ directory")
    print()
    print(f"Database DSN: {dsn}")
    print()

    if not args.yes:
        try:
            answer = input("Are you sure you want to proceed? [y/N]: ").strip().lower()
        except EOFError:
            print("Aborted (non-interactive). Use --yes to force.")
            return 1
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    print()
    logger.info("Starting wipe...")

    # Database
    try:
        await wipe_database(dsn)
    except Exception as exc:
        logger.error("Database wipe failed: %s", exc)
        return 1

    # Local files
    try:
        wipe_local_files()
    except Exception as exc:
        logger.error("Local file wipe failed: %s", exc)
        return 1

    logger.info("Wipe complete. System is clean.")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
