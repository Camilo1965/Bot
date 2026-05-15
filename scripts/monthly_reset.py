"""Monthly reset script for ClawdBot.

Usage:
    python scripts/monthly_reset.py --report-only
    python scripts/monthly_reset.py --dry-run
    python scripts/monthly_reset.py
    python scripts/monthly_reset.py --month 2026-05
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("clawdbot.monthly_reset")

_ARCHIVE_DIR = Path("archives")
_LOGS_DIR = Path("logs")
_DB_NAME = os.environ.get("DB_NAME", "clawdbot")
_DB_USER = os.environ.get("DB_USER", "clawdbot")
_DB_HOST = os.environ.get("DB_HOST", "localhost")
_DB_PORT = os.environ.get("DB_PORT", "5432")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _month_str(month: str | None) -> str:
    if month:
        return month
    return _now().strftime("%Y-%m")


def can_reset_safely() -> bool:
    """Abort if there are open positions in MT5 or local state."""
    state_file = _LOGS_DIR / "state.json"
    if state_file.exists():
        try:
            with state_file.open() as f:
                state = json.load(f)
            if state.get("positions"):
                logger.error("Reset blocked: local positions exist in state.json.")
                return False
        except Exception:
            pass
    # If MT5 is available, check there too
    try:
        import MetaTrader5 as mt5
        if mt5.positions_total() > 0:
            logger.error("Reset blocked: MT5 has open positions.")
            return False
    except Exception:
        pass
    return True


def calculate_stats(month: str) -> dict[str, Any]:
    """Calculate P&L, winrate, breakdown from trade journal."""
    journal = _LOGS_DIR / "trade_journal.csv"
    stats: dict[str, Any] = {
        "month": month,
        "total_pnl": 0.0,
        "winrate": 0.0,
        "wins": 0,
        "losses": 0,
        "total": 0,
        "by_symbol": {},
        "best_trade": None,
        "worst_trade": None,
    }
    if not journal.exists():
        return stats

    import csv
    rows: list[dict[str, Any]] = []
    with journal.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                et = datetime.fromisoformat(row["entry_time"])
                if et.strftime("%Y-%m") == month:
                    rows.append(row)
            except Exception:
                continue

    pnls: list[float] = []
    for row in rows:
        try:
            pnl = float(row.get("net_pnl", 0))
        except Exception:
            pnl = 0.0
        pnls.append(pnl)
        sym = row.get("symbol", "UNKNOWN")
        stats["by_symbol"].setdefault(sym, {"pnl": 0.0, "count": 0})
        stats["by_symbol"][sym]["pnl"] += pnl
        stats["by_symbol"][sym]["count"] += 1

    stats["total_pnl"] = sum(pnls)
    stats["wins"] = sum(1 for p in pnls if p > 0)
    stats["losses"] = sum(1 for p in pnls if p < 0)
    stats["total"] = len(pnls)
    stats["winrate"] = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
    if pnls:
        stats["best_trade"] = max(pnls)
        stats["worst_trade"] = min(pnls)
    return stats


def backup_postgres(month: str, dry_run: bool = False) -> Path | None:
    """pg_dump → archives/YYYY-MM/backup_YYYY-MM-DD_HHMM.sql.gz"""
    out_dir = _ARCHIVE_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"backup_{ts}.sql.gz"
    if dry_run:
        logger.info("[DRY-RUN] Would backup Postgres → %s", out_path)
        return out_path
    try:
        with out_path.open("wb") as f:
            proc = subprocess.run(
                ["pg_dump", "-h", _DB_HOST, "-p", _DB_PORT, "-U", _DB_USER, _DB_NAME],
                stdout=subprocess.PIPE,
                check=True,
                env={**os.environ, "PGPASSWORD": os.environ.get("DB_PASSWORD", "")},
            )
            f.write(gzip.compress(proc.stdout))
        logger.info("Postgres backup created → %s", out_path)
        return out_path
    except Exception as exc:
        logger.error("Postgres backup failed: %s", exc)
        return None


def archive_files(month: str, dry_run: bool = False) -> None:
    """Move logs and state into archives/YYYY-MM/."""
    out_dir = _ARCHIVE_DIR / month / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in _LOGS_DIR.iterdir():
        if src.name in ("state.json", "trade_journal.csv", "runtime_metrics.jsonl"):
            dst = out_dir / src.name
            if dry_run:
                logger.info("[DRY-RUN] Would move %s → %s", src, dst)
            else:
                shutil.copy2(src, dst)
                logger.info("Archived %s → %s", src, dst)


def generate_html_report(stats: dict[str, Any], month: str, out_path: Path) -> None:
    """Generate a simple HTML report."""
    best_str = f"{stats['best_trade']:.2f}" if stats['best_trade'] is not None else "N/A"
    worst_str = f"{stats['worst_trade']:.2f}" if stats['worst_trade'] is not None else "N/A"
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ClawdBot Report {month}</title></head>
<body>
<h1>ClawdBot Monthly Report - {month}</h1>
<p><strong>Total P&L:</strong> {stats['total_pnl']:.2f} USDT</p>
<p><strong>Winrate:</strong> {stats['winrate']:.1f}% ({stats['wins']}W / {stats['losses']}L / {stats['total']}T)</p>
<p><strong>Best trade:</strong> {best_str}</p>
<p><strong>Worst trade:</strong> {worst_str}</p>
<h2>Breakdown by Symbol</h2>
<table border="1"><tr><th>Symbol</th><th>Trades</th><th>PnL</th></tr>
"""
    for sym, data in stats.get("by_symbol", {}).items():
        html += f"<tr><td>{sym}</td><td>{data['count']}</td><td>{data['pnl']:.2f}</td></tr>\n"
    html += "</table></body></html>"
    out_path.write_text(html, encoding="utf-8")
    logger.info("HTML report generated → %s", out_path)


def reset_postgres(dry_run: bool = False) -> None:
    """TRUNCATE trades and positions tables (not DROP)."""
    if dry_run:
        logger.info("[DRY-RUN] Would TRUNCATE trades, positions tables.")
        return
    try:
        subprocess.run(
            [
                "psql",
                "-h", _DB_HOST,
                "-p", _DB_PORT,
                "-U", _DB_USER,
                "-d", _DB_NAME,
                "-c", "TRUNCATE TABLE trades, positions RESTART IDENTITY;",
            ],
            check=True,
            env={**os.environ, "PGPASSWORD": os.environ.get("DB_PASSWORD", "")},
        )
        logger.info("Postgres tables truncated.")
    except Exception as exc:
        logger.error("Postgres truncate failed: %s", exc)


def reset_local_files(dry_run: bool = False) -> None:
    """Clear state.json and trade_journal.csv."""
    for fname in ("state.json", "trade_journal.csv"):
        path = _LOGS_DIR / fname
        if path.exists():
            if dry_run:
                logger.info("[DRY-RUN] Would clear %s", path)
            else:
                path.write_text("")
                logger.info("Cleared %s", path)


def reset_vibe_state(dry_run: bool = False) -> None:
    """Remove vibe_state.json."""
    path = _LOGS_DIR / "vibe_state.json"
    if path.exists():
        if dry_run:
            logger.info("[DRY-RUN] Would remove %s", path)
        else:
            path.unlink()
            logger.info("Removed %s", path)


def notify_telegram(stats: dict[str, Any]) -> None:
    """Send summary via Telegram if configured."""
    from utils.telegram_notifier import send_telegram_alert
    msg = (
        f"📊 *Monthly Reset* - {stats['month']}\n"
        f"PnL: `{stats['total_pnl']:+.2f}` USDT\n"
        f"Winrate: `{stats['winrate']:.1f}%` ({stats['wins']}W/{stats['losses']}L)\n"
    )
    try:
        asyncio.run(send_telegram_alert(msg))
    except Exception as exc:
        logger.warning("Telegram notify failed: %s", exc)


def verify_clean_state() -> bool:
    """Confirm state.json is empty and no positions remain."""
    state_file = _LOGS_DIR / "state.json"
    if state_file.exists() and state_file.stat().st_size > 0:
        try:
            with state_file.open() as f:
                data = json.load(f)
            if data.get("positions"):
                logger.error("Verify failed: positions still in state.json")
                return False
        except Exception:
            pass
    logger.info("Clean-state verification passed.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="ClawdBot monthly reset")
    parser.add_argument("--report-only", action="store_true", help="Only generate report")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without changing data")
    parser.add_argument("--month", default=None, help="YYYY-MM format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    month = _month_str(args.month)
    stats = calculate_stats(month)
    report_path = _ARCHIVE_DIR / month / f"report_{month}.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(stats, month, report_path)

    if args.report_only:
        logger.info("Report-only mode. Exiting.")
        return 0

    if not can_reset_safely():
        logger.error("Reset aborted due to open positions.")
        return 1

    backup = backup_postgres(month, dry_run=args.dry_run)
    if not backup and not args.dry_run:
        logger.error("Backup failed - aborting reset.")
        return 1

    archive_files(month, dry_run=args.dry_run)
    reset_postgres(dry_run=args.dry_run)
    reset_local_files(dry_run=args.dry_run)
    reset_vibe_state(dry_run=args.dry_run)
    notify_telegram(stats)

    if not args.dry_run:
        if not verify_clean_state():
            return 1

    logger.info("Monthly reset complete for %s.", month)
    return 0


if __name__ == "__main__":
    sys.exit(main())
