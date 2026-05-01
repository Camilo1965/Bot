#!/usr/bin/env python3
"""
Export Markdown para revisión (snapshot corto o día completo).

Snapshot (rápido, colas)::

    .venv312\\Scripts\\python scripts\\export_diagnostic_bundle.py

Día completo — JSONL del día + ``bot_debug`` (INFO/DEBUG/WARNING/ERROR) + last_session/audit/journal ampliados::

    .venv312\\Scripts\\python scripts\\export_diagnostic_bundle.py --full-day

Otro día::

    .venv312\\Scripts\\python scripts\\export_diagnostic_bundle.py --full-day --date 2026-05-01

Salida snapshot default: ``logs/diagnostic_bundle_FOR_REVIEW.md``
Salida full-day default: ``DIAGNOSTIC_DAY_YYYY-MM-DD.md`` en la raíz del repo.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.diagnostic_bundle import write_diagnostic_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Export diagnostic Markdown bundle.")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta del .md (opcional; full-day usa nombre por defecto si omitís)",
    )
    ap.add_argument(
        "--full-day",
        action="store_true",
        help="JSONL del día + bot_debug (todos niveles JSON por defecto) + logs ampliados",
    )
    ap.add_argument(
        "--bot-debug-errors-only",
        action="store_true",
        help="Con --full-day: solo ERROR/WARNING en bot_debug (bundle más chico)",
    )
    ap.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Día local (REPORT_TIMEZONE). Default: hoy. Solo con --full-day.",
    )
    args = ap.parse_args()

    report_date: date | None = None
    if args.date:
        report_date = date.fromisoformat(args.date)

    mode = "full_day" if args.full_day else "snapshot"
    out_arg = args.output
    if mode == "snapshot" and out_arg is None:
        out_arg = _REPO / "logs" / "diagnostic_bundle_FOR_REVIEW.md"

    try:
        out = write_diagnostic_bundle(
            repo_root=_REPO,
            output=out_arg,
            load_dotenv_file=True,
            mode=mode,
            report_date=report_date,
            full_day_bot_debug_errors_only=args.bot_debug_errors_only,
        )
        print(f"OK -> {out.resolve()} ({out.stat().st_size // 1024} KB approx)")
    except OSError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
