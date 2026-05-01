#!/usr/bin/env python3
"""
Genera un único Markdown para revisión (mismo contenido que el bot escribe en la raíz).

Uso::

    .venv312\\Scripts\\python scripts\\export_diagnostic_bundle.py

Salida por defecto: ``logs/diagnostic_bundle_FOR_REVIEW.md``
(también existe ``DIAGNOSTIC_FOR_REVIEW.md`` en la raíz si el bot corre con intervalo > 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Allow running without full package path
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.diagnostic_bundle import write_diagnostic_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Export diagnostic Markdown bundle.")
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO / "logs" / "diagnostic_bundle_FOR_REVIEW.md",
        help="Ruta del .md",
    )
    args = ap.parse_args()
    try:
        out = write_diagnostic_bundle(repo_root=_REPO, output=args.output, load_dotenv_file=True)
        print(f"OK -> {out.resolve()} ({out.stat().st_size // 1024} KB approx)")
    except OSError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
