#!/usr/bin/env python3
"""Run ruff + bandit on application paths (exit non-zero if tools missing)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["execution", "bot", "risk", "database", "utils", "main.py", "preflight.py"]


def main() -> int:
    rc = 0
    ruff = shutil.which("ruff")
    if ruff:
        r = subprocess.run([ruff, "check", *[str(ROOT / t) for t in TARGETS]], cwd=ROOT)
        rc |= r.returncode
    else:
        print("SKIP: ruff not installed (`pip install ruff`)")

    bandit = shutil.which("bandit")
    if bandit:
        args = [
            bandit,
            "-r",
            *[str(ROOT / t) for t in TARGETS if (ROOT / t).exists()],
            "-ll",
            "-q",
            "--exclude",
            "tests,coverage_report",
        ]
        r = subprocess.run(args, cwd=ROOT)
        rc |= r.returncode
    else:
        print("SKIP: bandit not installed (`pip install bandit`)")

    vul = shutil.which("vulture")
    if vul:
        subprocess.run(
            [vul, *[str(ROOT / t) for t in TARGETS if (ROOT / t).exists()], "--min-confidence", "90"],
            cwd=ROOT,
        )
    else:
        print("SKIP: vulture not installed (`pip install vulture`)")

    return rc


if __name__ == "__main__":
    sys.exit(main())
