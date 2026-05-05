"""Load ``.env`` without python-dotenv UTF-16 parse failures (Windows Notepad)."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None  # type: ignore[assignment]


def load_env_file(path: Path, *, override: bool = False) -> None:
    """Parse ``path`` and apply to ``os.environ`` (default: do not override existing)."""
    if dotenv_values is None or not path.is_file():
        return
    raw = path.read_bytes()
    text: str | None = None
    for enc in ("utf-8-sig", "utf-16-le", "utf-16-be", "utf-8", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return
    vals = dotenv_values(stream=StringIO(text))
    for key, val in vals.items():
        if val is None or not key:
            continue
        if override or os.getenv(key) is None:
            os.environ[key] = val
