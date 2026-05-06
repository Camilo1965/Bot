"""Load ``.env`` without python-dotenv UTF-16 parse failures (Windows Notepad)."""

from __future__ import annotations

import logging
import os
import re
from io import StringIO
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None  # type: ignore[assignment]

_ENV_KEY_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_dotenv_text(text: str) -> str:
    """Strip decorative/garbage lines (unicode rules, lone ``.``) so dotenv does not fail whole file."""
    out: list[str] = []
    for line in text.splitlines():
        st = line.strip()
        if not st:
            out.append("")
            continue
        if st.startswith("#"):
            out.append(line.rstrip("\r\n"))
            continue
        if "=" not in st:
            continue
        key = st.split("=", 1)[0].strip()
        if _ENV_KEY_OK.fullmatch(key):
            out.append(line.rstrip("\r\n"))
    return "\n".join(out)


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
    safe = _sanitize_dotenv_text(text)
    vals = dotenv_values(stream=StringIO(safe))
    for key, val in vals.items():
        if val is None or not key:
            continue
        if override or os.getenv(key) is None:
            os.environ[key] = val
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)


def patch_dotenv_load() -> None:
    """Make ``dotenv.load_dotenv()`` UTF-safe so libs that call it stop spamming parse errors."""
    try:
        import dotenv.main as dm

        _orig = dm.load_dotenv

        def _patched(
            dotenv_path: str | os.PathLike[str] | None = None,
            stream: object | None = None,
            verbose: bool = False,
            override: bool = False,
            interpolate: bool = True,
            encoding: str | None = None,
        ) -> bool:
            if stream is not None:
                return bool(
                    _orig(
                        stream=stream,
                        verbose=verbose,
                        override=override,
                        interpolate=interpolate,
                        encoding=encoding,
                    )
                )
            path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
            load_env_file(path, override=override)
            return path.is_file()

        dm.load_dotenv = _patched  # type: ignore[method-assign]
    except Exception:
        pass


patch_dotenv_load()
