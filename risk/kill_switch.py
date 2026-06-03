"""
risk.kill_switch
~~~~~~~~~~~~~~~~

File-flag global halt: place a ``KILL`` file in the repo root (or set env
``KILL_SWITCH=1``) to block all new entries without stopping the process.
Existing positions continue to be managed by SL / smart-exit / TTL.

Usage::

    # Halt
    echo "manual halt" > KILL
    # or: KILL_SWITCH=1

    # Resume
    rm KILL
"""

from __future__ import annotations

import os
from pathlib import Path

_KILL_FILE = Path(__file__).resolve().parent.parent / "KILL"
_HALT_REASON: str | None = None


def is_halted() -> bool:
    """Return True when the kill switch is active (file or env var)."""
    if os.environ.get("KILL_SWITCH", "").strip() in ("1", "true", "yes"):
        return True
    if _KILL_FILE.is_file():
        return True
    return False


def reason() -> str | None:
    """Return the halt reason string, or None."""
    if os.environ.get("KILL_SWITCH", "").strip() in ("1", "true", "yes"):
        return os.environ.get("KILL_SWITCH_REASON") or "KILL_SWITCH env"
    if _KILL_FILE.is_file():
        try:
            text = _KILL_FILE.read_text(encoding="utf-8").strip()
            return text or "KILL file present"
        except Exception:
            return "KILL file present"
    return None


def halt(reason_str: str = "manual") -> None:
    """Activate the kill switch by writing the KILL file."""
    _KILL_FILE.write_text(reason_str, encoding="utf-8")


def resume() -> None:
    """Deactivate the kill switch by removing the KILL file."""
    if _KILL_FILE.is_file():
        _KILL_FILE.unlink()
