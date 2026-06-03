"""database — persistence layer with backend switch.

Set ``DB_BACKEND=sqlite`` in env (default ``postgres``) to use the
zero-dep SQLite/aiosqlite manager instead of TimescaleDB. SQLite is
ideal for low-resource VPS without Docker.

The two backends expose the same public API so existing callers don't
change.
"""

from __future__ import annotations

import os

DB_BACKEND = os.environ.get("DB_BACKEND", "postgres").strip().lower()
