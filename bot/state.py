"""Process-wide dashboard / UI state shared across asyncio tasks."""

from __future__ import annotations

from collections import deque
from datetime import datetime

# Mega-dashboard events panel (last N operational log lines as Rich markup).
dashboard_events: deque[str] = deque(maxlen=5)

# Set once at startup for uptime display.
bot_start_time: datetime | None = None
