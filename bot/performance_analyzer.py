"""Performance analysis for ClawdBot trade history."""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from database.db_manager import db

logger = logging.getLogger("clawdbot.performance")

class PerformanceAnalyzer:
    def __init__(self):
        self._cached_metrics: dict[str, Any] = {}
        self._last_refresh: datetime | None = None
        self._refresh_interval = timedelta(minutes=5)

    async def get_metrics(self, force_refresh: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not force_refresh and self._last_refresh and (now - self._last_refresh) < self._refresh_interval:
            return self._cached_metrics

        metrics = await self._calculate_metrics()
        self._cached_metrics = metrics
        self._last_refresh = now
        return metrics

    async def _calculate_metrics(self) -> dict[str, Any]:
        # Simple aggregation logic using DatabaseManager methods where possible
        # or custom queries if needed.
        summary = await db.fetch_period_summary("week") # reuse existing for weekly
        total_summary = await db.fetch_period_summary("month") # reuse for 30d/total
        
        # Add more deep analytics if trade_history table is available
        # This is a stub that will be expanded as we add more specific queries to DatabaseManager
        
        return {
            "weekly_pnl": summary.get("pnl_total", 0.0),
            "weekly_pf": summary.get("profit_factor", 0.0),
            "weekly_wr": summary.get("winrate", 0.0),
            "total_trades": total_summary.get("total_trades", 0),
            "total_pnl": total_summary.get("pnl_total", 0.0),
            "total_pf": total_summary.get("profit_factor", 0.0),
        }

analyzer = PerformanceAnalyzer()
