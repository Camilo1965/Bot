"""Shared literals used by ``main`` and the extracted ``bot.*`` runtime modules."""

# Hint appended to warning messages that have more detail in the debug log.
DEBUG_LOG_HINT = "(Revisa logs/last_session.log o bot_debug.log para detalles)"

# ── [PRO] News Filter parameters ─────────────────────────────────────────────
NEWS_FILTER_VOLATILITY_THRESHOLD: float = 0.4
NEWS_FILTER_HOLD_MINUTES: int = 30
