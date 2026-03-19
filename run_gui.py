"""
run_gui.py
~~~~~~~~~~

Root-level entry point for the ClawdBot institutional desktop dashboard.

Usage
-----
    python run_gui.py

The GUI runs independently from the trading engine (main.py) and reads
state directly from TimescaleDB, so it never blocks the trading event loop.
"""

from __future__ import annotations

from dotenv import load_dotenv

# Load .env so DB_* variables are available before the GUI starts
load_dotenv()

from gui.main_window import launch  # noqa: E402  (import after load_dotenv)

if __name__ == "__main__":
    launch()
