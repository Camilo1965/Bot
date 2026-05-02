# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ClawdBot is a Python 3.12 algorithmic crypto-trading bot (asyncio, XGBoost, TimescaleDB). Single-process architecture with multiple async tasks.

### Platform constraint

MetaTrader5 Python package and `PyQt6`/`pyqtgraph` are **Windows-only**. On Linux, exclude them from install (CI does the same). The full trading loop cannot run on Linux — only tests and startup/shutdown paths work.

### Infrastructure

- **TimescaleDB** (required): `docker compose up -d db` — PostgreSQL 15 + TimescaleDB on port 5432. Credentials default to `clawdbot`/`clawdbot_secret`.
- **Redis** (optional): defined in `docker-compose.yml` but not used by default `main.py` loop.
- Docker daemon must be started first if not already running. In Cloud Agent VMs, use `fuse-overlayfs` storage driver and `iptables-legacy`.

### Running tests

```
python3 -m pytest tests/ -q --tb=short --ignore=tests/manual_order_test.py
```

`tests/manual_order_test.py` imports `MetaTrader5` (Windows-only) — always exclude on Linux. One pre-existing test failure (`test_generate_signal_buy_muted_when_1h_not_bullish`) exists in the repo.

### Running the app

1. Copy `.env.example` to `.env` and set `EXECUTION_MODE=paper` for Linux.
2. Start TimescaleDB: `docker compose up -d db`
3. Run: `python3 main.py`

The bot will start, connect to DB, create hypertables, load the XGBoost model, render the Rich TUI dashboard, then exit because the MT5 market feed is unavailable on Linux. This is expected behavior — the full trading loop requires a Windows MT5 terminal.

### Dependency install (Linux/CI)

Filter `MetaTrader5`, `PyQt6`, `pyqtgraph` from `requirements.txt` before pip install — same approach as `.github/workflows/ci.yml`.

### Web dashboard

Embedded aiohttp server on port 8080 (Tailwind CSS SPA). Only runs when the full bot loop is active (requires MT5 feed).

### Optional external services

- `GEMINI_API_KEY`: enables LLM sentiment analysis. Bot runs with neutral sentiment (0.0) without it.
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`: enables trade alert notifications.
