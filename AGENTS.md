# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ClawdBot is a Python 3.12 algorithmic crypto-trading bot (asyncio, XGBoost, TimescaleDB). Single-process architecture with multiple async tasks.

### Platform constraints

MetaTrader5 and `PyQt6`/`pyqtgraph` are Windows-only. On Linux, exclude them from install (CI does the same). The full trading loop cannot run on Linux because the MT5 terminal is required; only tests and startup/shutdown paths work.

### Infrastructure

- **TimescaleDB** (required): `docker compose up -d db` (PostgreSQL 15 + TimescaleDB on port 5432). Defaults are in `.env.example`.
- **Redis** (optional): defined in `docker-compose.yml` but not required by the default `main.py` loop.
- Docker daemon must be started first if not already running. In Cloud Agent VMs, use `fuse-overlayfs` storage driver and `iptables-legacy`.

### Setup (local or CI)

1. Copy `.env.example` to `.env` and set `EXECUTION_MODE=paper` for Linux.
2. Start TimescaleDB: `docker compose up -d db`
3. Install dependencies (Linux/CI should filter Windows-only packages):

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python - <<'PY'
from pathlib import Path
import re

out = []
for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if re.search(r"MetaTrader5|PyQt6|pyqtgraph", s):
        continue
    out.append(line)

Path("requirements_ci.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
PY
pip install -r requirements_ci.txt
```

### Running tests

```
python -m pytest tests/ -q --tb=short
```

### Running the app

```
python main.py
```

On Linux, the bot will start, connect to DB, create hypertables, load the XGBoost model, render the Rich TUI dashboard, then exit because the MT5 market feed is unavailable. This is expected behavior; the full trading loop requires a Windows MT5 terminal.

### Web dashboard

Embedded aiohttp server on port 8080 (Tailwind CSS SPA). Only runs when the full bot loop is active (requires MT5 feed).

### Optional external services

- `GEMINI_API_KEY`: enables LLM sentiment analysis. Bot runs with neutral sentiment (0.0) without it.
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`: enables trade alert notifications.
