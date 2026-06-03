#!/usr/bin/env bash
# scripts/install_vps_zerodep.sh
# One-shot installer for ClawdBot on a VPS without Docker.
# Uses SQLite (aiosqlite) + fakeredis (in-memory) — no daemons, no sudo.
#
# Usage:
#   bash scripts/install_vps_zerodep.sh
#
# After install, configure .env then start with:
#   DB_BACKEND=sqlite REDIS_BACKEND=memory python main.py

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "==> ClawdBot zero-dep VPS installer"
echo "    Repo: $REPO_DIR"

# 1. Verify Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "==> Python: $PYV"

# 2. Create venv if missing
if [ ! -d "venv" ]; then
    echo "==> Creating virtualenv"
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

# 3. Install Python deps
echo "==> Installing requirements"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r dashboard/api/requirements.txt

# 4. Ensure data/logs/models dirs
mkdir -p data logs models

# 5. Bootstrap .env with zero-dep defaults if missing
if [ ! -f .env ]; then
    cat > .env <<'EOF'
# Zero-dep VPS profile — no Docker, no daemons
DB_BACKEND=sqlite
SQLITE_DB_PATH=data/clawdbot.sqlite
REDIS_BACKEND=memory

EXECUTION_MODE=paper
WATCHLIST=BTC/USDT,ETH/USDT,SOL/USDT,DOGE/USDT,NEAR/USDT,LINK/USDT,JTO/USDT,INJ/USDT
MAX_POSITIONS=2
INITIAL_BALANCE=10000
EOF
    echo "==> Wrote default .env (edit before first run)"
fi

# 6. Init SQLite schema
echo "==> Initializing SQLite schema"
DB_BACKEND=sqlite python -c "
import asyncio
from database.db_manager import init_db, close_db
async def main():
    await init_db()
    await close_db()
asyncio.run(main())
"

# 7. Print next steps
cat <<EOF

==> Install complete.

Activate venv:    source venv/bin/activate
Run bot:          DB_BACKEND=sqlite REDIS_BACKEND=memory python main.py
Run dashboard:    cd dashboard/api && uvicorn main:app --host 0.0.0.0 --port 8000

To persist as service, create a systemd unit pointing at venv/bin/python main.py.
EOF
