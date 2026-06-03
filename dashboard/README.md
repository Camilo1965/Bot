# ClawdBot Dashboard

## Stack
- **API**: FastAPI + Uvicorn + Redis + DuckDB (Python)
- **Frontend**: Next.js 15 + React 19 + Tailwind CSS v4 + Recharts

## Quick Start

### 1. Start infrastructure
```powershell
docker compose up -d   # TimescaleDB + Redis
```

### 2. Start API
```powershell
cd C:\Users\WinterOS\Desktop\projectos\Bot
pip install -r dashboard/api/requirements.txt
uvicorn dashboard.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start frontend
```powershell
cd dashboard/web
npm install
npm run dev   # http://localhost:3000
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_API_KEY` | `dev-secret` | API auth header `X-Api-Key` |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API base URL for frontend |

## Pages

| Route | Description |
|---|---|
| `/` | Overview — equity, open positions, recent signals |
| `/positions` | Open/Closed/All positions with PnL |
| `/signals` | Live signal stream per symbol |
| `/performance` | KPIs, equity curve, heatmaps |
| `/risk` | Kill switch, drawdown gauges, exposure |
| `/models` | AUC trends, calibration, drift test |
| `/backtest` | Run and compare backtests |
| `/journal` | Trade history, VIBE insights |
| `/settings` | Symbol config, risk params, notifications |
| `/alerts` | Kill switch + drift event inbox |

## API Endpoints

Base URL: `http://localhost:8000`

```
GET  /api/state
GET  /api/equity?days=30
GET  /api/positions/open
GET  /api/positions/history?limit=100&symbol=
GET  /api/signals/recent?limit=50
GET  /api/signals/by-symbol/{sym}?limit=200
GET  /api/performance?range=30d&symbol=all
GET  /api/risk/state
POST /api/risk/killswitch/trigger   body: {reason, duration_hours}
POST /api/risk/killswitch/reset
GET  /api/models
GET  /api/alerts?status=unread
WS   /ws/stream   (equity.tick, position.*, signal.evaluated events)
```

## Bot Integration

Add to `bot/main.py` after each tick:

```python
from dashboard.api.bot_writer import write_state, write_position_opened, write_position_closed
# call write_state(balance, equity, daily_pnl_pct, open_positions) each loop
```
