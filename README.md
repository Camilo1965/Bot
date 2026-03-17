# ClawdBot 🐾

> **Institutional-grade algorithmic trading system built on an Event-Driven Architecture (EDA)**
> Python 3.10+ · asyncio · PostgreSQL / TimescaleDB · Redis

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Module Reference](#module-reference)
6. [Configuration](#configuration)
7. [Roadmap](#roadmap)

---

## Architecture Overview

ClawdBot is built around an **Event-Driven Architecture (EDA)**. Every significant
action in the system – a new market tick, a generated trading signal, an order
fill, a risk-limit breach – is represented as an *event* that flows through a
central **Event Bus**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ClawdBot Process                            │
│                                                                      │
│  ┌─────────────┐   events    ┌──────────────┐   events              │
│  │ data_ingestion│──────────▶│  Event Bus   │──────────┐            │
│  │  (WebSocket /│            │  (asyncio    │          │            │
│  │   REST feed) │            │   queues)    │          ▼            │
│  └─────────────┘            └──────────────┘   ┌─────────────┐     │
│                                    ▲            │  strategy   │     │
│  ┌─────────────┐   orders          │            │  (ML models │     │
│  │  execution  │◀──────────────────┘            │  & signals) │     │
│  │  (OMS /     │                                └─────────────┘     │
│  │   exchange) │                                       │            │
│  └─────────────┘                                       │ signals    │
│         │                                              ▼            │
│         │               ┌─────────────┐        ┌─────────────┐     │
│         └──────────────▶│    risk     │        │    utils    │     │
│           fills         │  (sizing,  │        │  (logging,  │     │
│                         │   limits)  │        │   config)   │     │
│                         └─────────────┘        └─────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
  ┌─────────────┐                        ┌─────────────┐
  │ TimescaleDB │                        │    Redis    │
  │ (tick / bar │                        │  (cache /   │
  │  history)   │                        │   pub-sub)  │
  └─────────────┘                        └─────────────┘
```

### Key Design Principles

| Principle | Implementation |
|---|---|
| **Loose coupling** | Modules communicate only via events; no direct imports across domain boundaries |
| **Non-blocking I/O** | Every network call uses `asyncio` coroutines (`asyncpg`, `redis-py async`, `ccxt.async_support`) |
| **Time-series storage** | TimescaleDB hypertables give sub-second query performance on millions of OHLCV rows |
| **Low-latency cache** | Redis holds the latest order book snapshots and computed feature vectors |
| **Reproducible ML** | Feature engineering pipelines are version-controlled inside `/strategy` |

---

## Project Structure

```
Bot/
├── core/                  # Event bus and abstract base classes
│   └── __init__.py
├── data_ingestion/        # Async WebSocket feeds & historical REST loaders
│   └── __init__.py
├── strategy/              # ML models, signal generation, feature engineering
│   └── __init__.py
├── execution/             # Order Management System (OMS)
│   └── __init__.py
├── risk/                  # Position sizing and risk management
│   └── __init__.py
├── utils/                 # Structured logging and configuration helpers
│   └── __init__.py
├── main.py                # Application entry point (asyncio event loop)
├── docker-compose.yml     # TimescaleDB + Redis services
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md
```

---

## Prerequisites

- **Docker** ≥ 24 and **Docker Compose** ≥ 2
- **Python** 3.10 or later
- (Optional) a virtual-environment manager such as `venv` or `conda`

---

## Quick Start

### 1 – Start the infrastructure containers

```bash
# Copy and edit the environment file
cp .env.example .env
# (fill in DB_USER, DB_PASSWORD, exchange credentials, etc.)

# Start PostgreSQL/TimescaleDB and Redis in the background
docker compose up -d

# Verify both containers are healthy
docker compose ps
```

### 2 – Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3 – Run the application

```bash
python main.py
```

You should see structured JSON log output similar to:

```json
{"timestamp": "2025-01-01T00:00:00+00:00", "level": "INFO", "logger": "clawdbot", "message": "ClawdBot starting up"}
{"timestamp": "2025-01-01T00:00:00+00:00", "level": "INFO", "logger": "clawdbot", "message": "ClawdBot shut down cleanly"}
```

### 4 – Tear down containers

```bash
docker compose down          # keep volumes
docker compose down -v       # also remove volumes (destroys data)
```

---

## Module Reference

### `core/`
Contains the **Event Bus** and abstract base classes that every domain module
inherits from. The event bus exposes `publish(event)` and `subscribe(event_type,
handler)` coroutines backed by `asyncio.Queue`.

### `data_ingestion/`
Responsible for streaming real-time market data via **WebSocket** (using
`ccxt.async_support` or raw `websockets`) and loading historical OHLCV data from
exchange REST APIs or TimescaleDB.

### `strategy/`
Houses **feature engineering** pipelines (`pandas` / `polars`), **ML model**
training and inference (`xgboost`, `transformers`), and **signal generation**
logic that publishes `SignalEvent` objects onto the bus.

### `execution/`
The **Order Management System** converts signals into exchange orders via
`ccxt.async_support`, tracks open positions, and emits `FillEvent` objects on
confirmation.

### `risk/`
Subscribes to `SignalEvent` and `FillEvent` to enforce **position limits**,
**max drawdown** guards, and **Kelly-criterion-based position sizing** before
orders reach the exchange.

### `utils/`
Provides a reusable **structured JSON logger** and a `Config` class that loads
settings from the `.env` file via `python-dotenv`.

---

## Configuration

All secrets and environment-specific settings live in `.env` (never committed).
Copy `.env.example` and populate the values:

| Variable | Description |
|---|---|
| `EXCHANGE_API_KEY` | API key for the target exchange |
| `EXCHANGE_SECRET` | API secret for the target exchange |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `REDIS_URL` | Redis connection URL (e.g. `redis://localhost:6379/0`) |

---

## Roadmap

- [ ] Implement `core/event_bus.py` with async publish/subscribe
- [ ] Add WebSocket feed in `data_ingestion/`
- [ ] Build feature engineering pipeline in `strategy/`
- [ ] Implement OMS in `execution/`
- [ ] Add Kelly-criterion position sizing in `risk/`
- [ ] Wire up structured config loader in `utils/`
- [ ] Add pytest test suite
- [ ] CI/CD pipeline with GitHub Actions