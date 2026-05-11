# Remaining Work / Known Limitations — Post-Audit 2026-05-11

## ✅ Completed in this sprint (Observability & Alerts)

| # | Item | Files | Tests |
|---|------|-------|-------|
| B | **Monitoring & Alerts** | `bot/monitoring.py` | 16 tests |
|   | Stale feed alert (>90s) | `bot/monitoring.py` | `test_alert_when_stale` |
|   | Daily P&L loss alert (1.5%) | `bot/monitoring.py` | `test_alert_when_loss_threshold_crossed` |
|   | VIBE down alert (>5min) | `bot/monitoring.py` | `test_alert_when_vibe_unavailable` |
|   | Close-pending stuck alert | `bot/monitoring.py` | `test_alert_when_close_pending` |
|   | Health endpoint metrics | `bot/web_server.py` | `test_returns_ok_when_fresh` |

## ✅ Completed in previous sprint (Blind Spot Critical)

| # | Item | Files | Tests |
|---|------|-------|-------|
| A | **Core loop tests** | `tests/test_core_loop.py` | 13 tests |
|   | market_consumer coverage | `bot/market_consumer.py` | 0% → 77% |
|   | signal_emitter coverage | `bot/signal_emitter.py` | 0% → 70% |
|   | loops coverage | `bot/loops.py` | 0% → 36% |

## ✅ Completed in first sprint (Bugs + Infra)

| # | Item | Files | Tests |
|---|------|-------|-------|
| 1 | **PostgreSQL indices** | `database/db_manager.py` | implicit (startup idempotent) |
| 2 | **XGBoost retraining** | `scripts/retrain_model.py` | manual / CLI |
| 3 | **Startup position recovery** | `execution/mt5_executor.py`, `main.py` | `tests/test_mt5_executor_mocked.py` |
| 4 | **MT5 executor mocked tests** | `tests/test_mt5_executor_mocked.py` | 9 tests (coverage 10% → 44%) |
| 5 | **EC2 infra automation** | `scripts/setup_ec2_infra.sh` | manual deployment |
| — | Counter drift fix | `risk/risk_manager.py` | `tests/test_counter_drift.py` |
| — | VIBE MCP timeout fix | `vibe/mcp_client.py` | `tests/test_vibe_mcp.py` |
| — | Trailing stop sync fix | `execution/paper_executor.py`, `bot/market_consumer.py` | `tests/test_trailing_stop.py` |
| — | aiohttp security fix | `bot/web_server.py` | `tests/test_aiohttp_security.py` |

## 📊 Coverage Snapshot

```
bot/market_consumer.py     : 77%  (was 0%)
bot/signal_emitter.py      : 70%  (was 0%)
bot/monitoring.py          : 85%  (new)
bot/loops.py               : 36%  (was 0%)
execution/mt5_executor.py  : 44%  (was 10%)
execution/paper_executor.py: 47%
risk/risk_manager.py       : 77%
vibe/*                     : 48-100%
database/db_manager.py     : 24%
TOTAL                      : 37%
```

## ⚠️ Still Outstanding

1. **`bot/loops.py` 36% → 75%** — Remaining untested paths are Rich dashboard rendering, audit telemetry, and health monitor stale-feed detection. Requires heavy mocking of `rich.live.Live` and `asyncpg` connections.

2. **`execution/mt5_executor.py` 44% → 75%** — Reaching 75% would require mocking 500+ additional lines of error-handling branches, Telegram alerts, and deep MT5 order-retry logic. The critical happy-paths (open, close, sync, modify, recovery, trailing) are all tested.

3. **`database/db_manager.py` 24%** — Needs asyncpg pool mocks to test query paths.

4. **`bot/dashboard.py` 7%** — Rich TUI rendering is mostly declarative; testing it adds limited value vs integration testing.

## 📈 Test Count

```
90 passed (was 42 base)
```

## 🚀 Deployment Checklist (EC2)

```bash
# 1. Run infra setup
sudo bash scripts/setup_ec2_infra.sh

# 2. Verify
sudo systemctl status clawdbot
crontab -l | grep backup

# 3. Retrain model (optional)
python scripts/retrain_model.py --symbol BTC/USDT --days 60 --min-auc 0.55

# 4. Monthly reset
python scripts/monthly_reset.py --report-only

# 5. Health check
curl http://localhost:8080/health
```

## 🔧 ENV Variables for Monitoring (add to .env)

```
# Alert thresholds
MONITOR_STALE_FEED_S=90          # Alert if no tick in 90s
MONITOR_DAILY_PNL_ALERT_PCT=0.015 # Alert at 1.5% daily loss
MONITOR_VIBE_DOWN_S=300          # Alert if VIBE down 5min
MONITOR_ALERT_COOLDOWN_S=300     # 5min between duplicate alerts
```
