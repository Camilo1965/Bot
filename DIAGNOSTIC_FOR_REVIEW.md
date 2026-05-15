# ClawdBot - paquete único para revisión (IA / humano)
_Generado: `2026-05-14T04:04:41Z` - repo: `C:\Users\WinterOS\Desktop\projectos\Bot`_
## Cómo usar
Adjunta o pega **este archivo completo** al asistente.

**Fin de día (todo el día):** `python scripts/export_diagnostic_bundle.py --full-day`

## 1) Resumen rápido
- Líneas en `runtime_metrics.jsonl`: **2**
- Primera muestra (último archivo): `2026-05-14T01:28:57.922180+00:00`
- Última muestra: `2026-05-14T01:32:56.069417+00:00`

## 2) Variables entorno (sin secretos)
- `EXECUTION_MODE`: `mt5`
- `MT5_SERVER`: `AdmiralsSC-Demo`
- `MT5_LOGIN`: `4286…` (truncado)
- `DB_HOST`: `localhost`
- `DB_PORT`: `5432`
- `DB_NAME`: `clawdbot`
- `DB_USER`: `clawdbot`
- `RUNTIME_METRICS_INTERVAL_S`: `60`
- `DIAGNOSTIC_BUNDLE_INTERVAL_S`: `1800`
- `TELEGRAM_LOG_ALERTS`: `true`
- `TELEGRAM_LOG_MIN_LEVEL`: `WARNING`
- `BUY_PROB_THRESHOLD`: `0.55`
- Secretos (`MT5_PASSWORD`, `DB_PASSWORD`, tokens): **omitidos**

## 3) Snapshot de arranque (`logs/bot_startup_snapshot.json`)
```json
{
  "saved_at": "2026-05-14T04:03:54.028854+00:00",
  "session_id": "5e032c35a5cf",
  "execution_mode": "mt5",
  "watchlist": [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "XRP/USDT"
  ],
  "balance": 10000.94,
  "max_positions": 2,
  "open_positions": [],
  "notes": "SL sube con peak tras activation_pct (trailing). TP_hint = peak+gap si peak>entry (compute_dynamic_tp_hint). MT5 puede mostrar otros niveles hasta sync."
}
```

## 4) Cronología SL / pico / trailing (desde JSONL)
Si **SL_now** sube cuando **peak** sube y **trailing** pasa a `true`, el ratchet funcionaba.

_Sin posiciones en el tramo final del JSONL (o vacío)._

## 5) Últimas líneas de `runtime_metrics.jsonl` (crudo)
```text
{"ts": "2026-05-14T01:28:57.922180+00:00", "session_id": "15ec223f2e9b", "execution_mode": "mt5", "balance": 10000.94, "open_count": 0, "total_pnl_session": -49.66999999999971, "symbols_open": [], "positions": [], "watchlist_len": 5}
{"ts": "2026-05-14T01:32:56.069417+00:00", "session_id": "38483633d0b5", "execution_mode": "mt5", "balance": 10000.94, "open_count": 0, "total_pnl_session": 0.0, "symbols_open": [], "positions": [], "watchlist_len": 5}
```

## 6) `logs/last_session.log` (últimas 200 líneas)
```text
2026-05-13 23:03:45 | INFO     | clawdbot | SESSION_START session_id=5e032c35a5cf pid=10768 | full JSON: bot_debug.log | esta sesión: logs/last_session.log
2026-05-13 23:03:45 | INFO     | clawdbot | 🚀 ClawdBot starting up...
2026-05-13 23:03:46 | INFO     | database.db_manager | Connected to TimescaleDB.
2026-05-13 23:03:46 | INFO     | database.db_manager | Database schema initialised.
2026-05-13 23:03:46 | INFO     | strategy.ml_predictor | MLPredictor initialised (XGBoost).
2026-05-13 23:03:46 | INFO     | clawdbot | [ENV] INITIAL_BALANCE=10000.00 (paper / fallback si MT5 no aporta equity)
2026-05-13 23:03:46 | INFO     | clawdbot | 🔌 [MT5] Connecting to MetaTrader 5 | server=AdmiralsSC-Demo | login=42861847
2026-05-13 23:03:50 | INFO     | execution.mt5_executor | [MT5] Connected to AdmiralsSC-Demo | Account: 42861847 | Balance: 10000.94 USD
2026-05-13 23:03:50 | INFO     | clawdbot | ✅ [MT5] Account balance fetched: 10000.94 USDT
2026-05-13 23:03:50 | INFO     | clawdbot | ✅ [MT5] MT5Executor initialised in LIVE mode – orders will be sent to MetaTrader 5.
2026-05-13 23:03:50 | INFO     | clawdbot | [MT5 FEED] tick_interval=0.25s  kline_poll=5.0s
2026-05-13 23:03:50 | INFO     | clawdbot | SESSION_CONFIG session_id=5e032c35a5cf execution_mode=mt5 mt5_initialized=True market_feed=True initial_balance=10000.94 watchlist=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'XRP/USDT']
2026-05-13 23:03:50 | INFO     | execution.paper_executor | Cleaned 1 orphan position(s) from state.json.
2026-05-13 23:03:50 | INFO     | risk.risk_manager | Open-position counter synchronised to 0.
2026-05-13 23:03:51 | INFO     | clawdbot | 🔄 [MT5 SYNC] No live MT5 positions at startup – starting fresh session.
2026-05-13 23:03:51 | INFO     | clawdbot | [MT5 SYNC] Startup ghost confirmation override=1 (runtime=3).
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML booster cached: BTC_USDT_v2.json
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML booster cached: ETH_USDT_v2.json
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML booster cached: SOL_USDT_v2.json
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML booster cached: DOGE_USDT_v2.json
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML booster cached: XRP_USDT_v2.json
2026-05-13 23:03:51 | INFO     | strategy.ml_predictor | load_model: file not found at C:\Users\WinterOS\Desktop\projectos\Bot\models\BTC_USDT_v1.json.
2026-05-13 23:03:51 | INFO     | clawdbot | ℹ️ No pre-trained model found – will warm-start from historical DB data.
2026-05-13 23:03:51 | ERROR    | utils.telegram_notifier | Failed to send Telegram alert: 
2026-05-13 23:03:51 | WARNING  | strategy.ml_predictor | warm_start: not enough labelled samples (0). Model not trained.
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML model warm-started with 13 historical prices for BTC/USDT.
2026-05-13 23:03:51 | WARNING  | strategy.ml_predictor | warm_start: not enough labelled samples (0). Model not trained.
2026-05-13 23:03:51 | INFO     | clawdbot | ✅ ML model warm-started with 13 historical prices for ETH/USDT.
2026-05-13 23:03:52 | WARNING  | strategy.ml_predictor | warm_start: not enough labelled samples (0). Model not trained.
2026-05-13 23:03:52 | INFO     | clawdbot | ✅ ML model warm-started with 13 historical prices for SOL/USDT.
2026-05-13 23:03:52 | WARNING  | strategy.ml_predictor | warm_start: not enough labelled samples (0). Model not trained.
2026-05-13 23:03:52 | INFO     | clawdbot | ✅ ML model warm-started with 13 historical prices for DOGE/USDT.
2026-05-13 23:03:52 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:52 | WARNING  | strategy.ml_predictor | warm_start: not enough labelled samples (0). Model not trained.
2026-05-13 23:03:52 | INFO     | clawdbot | ✅ ML model warm-started with 13 historical prices for XRP/USDT.
2026-05-13 23:03:52 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m candles for BTC/USDT.
2026-05-13 23:03:52 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:52 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m closes for dashboard RSI (BTC/USDT).
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 1H candles for BTC/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 4H candles for BTC/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m candles for ETH/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m closes for dashboard RSI (ETH/USDT).
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 1H candles for ETH/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 4H candles for ETH/USDT.
2026-05-13 23:03:53 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m candles for SOL/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m closes for dashboard RSI (SOL/USDT).
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 1H candles for SOL/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 4H candles for SOL/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m candles for DOGE/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m closes for dashboard RSI (DOGE/USDT).
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 1H candles for DOGE/USDT.
2026-05-13 23:03:53 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 4H candles for DOGE/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m candles for XRP/USDT.
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 15m closes for dashboard RSI (XRP/USDT).
2026-05-13 23:03:53 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 1H candles for XRP/USDT.
2026-05-13 23:03:54 | INFO     | clawdbot.preload | [MT5] Preloaded 1000 4H candles for XRP/USDT.
2026-05-13 23:03:54 | INFO     | clawdbot | [AUDIT] Decision pipeline: ML_BUY_PROB>=0.55 | symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'XRP/USDT'] (ML-only entries).
2026-05-13 23:03:54 | INFO     | clawdbot | [AUDIT] Session state reset: max_drawdown=0.0  trading_halted=False
2026-05-13 23:03:54 | INFO     | clawdbot | [AUDIT] UI coherence: COMPRAR/BUY only when ml_signals[symbol]==BUY and prob>=0.55.
2026-05-13 23:03:54 | WARNING  | clawdbot | [RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%.
2026-05-13 23:03:54 | INFO     | utils.runtime_snapshot | Runtime snapshot written: C:\Users\WinterOS\Desktop\projectos\Bot\logs\bot_startup_snapshot.json
2026-05-13 23:03:54 | INFO     | clawdbot | 📄 Startup snapshot: C:\Users\WinterOS\Desktop\projectos\Bot\logs\bot_startup_snapshot.json
2026-05-13 23:03:54 | INFO     | clawdbot | 📎 Primer DIAGNOSTIC_FOR_REVIEW.md → C:\Users\WinterOS\Desktop\projectos\Bot\DIAGNOSTIC_FOR_REVIEW.md
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task mt5_market_client (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task market_consumer (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task signal_emitter (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task dashboard_logger (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task position_sync (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task health_monitor (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task close_pending_reconciler (critical=True)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task weekly_retrainer (critical=False)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task telegram_command_poller (critical=False)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task weekly_report (critical=False)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task monthly_report (critical=False)
2026-05-13 23:03:54 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task web_dashboard (critical=False)
2026-05-13 23:03:54 | INFO     | vibe_mcp.server | 🧠 VIBE MCP internal server listening on http://0.0.0.0:5000
2026-05-13 23:03:54 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:55 | INFO     | utils.telegram_notifier | Telegram alert sent successfully.
2026-05-13 23:03:56 | INFO     | aiohttp.access | 127.0.0.1 [13/May/2026:23:03:56 -0500] "GET /health HTTP/1.1" 200 175 "-" "Python/3.12 aiohttp/3.13.3"
2026-05-13 23:03:56 | INFO     | clawdbot.vibe | [VIBE] Hybrid client connected to internal HTTP server.
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task vibe_journal (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task vibe_backtest (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task vibe_patterns (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task vibe_factors (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task vibe_shadow (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task vibe_swarm (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot | Vibe-Trading active - 5 scheduled tasks + SWARM (supervised).
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task runtime_metrics (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot | 📊 Runtime metrics JSONL every 60s → logs/runtime_metrics.jsonl
2026-05-13 23:03:56 | INFO     | clawdbot.supervisor | [SUPERVISOR] Registered task diagnostic_bundle (critical=False)
2026-05-13 23:03:56 | INFO     | clawdbot | 📎 Cada 1800s → C:\Users\WinterOS\Desktop\projectos\Bot\DIAGNOSTIC_FOR_REVIEW.md (un solo archivo para pegar al asistente)
2026-05-13 23:03:56 | INFO     | clawdbot.retrainer | [PRO] Weekly Re-trainer sleeping 67.9 hours until Sunday 00:00 UTC.
2026-05-13 23:03:56 | INFO     | clawdbot.reports.weekly | Weekly report scheduled for 2026-05-18T02:00:00+00:00 UTC (338163s).
2026-05-13 23:03:56 | INFO     | clawdbot.reports.monthly | Monthly report scheduled for 2026-06-02T02:00:00+00:00 UTC (1634163s).
2026-05-13 23:03:56 | INFO     | bot.web_server | 🌐 Web dashboard server listening on http://0.0.0.0:8080
2026-05-13 23:04:11 | INFO     | strategy.regime_predictor | [REGIME] Loaded model for XRP/USDT from C:\Users\WinterOS\Desktop\projectos\Bot\models\XRP_USDT_regime.json
2026-05-13 23:04:11 | INFO     | clawdbot.signal | [REGIME] Entry blocked for XRP/USDT: market is RANGING (prob=0.10)
2026-05-13 23:04:26 | INFO     | clawdbot.signal | [REGIME] Entry blocked for XRP/USDT: market is RANGING (prob=0.10)
2026-05-13 23:04:41 | INFO     | clawdbot.signal | [REGIME] Entry blocked for XRP/USDT: market is RANGING (prob=0.10)
```

## 7) ERROR / WARNING en `bot_debug` - `bot_debug.log` (raíz del repo) (filtrado)
```text
{"timestamp": "2026-05-13T22:43:58.512509+00:00", "session_id": "c7f93d963e63", "level": "ERROR", "logger": "utils.telegram_notifier", "message": "Failed to send Telegram alert: "}
{"timestamp": "2026-05-13T22:43:58.518515+00:00", "session_id": "c7f93d963e63", "level": "ERROR", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade failed trade_id=222 — invalid input for query argument $1: '222' ('str' object cannot be interpreted as an integer)", "exception": "Traceback (most recent call last):\n  File \"asyncpg/protocol/prepared_stmt.pyx\", line 175, in asyncpg.protocol.protocol.PreparedStatementState._encode_bind_msg\n  File \"asyncpg/protocol/codecs/base.pyx\", line 251, in asyncpg.protocol.protocol.Codec.encode\n  File \"asyncpg/protocol/codecs/base.pyx\", line 153, in asyncpg.protocol.protocol.Codec.encode_scalar\n  File \"asyncpg/pgproto/codecs/int.pyx\", line 54, in asyncpg.pgproto.pgproto.int4_encode\nTypeError: 'str' object cannot be interpreted as an integer\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\execution\\mt5_executor.py\", line 1929, in _apply_mt5_closed_bookkeeping\n    await self._db.close_trade(\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\database\\db_manager.py\", line 455, in close_trade\n    await self.update_trade_exit(\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\database\\db_manager.py\", line 433, in update_trade_exit\n    await conn.execute(\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\.venv\\Lib\\site-packages\\asyncpg\\connection.py\", line 357, in execute\n    _, status, _ = await self._execute(\n                   ^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\.venv\\Lib\\site-packages\\asyncpg\\connection.py\", line 1873, in _execute\n    result, _ = await self.__execute(\n                ^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\.venv\\Lib\\site-packages\\asyncpg\\connection.py\", line 1970, in __execute\n    result, stmt = await self._do_execute(\n                   ^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\WinterOS\\Desktop\\projectos\\Bot\\.venv\\Lib\\site-packages\\asyncpg\\connection.py\", line 2033, in _do_execute\n    result = await executor(stmt, None)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"asyncpg/protocol/protocol.pyx\", line 184, in bind_execute\n  File \"asyncpg/protocol/prepared_stmt.pyx\", line 204, in asyncpg.protocol.protocol.PreparedStatementState._encode_bind_msg\nasyncpg.exceptions.DataError: invalid input for query argument $1: '222' ('str' object cannot be interpreted as an integer)"}
{"timestamp": "2026-05-13T22:44:02.203592+00:00", "session_id": "c7f93d963e63", "level": "ERROR", "logger": "utils.telegram_notifier", "message": "Telegram API returned HTTP 400: {\"ok\":false,\"error_code\":400,\"description\":\"Bad Request: message text is empty\"}"}
{"timestamp": "2026-05-13T22:44:09.950772+00:00", "session_id": "c7f93d963e63", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m — valores activos prob≥0.55 risk=2.0%."}
{"timestamp": "2026-05-13T22:52:18.430999+00:00", "session_id": "019765fb3aed", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-13T22:53:22.063877+00:00", "session_id": "1888f362b2a9", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-13T23:15:35.519088+00:00", "session_id": "d022b3672e89", "level": "WARNING", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade skipped for numeric trade_id=222 (not in DB)."}
{"timestamp": "2026-05-13T23:15:37.461971+00:00", "session_id": "d022b3672e89", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-13T23:26:35.453584+00:00", "session_id": "7ae6b17327dd", "level": "WARNING", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade skipped for numeric trade_id=222 (not in DB)."}
{"timestamp": "2026-05-13T23:26:37.123020+00:00", "session_id": "7ae6b17327dd", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-13T23:43:55.601979+00:00", "session_id": "6d47d48c8dde", "level": "WARNING", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade skipped for numeric trade_id=222 (not in DB)."}
{"timestamp": "2026-05-13T23:43:56.742557+00:00", "session_id": "6d47d48c8dde", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T00:06:28.556308+00:00", "session_id": "18d73d15ee01", "level": "WARNING", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade skipped for numeric trade_id=222 (not in DB)."}
{"timestamp": "2026-05-14T00:06:29.494026+00:00", "session_id": "18d73d15ee01", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T00:06:32.552354+00:00", "session_id": "18d73d15ee01", "level": "WARNING", "logger": "clawdbot", "message": "Vibe-Trading MCP client started - 5 scheduled tasks active (swarm disabled)."}
{"timestamp": "2026-05-14T01:08:07.156606+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade skipped for numeric trade_id=222 (not in DB)."}
{"timestamp": "2026-05-14T01:08:07.488262+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:08:07.793576+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:08:08.090833+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:08:08.451190+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:08:08.767920+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:08:09.924242+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T01:08:12.592741+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "clawdbot.vibe", "message": "[VIBE] Internal HTTP server not reachable (Cannot connect to host localhost:5000 ssl:default [Multiple exceptions: [Errno 10061] Connect call failed ('::1', 5000, 0, 0), [Errno 10061] Connect call failed ('127.0.0.1', 5000)])."}
{"timestamp": "2026-05-14T01:08:12.594746+00:00", "session_id": "24881ba720a7", "level": "WARNING", "logger": "clawdbot", "message": "Vibe-Trading start failed (Cannot connect to host localhost:5000 ssl:default [Multiple exceptions: [Errno 10061] Connect call failed ('::1', 5000, 0, 0), [Errno 10061] Connect call failed ('127.0.0.1', 5000)]) - tools disabled. Bot runs normally."}
{"timestamp": "2026-05-14T01:09:42.660481+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:09:42.986699+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:09:43.292825+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:09:43.599295+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:09:43.933834+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:09:45.222847+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T01:10:18.502974+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "clawdbot.vibe", "message": "[VIBE] Internal HTTP server not reachable after 10s (Cannot connect to host localhost:5000 ssl:default [Multiple exceptions: [Errno 10061] Connect call failed ('127.0.0.1', 5000), [Errno 10061] Connect call failed ('::1', 5000, 0, 0)])."}
{"timestamp": "2026-05-14T01:10:18.504976+00:00", "session_id": "7ad543e5e38d", "level": "WARNING", "logger": "clawdbot", "message": "Vibe-Trading start failed (Cannot connect to host localhost:5000 ssl:default [Multiple exceptions: [Errno 10061] Connect call failed ('127.0.0.1', 5000), [Errno 10061] Connect call failed ('::1', 5000, 0, 0)]) - tools disabled. Bot runs normally."}
{"timestamp": "2026-05-14T01:12:38.096990+00:00", "session_id": "0b81a3d29987", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:12:38.410106+00:00", "session_id": "0b81a3d29987", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:12:38.730531+00:00", "session_id": "0b81a3d29987", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:12:39.062344+00:00", "session_id": "0b81a3d29987", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:12:39.387127+00:00", "session_id": "0b81a3d29987", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:12:40.392684+00:00", "session_id": "0b81a3d29987", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T01:19:55.256986+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "execution.mt5_executor", "message": "[MT5][DB] close_trade skipped for numeric trade_id=222 (not in DB)."}
{"timestamp": "2026-05-14T01:19:55.626484+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:19:56.044020+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:19:56.471916+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:19:56.854847+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:19:57.261751+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T01:19:58.176262+00:00", "session_id": "aa41bee58abd", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T01:27:55.129759+00:00", "session_id": "15ec223f2e9b", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T01:28:57.938271+00:00", "session_id": "15ec223f2e9b", "level": "WARNING", "logger": "clawdbot.vibe", "message": "[VIBE] analyze_trade_journal failed: cannot reindex on an axis with duplicate labels"}
{"timestamp": "2026-05-14T01:31:53.471600+00:00", "session_id": "38483633d0b5", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T01:32:56.076928+00:00", "session_id": "38483633d0b5", "level": "WARNING", "logger": "clawdbot.vibe", "message": "[VIBE] analyze_trade_journal failed: cannot reindex on an axis with duplicate labels"}
{"timestamp": "2026-05-14T01:34:46.585088+00:00", "session_id": "c167dc20cf6b", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
{"timestamp": "2026-05-14T04:03:51.292697+00:00", "session_id": "5e032c35a5cf", "level": "ERROR", "logger": "utils.telegram_notifier", "message": "Failed to send Telegram alert: "}
{"timestamp": "2026-05-14T04:03:51.338453+00:00", "session_id": "5e032c35a5cf", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T04:03:51.712179+00:00", "session_id": "5e032c35a5cf", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T04:03:52.044031+00:00", "session_id": "5e032c35a5cf", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T04:03:52.349835+00:00", "session_id": "5e032c35a5cf", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T04:03:52.629776+00:00", "session_id": "5e032c35a5cf", "level": "WARNING", "logger": "strategy.ml_predictor", "message": "warm_start: not enough labelled samples (0). Model not trained."}
{"timestamp": "2026-05-14T04:03:54.026348+00:00", "session_id": "5e032c35a5cf", "level": "WARNING", "logger": "clawdbot", "message": "[RIESGO] Perfil ETH/USDT 15m - valores activos prob>=0.55 risk=2.0%."}
```

## 8) `audit.log` (últimas 120 líneas)
```text
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
# [HORA]    | [MONEDA]   | PRECIO_ACTUAL | PICO_MÁX    | ATR_VAL  | STOP_CALCULADO | XGB_CONF | DISTANCIA_%
```

## 9) `logs/trade_journal.csv` (últimas 80 líneas)
```text
t1,BTC/USDT,2026-05-14T01:06:04.056121+00:00,2026-05-14T01:06:04.057122+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:05.086476+00:00,2026-05-14T01:06:05.087476+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:06.112022+00:00,2026-05-14T01:06:06.113029+00:00,10.00000000,9.29822440,1000.00,-70.1776,-70.1776,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:07.159702+00:00,2026-05-14T01:06:07.160702+00:00,75581.18555783,75505.60437227,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:08.137292+00:00,2026-05-14T01:06:08.138293+00:00,75581.18555783,44075.81082982,1000.00,-416.8415,-416.8415,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:09.148778+00:00,2026-05-14T01:06:09.148778+00:00,28082.68843533,28054.60574690,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:10.215658+00:00,2026-05-14T01:06:10.216666+00:00,28082.68843533,23909.35294989,1000.00,-148.6088,-148.6088,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:11.298121+00:00,2026-05-14T01:06:11.298121+00:00,30546.78115415,30516.23437299,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:12.357075+00:00,2026-05-14T01:06:12.358078+00:00,30546.78115415,26145.21139766,1000.00,-144.0928,-144.0928,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:13.402455+00:00,2026-05-14T01:06:13.403461+00:00,100.00000000,99.90000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:14.458342+00:00,2026-05-14T01:06:14.458342+00:00,100.00000000,80.00000000,1000.00,-200.0000,-200.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:15.420387+00:00,2026-05-14T01:06:15.421892+00:00,10.00000000,8.00000000,1000.00,-200.0000,-200.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:16.400513+00:00,2026-05-14T01:06:16.401516+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:17.353576+00:00,2026-05-14T01:06:17.354579+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:18.310895+00:00,2026-05-14T01:06:18.311894+00:00,10.00000000,9.90000000,1000.00,-10.0000,-10.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:19.274677+00:00,2026-05-14T01:06:19.275674+00:00,25.00000000,13.93461972,1000.00,-442.6152,-442.6152,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:20.279235+00:00,2026-05-14T01:06:20.279235+00:00,25.00000000,24.97500000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:21.300244+00:00,2026-05-14T01:06:21.301765+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:22.373313+00:00,2026-05-14T01:06:22.374304+00:00,10.00000000,9.90000000,1000.00,-10.0000,-10.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:23.423032+00:00,2026-05-14T01:06:23.424037+00:00,42620.50301211,30383.43408500,1000.00,-287.1170,-287.1170,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:24.491452+00:00,2026-05-14T01:06:24.492458+00:00,42620.50301211,30383.43408500,1000.00,-287.1170,-287.1170,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:25.576722+00:00,2026-05-14T01:06:25.576722+00:00,10.00000000,7.12883048,1000.00,-287.1170,-287.1170,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:26.718206+00:00,2026-05-14T01:06:26.719200+00:00,10.00000000,7.12883048,1000.00,-287.1170,-287.1170,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:27.866541+00:00,2026-05-14T01:06:27.867543+00:00,99029.93300190,49514.96650095,1000.00,-500.0000,-500.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:29.023529+00:00,2026-05-14T01:06:29.023529+00:00,10.00000000,5.00000000,1000.00,-500.0000,-500.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:30.020195+00:00,2026-05-14T01:06:30.021700+00:00,10.00000000,5.00000000,1000.00,-500.0000,-500.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:31.215751+00:00,2026-05-14T01:06:31.215751+00:00,100000.00000000,58799.10848444,1000.00,-412.0089,-412.0089,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:32.166726+00:00,2026-05-14T01:06:32.167726+00:00,10.00000000,5.87991085,1000.00,-412.0089,-412.0089,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:33.150492+00:00,2026-05-14T01:06:33.150492+00:00,10.00000000,5.87991085,1000.00,-412.0089,-412.0089,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:34.108654+00:00,2026-05-14T01:06:34.108654+00:00,62667.65037796,62280.57757262,1000.00,-6.1766,-6.1766,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:35.146928+00:00,2026-05-14T01:06:35.147928+00:00,62667.65037796,62604.98272758,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:36.230202+00:00,2026-05-14T01:06:36.231710+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:37.328361+00:00,2026-05-14T01:06:37.328361+00:00,63750.46443948,53672.17988714,1000.00,-158.0896,-158.0896,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:38.356660+00:00,2026-05-14T01:06:38.357657+00:00,10.00000000,8.41910414,1000.00,-158.0896,-158.0896,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:39.300059+00:00,2026-05-14T01:06:39.300059+00:00,10.00000000,8.41910414,1000.00,-158.0896,-158.0896,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:40.319559+00:00,2026-05-14T01:06:40.319559+00:00,30796.85713758,26529.94859039,1000.00,-138.5501,-138.5501,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:41.349683+00:00,2026-05-14T01:06:41.349683+00:00,30796.85713758,26529.94859039,1000.00,-138.5501,-138.5501,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:42.312063+00:00,2026-05-14T01:06:42.313067+00:00,10.00000000,8.61449870,1000.00,-138.5501,-138.5501,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:43.282518+00:00,2026-05-14T01:06:43.283513+00:00,1000.00000000,685.44600216,1000.00,-314.5540,-314.5540,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:44.261406+00:00,2026-05-14T01:06:44.261406+00:00,1000.00000000,999.00000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:45.255637+00:00,2026-05-14T01:06:45.255637+00:00,1000.00000000,999.00000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:46.218826+00:00,2026-05-14T01:06:46.219834+00:00,1000.00000000,999.00000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:47.187527+00:00,2026-05-14T01:06:47.188532+00:00,1000.00000000,990.00000000,1000.00,-10.0000,-10.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:48.183372+00:00,2026-05-14T01:06:48.184372+00:00,120.00000000,80.00000000,1000.00,-333.3333,-333.3333,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:49.167389+00:00,2026-05-14T01:06:49.168387+00:00,120.00000000,119.88000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:50.167008+00:00,2026-05-14T01:06:50.167008+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:51.218497+00:00,2026-05-14T01:06:51.219496+00:00,9081.05396132,8145.68746955,1000.00,-103.0020,-103.0020,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:52.276413+00:00,2026-05-14T01:06:52.277413+00:00,9081.05396132,9071.97290735,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:53.319746+00:00,2026-05-14T01:06:53.321253+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:54.347230+00:00,2026-05-14T01:06:54.347230+00:00,26128.76614042,17125.68899318,1000.00,-344.5657,-344.5657,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:55.474803+00:00,2026-05-14T01:06:55.474803+00:00,26128.76614042,17125.68899318,1000.00,-344.5657,-344.5657,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:56.557819+00:00,2026-05-14T01:06:56.557819+00:00,10.00000000,6.55434279,1000.00,-344.5657,-344.5657,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:57.648929+00:00,2026-05-14T01:06:57.648929+00:00,10.00000000,6.55434279,1000.00,-344.5657,-344.5657,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:58.762296+00:00,2026-05-14T01:06:58.762296+00:00,70245.33182587,59557.35730513,1000.00,-152.1521,-152.1521,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:06:59.816844+00:00,2026-05-14T01:06:59.817835+00:00,70245.33182587,59557.35730513,1000.00,-152.1521,-152.1521,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:00.769014+00:00,2026-05-14T01:07:00.769519+00:00,70245.33182587,59557.35730513,1000.00,-152.1521,-152.1521,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:01.734260+00:00,2026-05-14T01:07:01.734260+00:00,10.00000000,8.47847903,1000.00,-152.1521,-152.1521,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:02.687666+00:00,2026-05-14T01:07:02.688670+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:03.662047+00:00,2026-05-14T01:07:03.662047+00:00,12.00000000,11.98800000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:04.728625+00:00,2026-05-14T01:07:04.729625+00:00,12.00000000,11.98800000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:05.825057+00:00,2026-05-14T01:07:05.825057+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:06.947442+00:00,2026-05-14T01:07:06.948442+00:00,71903.13443869,48018.65461516,1000.00,-332.1758,-332.1758,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:07.899045+00:00,2026-05-14T01:07:07.899045+00:00,71903.13443869,48018.65461516,1000.00,-332.1758,-332.1758,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:08.876811+00:00,2026-05-14T01:07:08.877806+00:00,10.00000000,6.67824219,1000.00,-332.1758,-332.1758,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:09.867072+00:00,2026-05-14T01:07:09.868582+00:00,10.00000000,6.67824219,1000.00,-332.1758,-332.1758,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:10.906349+00:00,2026-05-14T01:07:10.907348+00:00,16897.19452062,15355.91670946,1000.00,-91.2150,-91.2150,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:11.908602+00:00,2026-05-14T01:07:11.908602+00:00,16897.19452062,16880.29732610,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:12.975230+00:00,2026-05-14T01:07:12.976230+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:14.027175+00:00,2026-05-14T01:07:14.028176+00:00,48805.01498622,31366.57348326,1000.00,-357.3084,-357.3084,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:15.059167+00:00,2026-05-14T01:07:15.060420+00:00,48805.01498622,48756.20997123,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:16.083466+00:00,2026-05-14T01:07:16.084476+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:17.140270+00:00,2026-05-14T01:07:17.141266+00:00,78515.15209068,39257.57604534,1000.00,-500.0000,-500.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:18.186015+00:00,2026-05-14T01:07:18.187519+00:00,78515.15209068,39257.57604534,1000.00,-500.0000,-500.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:19.266609+00:00,2026-05-14T01:07:19.267608+00:00,78515.15209068,78436.63693859,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:20.302041+00:00,2026-05-14T01:07:20.303046+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:21.348210+00:00,2026-05-14T01:07:21.349211+00:00,26393.71766831,14751.17367572,1000.00,-441.1104,-441.1104,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:22.383756+00:00,2026-05-14T01:07:22.383756+00:00,26393.71766831,26367.32395064,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:23.437238+00:00,2026-05-14T01:07:23.438237+00:00,26393.71766831,26367.32395064,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:24.389570+00:00,2026-05-14T01:07:24.389570+00:00,10.00000000,9.99000000,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
t1,BTC/USDT,2026-05-14T01:07:25.357766+00:00,2026-05-14T01:07:25.358760+00:00,26393.71766831,26367.32395064,1000.00,-1.0000,-1.0000,stop_loss,0.7000,0.0
```
