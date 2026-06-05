# ClawdBot — Auditoría Consolidada
_Consolidado: 2026-06-05 | Branch: feat/profitable-rebuild-2026-06_

---

## 1. Configuración Activa

| Parámetro | Valor |
|---|---|
| `EXECUTION_MODE` | `mt5` |
| `MT5_SERVER` | `Pepperstone-Demo` |
| `DB_NAME` | `clawdbot` |
| `BUY_PROB_THRESHOLD` | `0.55` |
| `RUNTIME_METRICS_INTERVAL_S` | `60` |
| `DIAGNOSTIC_BUNDLE_INTERVAL_S` | `1800` |
| `TELEGRAM_LOG_ALERTS` | `true` |
| `TELEGRAM_LOG_MIN_LEVEL` | `WARNING` |

**Snapshot arranque** (session `a222dd150ede`, 2026-06-04T16:21Z):
```json
{
  "execution_mode": "mt5",
  "watchlist": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "NEAR/USDT"],
  "balance": 10000.0,
  "max_positions": 2,
  "open_positions": []
}
```

---

## 2. Estado de Sesión

| Campo | Valor |
|---|---|
| `total_pnl` | `+60.0` |
| `open_positions` | `{}` (vacío) |
| `kill_switch_active` | `false` |
| `demoted_symbols` | `BTC/USDT` (threshold 0.95) |

---

## 3. Kill Switch — Historial Completo

Archivo fuente: `data/kill_switch_state.json` + `logs/kill_switch_events.log`

| Timestamp (UTC) | Evento | Razón | Duración |
|---|---|---|---|
| 2026-06-03T01:18:11 | activated | `T1:daily_pnl=-6.0%` | 24h |
| 2026-06-03T01:18:11 | deactivated | — | — |
| 2026-06-03T01:18:11 | symbol_demoted | `BTC/USDT` @ threshold=0.95 | — |
| 2026-06-03T01:18:11 | activated | `manual_test` | 0.001h |
| 2026-06-03T01:18:11 | deactivated | — | — |
| 2026-06-03T01:20:55 | symbol_demoted | `TEST/USDT` @ threshold=0.95 | — |
| 2026-06-03T14:10:52 | activated | `3_consecutive_losses` | 4h |
| 2026-06-03T14:10:52 | deactivated | — | — |
| 2026-06-03T14:12:20 | symbol_demoted | `BTC/USDT` @ threshold=0.95 | — |

**Estado actual:** `active=false`, BTC/USDT demoted (threshold 0.95 permanente)

---

## 4. Drift Detection — Eventos

Archivo fuente: `logs/drift_events.log`

| Timestamp (UTC) | Símbolo | KS-stat | p-value | n_live | n_ref |
|---|---|---|---|---|---|
| 2026-06-03T01:20:55 | TEST/USDT | 0.7973 | 0.0 | 150 | 500 |
| 2026-06-03T14:12:20 | BTC/USDT | 0.7567 | 0.0 | 50 | 300 |

---

## 5. Symbol Scan — Candidatos Rechazados

Archivo fuente: `logs/symbol_scan_results.csv`

| Símbolo | Trades | Win% | PnL% | MaxDD% | Score | Veredicto |
|---|---|---|---|---|---|---|
| WIF/USDT | 64 | 32.8% | -2.59% | 7.91% | -2.59 | REJECTED |
| PEPE/USDT | 45 | 37.8% | -5.53% | 6.86% | -5.53 | REJECTED |
| APT/USDT | 59 | 52.5% | -5.12% | 9.80% | -7.45 | REJECTED |
| ENA/USDT | 59 | 38.9% | -7.32% | 12.20% | -15.72 | REJECTED |

Ninguno pasó. Watchlist activa viene del retrain 2026-06: BTC, ETH, SOL, LINK, NEAR.

---

## 6. Backtest / Walk-Forward — Resultados Consolidados

### Backtest Disco 30d (2026-06-04, threshold=0.65, balance=$10k)

| Símbolo | Trades | Win% | PnL% | MaxDD% | PF | Sharpe |
|---|---|---|---|---|---|---|
| BTC/USDT | 0 | — | 0% | — | — | — |
| ETH/USDT | 3 | 0% | -1.72% | 2.13% | 0.0 | -19.65 |
| SOL/USDT | 11 | **81.8%** | **+8.61%** | 2.36% | 5.55 | 2.72 |
| LINK/USDT | 13 | **84.6%** | **+6.40%** | 0.64% | 46.51 | 4.10 |
| NEAR/USDT | 71 | 67.6% | **+37.11%** | 3.79% | 3.24 | 1.85 |

### Verificación Final 30d (2026-06-03, watchlist expandida)

| Símbolo | Trades | Win% | PnL% | MaxDD% | PF | Sharpe |
|---|---|---|---|---|---|---|
| BTC/USDT | 0 | — | 0% | — | — | — |
| ETH/USDT | 0 | — | 0% | — | — | — |
| SOL/USDT | 10 | **90.0%** | **+8.80%** | 2.39% | 6.59 | 3.05 |
| LINK/USDT | 13 | **92.3%** | **+6.05%** | 0.64% | 112.43 | 4.67 |
| NEAR/USDT | 82 | 62.2% | **+27.70%** | 3.97% | 2.16 | 1.23 |
| ATOM/USDT | 40 | 65.0% | **+10.03%** | 3.63% | 2.41 | 1.31 |
| DOGE/USDT | 20 | 60.0% | -1.66% | 2.29% | 0.82 | -0.25 |
| JTO/USDT | — | — | — | — | — | modelo faltante |

### Walk-Forward BTC/USDT — 5 Folds (2025-07 → 2026-05, 30m candles)

| Fold | Período | Trades | Win% | PnL% | MaxDD% | Alpha vs B&H |
|---|---|---|---|---|---|---|
| 1 | 2025-07-18 → 2025-09-19 | 14 | 57.1% | -1.38% | 2.87% | -0.51% |
| 2 | 2025-09-19 → 2025-11-20 | 6 | 66.7% | +2.06% | 1.37% | **+23.97%** |
| 3 | 2025-11-20 → 2026-01-22 | 15 | 66.7% | +1.33% | 2.54% | +2.22% |
| 4 | 2026-01-22 → 2026-03-25 | 14 | 71.4% | **+7.79%** | 2.85% | **+28.44%** |
| 5 | 2026-03-25 → 2026-05-27 | 21 | 57.1% | +0.11% | 4.14% | -6.38% |
| **Avg** | | **14** | **63.8%** | **+1.98%** | **3.15%** | **+9.55%** |

**Veredicto BTC WF:** `robust=true` — 4/5 folds profitable, 3/5 beat B&H

### Walk-Forward ETH/USDT — 5 Folds (2025-12 → 2026-05, 15m candles)

| Fold | Win% | PnL% | PF | Alpha vs B&H |
|---|---|---|---|---|
| 1 | 55.6% | -0.32% | 1.23 | -1.69% |
| 2 | 47.4% | -1.66% | 0.88 | +33.23% |
| 3 | 54.5% | +0.53% | 1.34 | -9.89% |
| 4 | 40.0% | -0.05% | 1.18 | -6.03% |
| 5 | 37.5% | -2.72% | 0.46 | +7.53% |
| **Avg** | **47.0%** | **-0.84%** | **1.02** | **+4.63%** |

**Veredicto ETH WF:** `robust=false` — 1/5 folds profitable. ETH en watchlist pero bajo vigilancia.

---

## 7. Trade Journal

Archivo: `logs/trade_journal_clean.csv` (264KB — demasiado grande para incrustar)

**ADVERTENCIA:** `logs/trade_journal.csv` contiene datos de test inválidos del 2026-05-13 — entradas/salidas en milisegundos con precios absurdos (entry=5.00, exit=321,994). Son artefactos de un test de paper executor. Ignorar para análisis de performance real.

---

## 8. Archivos Eliminados en Esta Limpieza

Los siguientes archivos fueron eliminados (datos históricos preservados arriba):

### Logs de Debug / Runtime
- `bot_debug.log` — 14,256 líneas JSON structured log (session 2026-06-04)
- `main_stdout.log` / `main_stderr.log` — stdout/stderr de sesión
- `logs/bot_stdout.log` / `logs/bot_stderr.log`
- `logs/last_session.log` / `logs/last_session_err.log`
- `logs/dashboard_api.log` / `logs/nextjs.log`
- `tests_run_output.log`
- `audit.log` — template vacío repetido 43 veces, sin datos reales

### Logs de Restart
- `logs/bot_restart.log` through `logs/bot_restart6.log` (6 archivos)

### Backtests
- `logs/backtest_baseline.log` / `backtest_after.log` / `backtest_final.log`
- `logs/walk_forward.log` / `walk_forward_baseline.log` / `walk_forward_final.log`
- `logs/wf_post_retrain.log` / `wf_sol_bnb.log`
- `logs/eth_sweep_wf.csv` / `eth_sweep_wf.log`
- `logs/eth_tf_test.log` / `eth_revalidate_30k.log`
- `logs/btc_enhance.log`

### Param Sweeps
- `logs/param_sweep.log`
- `logs/param_sweep_BTC_USDT.csv` / `.log`
- `logs/param_sweep_ETH_USDT.csv` / `.log`
- `logs/param_sweep_ATOM_USDT_disk.csv`
- `logs/param_sweep_BTC_USDT_disk.csv`
- `logs/param_sweep_DOGE_USDT_disk.csv`
- `logs/param_sweep_ETH_USDT_disk.csv`
- `logs/param_sweep_LINK_USDT_disk.csv`
- `logs/param_sweep_NEAR_USDT_disk.csv`
- `logs/param_sweep_SOL_USDT_disk.csv`
- `logs/param_sweep_XRP_USDT_disk.csv`

### Retrain Logs
- `logs/retrain_365d.log` / `retrain_365d_thr008.log`
- `logs/retrain_180d.log` / `retrain_force.log`

### Resultados JSON de Scripts (incorporados en §6)
- `logs/backtest_disk_loaded_30d.json` / `60d.json` / `7d.json`
- `logs/backtest_results.json`
- `logs/baseline_verification_30d.json`
- `logs/experiments_alpha_results.json`
- `logs/final_verification_report.json`
- `logs/pair_trade_backtest_90d.json`
- `logs/param_sweep_summary.json` / `param_sweep_summary_disk.json`
- `logs/portfolio_backtest_30d.json`
- `logs/threshold_sweep_results.json`
- `logs/walk_forward_oos_report.json`
- `logs/walk_forward_summary.json`
- `logs/reports/` — HTML reports directorio

### Otros
- `logs/shadow_run_signals.csv`
- `logs/trade_journal.csv` — datos de test inválidos (ver §7)
- `DIAGNOSTIC_FOR_REVIEW.md` — sustituido por este archivo
- `logs/vibe_runs/` — 200+ artefactos CSV/JSON de runs VIBE

---

## 9. Archivos Activos Retenidos

| Archivo | Propósito |
|---|---|
| `data/kill_switch_state.json` | Estado live kill switch — el bot lo lee/escribe |
| `logs/kill_switch_events.log` | Append-only kill switch events |
| `logs/drift_events.log` | Append-only drift events |
| `logs/state.json` | Estado de sesión live |
| `logs/vibe_state.json` | Estado VIBE live |
| `logs/trade_journal_clean.csv` | Historial de trades (datos reales) |
| `logs/symbol_scan_results.csv` | Resultados del último symbol scan |

---

## 10. TODO — Limpieza de Código

### Alta prioridad (romper antes de producción real)
- [ ] `logs/trade_journal.csv` — eliminar o truncar; tiene ~miles de filas de datos de test inválidos con precios absurdos que contaminan cualquier análisis
- [ ] `bot/signal_emitter.py` — revisar si sigue generando señales para `TEST/USDT` (símbolo demotado)
- [ ] `execution/paper_executor.py` — verificar que el executor no genera trades con precios ficticios (entry=5.00, exit=321k) en backtests

### Media prioridad
- [ ] `audit.log` (archivo fuente) — el generador escribe la misma línea de header 43 veces. Buscar y corregir el loop de escritura.
- [ ] `scripts/export_diagnostic_bundle.py` — genera `DIAGNOSTIC_FOR_REVIEW.md` con 1054 líneas de JSONL crudo. Simplificar para que solo exporte el resumen (§1-§7 de este archivo).
- [ ] `bot/loops.py` — verificar que `BTC/USDT` con threshold demoted no bloquea el loop de señales completo
- [ ] `vibe/scheduled_tasks.py` — los vibe_runs generan 200+ archivos CSV/JSON por run. Agregar rotación/limpieza automática.

### Baja prioridad
- [ ] `dashboard/api/routers/` — múltiples routers modificados en el branch. Auditar que ninguno expone paths con datos de test.
- [ ] `.env` / configuración — `DIAGNOSTIC_BUNDLE_INTERVAL_S=1800` genera un bundle cada 30min; considerar deshabilitar en producción o subir a 24h.
- [ ] `agent/uploads/trade_journal.csv` — copia del journal en directorio agent. Verificar si es necesario o artefacto.

---

_Para regenerar este archivo: consolidar manualmente desde `data/kill_switch_state.json`, `logs/kill_switch_events.log`, `logs/drift_events.log`, `logs/symbol_scan_results.csv`._
