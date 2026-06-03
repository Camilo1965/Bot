# ClawdBot Roadmap — 2026-06

Plan de evolución posterior a la rebuild de rentabilidad del 2026-06-02. Cubre los 24 ítems identificados, organizados por fase, con scope, archivos afectados, dependencias, criterios de verificación y estimación de esfuerzo.

Estado base al inicio del roadmap:
- `SYMBOL_CONFIG` tuneado vía disk-loaded backtest (BTC/ETH/SOL/DOGE/NEAR/ATOM/LINK activos, XRP demoted).
- Portfolio backtest 30d disk-loaded: +82.56% sumado, +22.74% sobre los 5 originales (delta vs baseline +20.24 pp).
- `scripts/backtest_disk_loaded.py`, `scripts/refit_calibrations.py`, `scripts/symbol_scan.py` operativos.
- `param_sweep.py` con `--mode {inline,disk}` y 8 símbolos cubiertos.

Convenciones del roadmap:
- **Effort**: S (≤2h) · M (≤1d) · L (1-3d) · XL (1 semana+).
- **Bloquea / Bloqueado por**: identificadores `[I-NN]` referenciados entre ítems.
- **Verificación**: comando exacto + criterio de paso.

---

## Fase 0 — Proteger trabajo y reproducibilidad (1-2 días)

### [I-24] Commit + PR del trabajo actual
**Por qué:** 19 archivos modificados + 18 untracked. Riesgo de pérdida.
**Scope:**
- Branch `feat/profitable-rebuild-2026-06`.
- Splits: (a) refactor `simulate_bot`, (b) nuevos scripts, (c) `SYMBOL_CONFIG` update, (d) `.env` WATCHLIST.
- Excluir secretos: `.env` ya tracked → confirmar que MT5 credentials no se exponen (revisar git history del .env).
- Snapshot artefactos: `logs/symbol_config_baseline.json`, `logs/final_verification_report.json`, `logs/param_sweep_summary_disk.json`.
**Archivos:** todos los modificados.
**Effort:** S.
**Verificación:** `git status` limpio; PR abierto con descripción y métricas del antes/después.
**Bloquea:** nada (paralelo a todo).

### [I-13.cron] Retrain semanal automatizado
**Por qué:** modelos envejecen rápido en cripto; calibración drift.
**Scope:**
- Windows Task Scheduler que corra todos los domingos 00:05 UTC:
  ```
  python scripts/retrain_all.py
  python scripts/refit_calibrations.py --all
  python scripts/backtest_disk_loaded.py --report --all --days 30
  ```
- Output a `logs/weekly_retrain_{YYYY-MM-DD}.log`.
- Pre-flight: verificar que disk space > 500 MB libre.
- Post-flight: comparar nuevo `final_verification_report.json` vs anterior; si portfolio composite cae >30% abortar promote y notificar.
- Versionar modelos: copiar `models/*_v2.json` a `models/archive/{YYYY-MM-DD}/` antes del retrain.
**Archivos:** `scripts/retrain_all.py` (existente, validar), nuevo `scripts/weekly_retrain.ps1`, nuevo `scripts/compare_retrain_diff.py`.
**Effort:** M.
**Verificación:** dry-run manual de `weekly_retrain.ps1`; archivo log generado; carpeta archive con timestamp.
**Bloqueado por:** [I-24].

---

## Fase 1 — Honestidad del backtest (CRÍTICO antes de capital real)

### [I-1] Walk-forward true OOS para nuevos símbolos
**Por qué:** NEAR/ATOM/LINK retrenados 2026-06-02 sobre 180d → backtest 30d es in-sample. PnL backtest +14% a +40% probablemente cae a +5% a +15% en OOS real.
**Scope:**
- Refetch 240d CCXT para cada uno de NEAR, ATOM, LINK (también BTC/ETH/SOL/DOGE para comparabilidad).
- Walk-forward 5 folds: cortar últimos 30d del training, reentrenar v2, refit calibration, backtestear en esos 30d held-out.
- Repetir para 4 ventanas adicionales rolling de 30d → 5 puntos OOS por símbolo.
- Métrica: mean PnL ± std, IC 95% via bootstrap (1000 resamples).
**Archivos:** nuevo `scripts/walk_forward_oos.py` reutilizando `scripts.retrain_model`, `scripts.refit_calibrations.refit_one`, `scripts.backtest_disk_loaded.run_symbol`.
**Effort:** L (compute pesado: 7 símbolos × 5 folds × retrain).
**Verificación:** `logs/walk_forward_oos_report.json` por símbolo: mean_pnl, std_pnl, IC_low, IC_high, mean_pf, mean_wr; comparar IC_low vs backtest in-sample.
**Bloquea:** [I-4] (shadow run usa este IC como expectativa).

### [I-2] Portfolio simulator (un balance, capital compartido)
**Por qué:** backtest actual corre per-symbol con $10K aislados. Real bot tiene UN balance + `MAX_POSITIONS=2`. Resultados serán 50-70% del agregado.
**Scope:**
- Nuevo loop temporal cross-symbol: ordenar todas las velas de los 7 símbolos por timestamp, simular cada paso del tiempo evaluando todos los símbolos a la vez.
- Manejar concurrencia: máximo `MAX_POSITIONS` abiertas simultáneamente; signal queue prioritized por `calibrated_prob - threshold` margin.
- Capital: trade size = `balance × risk_per_trade / sl_frac`, descontado por posiciones abiertas.
- Compounding: balance actualiza por close de trade; PnL del siguiente trade usa nuevo balance.
- Métricas extra: peak concurrent positions, signals dropped por cap, capital utilization avg.
**Archivos:** nuevo `scripts/portfolio_backtest.py` reutilizando `_run_simulation_loop` adaptado para multi-symbol state.
**Effort:** L.
**Verificación:** `logs/portfolio_backtest_30d.json` muestra: portfolio_pnl_pct, max_dd_pct, signals_total, signals_executed, signals_dropped, peak_concurrent. PnL portfolio debe ser <= sum independent (sanity).
**Bloquea:** [I-4].

### [I-3] Slippage + spread realista en simulación
**Por qué:** backtest fill perfecto al close inflación de PnL ~10-25%.
**Scope:**
- En `_run_simulation_loop` al abrir: `entry_fill = entry_price × (1 + spread_bps/10000 + slippage_atr_mult × atr_pct)`.
- Cierre SL: `exit_fill = sl_price × (1 - spread_bps/10000)` (slippage adverso al stop).
- Cierre TP: similar pero menos adverso (liquidez en limit orders).
- Parámetros por símbolo:
  - BTC/ETH: spread_bps=2, slippage_atr_mult=0.05
  - SOL/DOGE/LINK: spread_bps=5, slippage_atr_mult=0.10
  - NEAR/ATOM: spread_bps=8, slippage_atr_mult=0.15
- Configurable vía nueva sección en `SYMBOL_CONFIG`: `exec_costs: {spread_bps, slippage_mult}`.
**Archivos:** `scripts/backtest_full_bot.py` (función `_run_simulation_loop`), `strategy/ml_predictor.py` (`SYMBOL_CONFIG`).
**Effort:** M.
**Verificación:** re-correr Phase F; PnL debe caer un 10-25%; ratio TP/SL hits debe disminuir levemente; Sharpe baja pero PF más estable.
**Bloqueado por:** [I-2] (portfolio sim).
**Bloquea:** [I-4].

### [I-4] Shadow paper run 30d
**Por qué:** gate obligatorio antes de capital real. Compara live execution vs backtest predictions.
**Scope:**
- Arrancar bot con `EXECUTION_MODE=paper` durante 30d.
- Logs comparativos: para cada signal emitido, registrar prob_cal, prob_threshold, regime_prob, htf_filter_pass, vol_filter_pass.
- Diario: agrupar (símbolo, día) → trades, WR, PnL.
- Comparar contra `backtest_disk_loaded --days 30` corrido mismo día.
- Métrica clave: signal_count_drift = |live_trades - backtest_trades| / backtest_trades. Aceptable <20%.
- Si drift > 30% en cualquier símbolo → investigar (probable bug en feature pipeline o data feed mismatch).
**Archivos:** nuevo `scripts/shadow_run_compare.py`, `bot/signal_emitter.py` (añadir telemetría más rica), nuevo `logs/shadow_run_daily.csv`.
**Effort:** XL (calendario, no horas de coding).
**Verificación:** después de 30d: PnL paper dentro del IC_95% de [I-1]; signal_count_drift <20% en >=5/7 símbolos.
**Bloqueado por:** [I-1], [I-2], [I-3].
**Bloquea:** activación de capital real.

---

## Fase 2 — Doblar oportunidad: SHORT models + VIBE + features

### [I-5] Modelos SHORT por símbolo
**Por qué:** hoy solo LONG. En cripto bajista pierdes 50% de oportunidad.
**Scope:**
- Reusar `strategy.ml_predictor.short_model_json_path_for_symbol` (existe, return None hoy).
- Label SHORT: `forward_return_label(close, horizon, -DEFAULT_LABEL_ROUND_TRIP)` → invertido. Mejor: triple_barrier inverse.
- Entrenar `{SYMBOL}_short_v2.json` con mismas 26 features.
- En `signal_emitter`: lógica dual — si long_model_prob >= long_threshold AND > short_model_prob: BUY; si short_model_prob >= short_threshold AND > long_model_prob: SELL.
- Risk manager: gestionar posición short (sl arriba, tp abajo) — `execution/paper_executor.py` puede requerir refactor menor.
- Calibración separada para short: `{SYMBOL}_short_calibration.json`.
**Archivos:** `scripts/retrain_model.py` (flag `--side {long,short,both}`), `scripts/refit_calibrations.py`, `bot/signal_emitter.py`, `execution/paper_executor.py`, `strategy/ml_predictor.py`.
**Effort:** XL.
**Verificación:** Phase F-equivalent con modo dual side → portfolio PnL debe subir +15-30% (gana en regímenes bajistas que actualmente pierdes por inactividad).
**Bloqueado por:** [I-3].

### [I-7] VIBE features (v3 models, 28 features)
**Por qué:** 4 features extra disponibles (`vibe_pattern_score`, `vibe_factor_ic`, `vibe_journal_health`, `vibe_backtest_sharpe`). Hoy desactivadas (`VIBE_FEATURES_ENABLED=0`).
**Scope:**
- Verificar pipeline `vibe.feature_bridge.extract_vibe_features` — testing end-to-end con MCP availability check.
- Reentrenar v3 para los 7 símbolos: `python scripts/retrain_model.py --symbols ... --vibe --days 180`.
- Activar flag `VIBE_FEATURES_ENABLED=1` en `.env`.
- A/B test: comparar Phase F 30d con v2 vs v3 (delta AUC, f1, portfolio PnL).
- Promote v3 solo si f1 mejora >10% y portfolio PnL mejora >5pp.
- Si VIBE MCP no disponible en live → fallback a v2 via `VIBE_FEATURE_NEUTRAL` (ya implementado en `vibe.feature_bridge`).
**Archivos:** `.env`, modelos v3, `bot/signal_emitter.py` (telemetría VIBE).
**Effort:** L.
**Verificación:** logs/vibe_v2_vs_v3_compare.json con métricas A/B; decisión promote/reject por símbolo.

### [I-8] Multi-timeframe stacking (1h/4h features)
**Por qué:** modelo 15m ve solo ruido de microestructura. Features 1h/4h dan contexto trend macro.
**Scope:**
- Añadir a `strategy.quant_features.add_quant_features`:
  - Refetchar 1h y 4h del mismo símbolo cuando se computa features.
  - Calcular `rsi_1h`, `macd_hist_1h`, `atr_1h`, `bb_width_1h`, `rsi_4h`, `macd_hist_4h`, `adx_4h` y forward-fill al timeframe base.
  - Total features: 26 base + 7 multi-TF = 33.
- Refactor flujo fetch: hoy `fetch_ohlcv_ccxt` retorna un solo TF. Crear `fetch_multi_tf(symbol, tfs=["15m","1h","4h"])`.
- Reentrenar todos los símbolos con feature set ampliado.
- Calibrar.
- A/B comparar.
**Archivos:** `strategy/quant_features.py`, `scripts/deep_strategy_audit.py` (o nuevo `data/multi_tf_fetcher.py`), `scripts/retrain_model.py`.
**Effort:** XL.
**Verificación:** AUC v2 vs v_mtf por símbolo; promote si AUC mejora >0.03.

### [I-9] Triple-barrier labels vs forward_return
**Por qué:** `forward_return_label` solo mira fwd_return > threshold. No considera que SL pudo gatillarse antes. Triple barrier es más realista (TP/SL race).
**Scope:**
- `strategy.quant_features.triple_barrier_label` ya existe.
- Modificar `scripts/retrain_model.py` para aceptar `--label {forward_return, triple_barrier}`.
- En triple_barrier: usar `fixed_sl_pct` y `fixed_tp_pct` del `SYMBOL_CONFIG`.
- Reentrenar y comparar.
**Archivos:** `scripts/retrain_model.py`.
**Effort:** M.
**Verificación:** A/B AUC + portfolio PnL Phase F.

### [I-10] Ensemble XGB + LGBM + LogReg
**Por qué:** XGB solo → varianza alta. Ensemble reduce overfit.
**Scope:**
- Añadir LightGBM y LogReg como models adicionales en `train_direction_model`.
- Predicción: promedio simple de las 3 probas (o stacking con meta-learner LR sobre las probas).
- Save format: `{SYMBOL}_ensemble.json` con paths a los 3 sub-models.
- Calibration over ensemble output.
**Archivos:** nuevo `strategy/ensemble.py`, `scripts/retrain_model.py`, `strategy/ml_predictor.py` (load_ensemble).
**Effort:** L.
**Verificación:** AUC ensemble vs XGB solo; PF estabilidad cross-fold; decisión por símbolo.

---

## Fase 3 — Riesgo, kill-switch, alertas

### [I-11] Kill-switch automático
**Por qué:** archivo `risk/kill_switch.py` untracked. Implementarlo y conectarlo al loop.
**Scope:**
- Triggers (todos activos):
  - Daily PnL <= -5% del balance → freeze 24h.
  - 3 losses consecutivas (cross-symbol) → pause 4h.
  - Drawdown 7d >= 15% → close all + pause 48h.
  - Cualquier símbolo con WR rolling 20-trade <30% → demote ese símbolo (set prob_threshold=0.95 dinámicamente).
- Estado persistido en `data/kill_switch_state.json` (resilience a restart).
- Hook en `bot/signal_emitter.py:single_symbol_eval` antes de emitir signal: si killswitch active → return HOLD con razón.
- Hook en `execution/paper_executor`: si killswitch flag close_all → cerrar todas las posiciones a mercado.
- Logs: cada activación a `logs/kill_switch_events.log` con timestamp + razón.
**Archivos:** `risk/kill_switch.py`, `bot/signal_emitter.py`, `execution/paper_executor.py`, nuevo `data/kill_switch_state.json`.
**Effort:** L.
**Verificación:** test unitario simulando 3 losses consecutivos; bot pausa correctamente.

### [I-12] MAX_PORTFOLIO_RISK_PCT enforcement audit
**Por qué:** existe en `risk/risk_manager.py:274` pero hay que verificar que efectivamente bloquea cross-symbol.
**Scope:**
- Audit: cuando el signal_emitter pide sizing a `calculate_position_size`, el manager debe sumar `risk_usd` de todas las posiciones abiertas y rechazar si total + nuevo > MAX_PORTFOLIO_RISK_PCT × balance.
- Hoy `state.open_positions` es per-symbol en backtest, no global. Verificar live state tracking.
- Test: simular abrir 2 posiciones máximas (cada 5% risk) + intentar abrir 3ra → debe rechazar.
**Archivos:** `risk/risk_manager.py`, `bot/signal_emitter.py`.
**Effort:** M.
**Verificación:** test integración + audit log de bloqueos.

### [I-14] Drift detector (live raw_prob vs backtest)
**Por qué:** si distribución live de raw probs cambia vs backtest → modelo en data drift → desactivar antes de perder.
**Scope:**
- Cron horario (o post-trade): tomar últimos N=200 raw_probs predichas live.
- Comparar con `models/{SYMBOL}_v2.json` backtest reference distribution (snapshot al retrain).
- KS-test 2-sample. Si p < 0.01 → ALERT + auto-pausa entries de ese símbolo por 24h.
- Snapshot ref distribution: durante `retrain_model.py` guardar `models/{SYMBOL}_prob_ref.json` con los raw_probs OOS.
**Archivos:** nuevo `risk/drift_detector.py`, `scripts/retrain_model.py`, `bot/signal_emitter.py`.
**Effort:** M.
**Verificación:** ejecutar con prob distribution artificialmente desplazada → detecta drift y pausa.

### [I-15] Alertas Telegram/Discord
**Por qué:** operador debe saber en tiempo real qué pasa.
**Scope:**
- Webhook Telegram bot (más simple) o Discord webhook.
- Eventos a notificar:
  - Trade abierto: símbolo, lado, entry, SL, TP, size, prob, balance restante.
  - Trade cerrado: razón (SL/TP/TTL), PnL, WR rolling 7d.
  - Daily PnL report a las 23:55 UTC.
  - Kill-switch activado (CRÍTICO).
  - Drift detectado.
  - Retrain semanal completado con métricas.
- Implementación: nuevo `vibe/notifier.py` con interfaz `notify(level, msg, attachments=None)`.
- Config `.env`: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` o `DISCORD_WEBHOOK_URL`.
- Async, non-blocking (no detener trade loop si falla red).
**Archivos:** nuevo `vibe/notifier.py`, hooks en `execution/paper_executor.py`, `risk/kill_switch.py`, `risk/drift_detector.py`.
**Effort:** M.
**Verificación:** smoke test enviando mensaje a chat.

---

## Fase 4 — Datos + pipeline

### [I-6] XRP reentrenamiento o entierro definitivo
**Por qué:** modelo actual perdió alfa. Decidir si vale rebuild o demote permanente.
**Scope:**
- Opción A (rebuild): probar feature set diferente (multi-TF de [I-8] + VIBE de [I-7]). Si AUC >0.62 y backtest 30d positivo → reactivar.
- Opción B (entierro): documentar en memory file, remover de `SYMBOL_CONFIG`, retirar `models/XRP_USDT_*.json` a `models/archive/`.
- Decisión empírica: rebuild una vez con [I-8] disponible.
**Archivos:** `strategy/ml_predictor.py`, `scripts/backtest_full_bot.py`, models/.
**Effort:** S (decisión) + M (rebuild si se intenta).
**Verificación:** o reactivado con AUC documentado o archivado con razón.

### [I-16] TimescaleDB conexión arreglada
**Por qué:** `WinError 1225` en cada retrain → cae a CCXT (más lento, depende de internet).
**Scope:**
- Diagnosticar: ¿Postgres corre? `Get-Service | findstr postgres`. ¿Puerto 5432 abierto? `netstat -an | findstr 5432`.
- Si no instalado: usar Docker `docker compose up timescaledb` o instalar nativo.
- Schema: tabla `ohlcv(symbol TEXT, ts TIMESTAMPTZ, open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, vol NUMERIC, PRIMARY KEY(symbol, ts))`.
- Hypertable on `ts` (TimescaleDB).
- Backfill: script `scripts/backfill_db.py` que descarga 365d para cada símbolo en watchlist y popula la tabla.
- Job cron horario para mantener actualizado.
**Archivos:** nuevo `data/db_schema.sql`, `scripts/backfill_db.py`, `docker-compose.yml` si Docker.
**Effort:** L.
**Verificación:** `psql -c "SELECT count(*) FROM ohlcv WHERE symbol='BTC/USDT'"` retorna >150_000.

### [I-17] CCXT cache local (parquet/duckdb)
**Por qué:** cada sweep refetchea miles de candles. Ahorrar I/O + tiempo.
**Scope:**
- DuckDB-backed cache: `data/cache/ohlcv.duckdb`.
- Wrapper `fetch_ohlcv_cached(symbol, timeframe, limit, max_age_seconds=300)`:
  - Si cache hit + age < max_age → return cache.
  - Else fetch CCXT, upsert a duckdb, return.
- Migrar todos los callsites de `fetch_ohlcv_ccxt` a este wrapper.
**Archivos:** nuevo `data/ohlcv_cache.py`, refactor en `scripts/backtest_disk_loaded.py`, `scripts/param_sweep.py`, `scripts/symbol_scan.py`, `scripts/refit_calibrations.py`.
**Effort:** M.
**Verificación:** correr `python scripts/param_sweep.py --mode disk` dos veces seguidas; segunda debe ser >5x más rápida.
**Bloqueado por:** opcional [I-16] (si DB ok, cache es menos necesario pero aún útil para hot lookups).

### [I-19] MATIC → POL/USDT retry
**Por qué:** Polygon renombrado. Modelos de Phase D scan fallaron con `insufficient_data:0`.
**Scope:**
- Cambiar candidate list en `scripts/symbol_scan.py`: `MATIC/USDT` → `POL/USDT`.
- Verificar listing en Binance (CCXT).
- Re-correr scan.
- Si survivor: retrain + refit + add to SYMBOL_CONFIG.
**Archivos:** `scripts/symbol_scan.py`.
**Effort:** S.
**Verificación:** `scripts/symbol_scan.py` produce row para POL/USDT.

### [I-20] Phase D extended scan (narrativa 2026)
**Por qué:** descubrir nuevas oportunidades. ARB/OP/SUI/SEI/INJ/TIA/JTO son hot.
**Scope:**
- Añadir a `CANDIDATES` de `scripts/symbol_scan.py`: ARB, OP, SUI, SEI, INJ, TIA, JTO, RUNE, KAS.
- Correr scan completo.
- Survivors: retrain + refit + disk sweep + Phase F validation.
- Promote a `SYMBOL_CONFIG` si pasan criterios.
**Archivos:** `scripts/symbol_scan.py`.
**Effort:** M (compute) + S (promotion).
**Verificación:** ≥2 nuevos survivors con Phase F PnL > +5% en 30d disk-loaded.

### [I-21] Cross-pair (BTC/ETH ratio) estrategia hedged
**Por qué:** diversificación tipo market-neutral. Independiente del mercado direccional.
**Scope:**
- Computar series ratio close_BTC / close_ETH a TF compartido (15m).
- Test cointegración (Engle-Granger) → si I(0) → válido para mean-reversion.
- Estrategia: si ratio Z-score > 2 → short BTC + long ETH (size matched); si Z < -2 → opuesto. Cierre cuando |Z| < 0.5.
- Backtest separado: `scripts/pair_trade_backtest.py`.
- Solo entrar si cointegration p-value < 0.05 en último 90d (re-test semanal).
**Archivos:** nuevo `strategy/pair_trader.py`, `scripts/pair_trade_backtest.py`.
**Effort:** XL.
**Verificación:** backtest 90d con PF >1.3 y correlación con BTC < 0.2.

---

## Fase 5 — Operación: journal, reportes, dashboard

### [I-18] Backtest report HTML con equity curves
**Por qué:** CSV/JSON no comunica. Necesitas vista visual para presentar.
**Scope:**
- Plotly o ECharts con equity curve por símbolo + portfolio agregado.
- Per-trade timeline con entry/exit markers.
- Tablas: per-symbol metrics, monthly breakdown.
- Output: `logs/reports/{YYYY-MM-DD}/backtest_report.html` standalone.
- Generador: `scripts/generate_html_report.py`.
**Archivos:** nuevo `scripts/generate_html_report.py`, template Jinja2.
**Effort:** M.
**Verificación:** abrir HTML → 1 equity curve, 1 trade timeline, ≥3 tablas de métricas.

### [I-23] Trade journal estructurado + VIBE analytics
**Por qué:** `vibe/journal_analyzer.py` existe pero no veo write hook. Sin journal → sin analytics behaviour.
**Scope:**
- En `execution/paper_executor.py` post-trade close → escribir fila a `data/trade_journal.csv`:
  - timestamp, symbol, side, entry_time, exit_time, entry_price, exit_price, sl_price, tp_price, size_usd, pnl_usd, pnl_pct, close_reason, ml_confidence, regime_prob, htf_filter, vol_filter, hour_open, dow, balance_before, balance_after, killswitch_state.
- Schema versionado en header.
- VIBE analyzer corre cada 24h sobre el journal (ya implementado, conectarlo en `bot/main.py` scheduler).
- Output VIBE insights a `data/vibe_analysis_{YYYY-MM-DD}.json`.
**Archivos:** `execution/paper_executor.py`, `bot/main.py`, `vibe/journal_analyzer.py` (verificar I/O).
**Effort:** M.
**Verificación:** después de 1 trade → fila en journal.csv; tras 5 trades → vibe analysis genera output.

### [I-22] **Dashboard web (super detallado en docs/DASHBOARD_DESIGN.md)**
**Por qué:** operador necesita visibilidad en tiempo real, no leer logs.
**Scope:** ver `docs/DASHBOARD_DESIGN.md` para spec completo de UX, componentes, color tokens, mockups, tech stack y plan de implementación por sprints.
**Effort:** XL (3-5 sprints).
**Verificación:** ver checkpoints en design doc.

---

## Resumen prioridades y secuencia recomendada

| Semana | Foco | Ítems |
|--------|------|-------|
| 1 | Proteger + honestidad backtest | I-24, I-13.cron, I-3, I-2 |
| 2 | OOS validation arranca | I-1 (compute), I-11, I-15 |
| 3-6 | Shadow paper run | I-4 (calendar) + I-22 (sprint 1-2 dashboard) |
| 4-5 | Mejora modelo | I-5, I-7, I-9 (paralelos) |
| 6-7 | Dashboard sprints 3-5 | I-22 |
| 7-8 | Cross-pair + multi-tf | I-8, I-21 |
| ongoing | Datos infra | I-16, I-17, I-19, I-20 |

## Métricas de éxito (8 semanas)

- Portfolio paper PnL real 30d (post Phase F) >= +15% en $100 (vs +22.7% backtest, descontando OOS shrinkage).
- Dashboard operacional con ≥5 vistas funcionales.
- Modelos SHORT activos en >=3 símbolos.
- Kill-switch ha bloqueado >=1 escenario adverso sin intervención manual.
- 0 incidentes de pérdida de datos / capital por bug.

## Notas finales

- Todo cambio en `SYMBOL_CONFIG` debe pasar `scripts/backtest_disk_loaded.py --report --days 30` antes del merge.
- Cada modelo retrained debe trigger `scripts/refit_calibrations.py` para ese símbolo.
- Snapshot baseline `logs/symbol_config_baseline.json` se actualiza solo cuando un Phase F nuevo pasa todos los gates.
