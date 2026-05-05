# BUGS_REPORT — auditoría MT5 / ejecutor

## Crítico — corregido

### C1 — Hueco entre `register_open()` y `open_positions[sym]`

**Síntoma:** Dos `try_open_trade` concurrentes (mismo símbolo o el cuarto slot mientras otro hace `await` a DB/MT5) podían pasar `sym not in open_positions` antes de que el primero insertara la posición.

**Ubicación:** `execution/paper_executor.py` (`try_open_trade`), `execution/mt5_executor.py` (`try_open_trade`).

**Fix:** Conjunto `_pending_symbols`: se añade en el mismo critical section que `register_open()`, se comprueba junto con `open_positions` para duplicados y sector, y se hace `discard` en éxito o en cualquier rollback (`register_close`).

## Alto — conocido / mitigado parcialmente

### A1 — Dashboard vs MT5

**Estado:** La UI Rich y `/api/state` leen estado del proceso (`RiskManager`, `open_positions`). La reconciliación con MT5 existe en `MT5Executor.sync_positions_with_exchange()` (fantasmas / adopción). No hay garantía fija de “&lt;5 s” salvo frecuencia del loop que llame a `sync` y a la API.

**Riesgo residual:** Si el web handler no dispara `sync` y el loop tampoco, el desfase puede superar pocos segundos hasta el próximo `sync_positions_with_exchange`.

### A2 — Órdenes MT5 fuera del magic del bot

**Estado:** Filtrado por `magic` en varios paths (`_find_magic_long_ticket`, adopción). Posiciones manuales mismo símbolo pueden interactuar con lógica de cierre; no se cuentan en `_open_count` salvo sync.

## Medio

### M1 — `register_close()` sin lock en algunos paths antiguos MT5

**Fix aplicado:** Rollbacks MT5 que solo llamaban `register_close()` ahora usan `async with self._positions_lock` junto con `_pending_symbols.discard`.

## Tests añadidos

- `tests/test_bot_critical.py`: bloqueo con DB lenta, máximo con tres opens en vuelo, burst concurrente.
