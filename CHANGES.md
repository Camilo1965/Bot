# CHANGES — auditoría 2026-05-04

## Qué cambió

1. **`PaperExecutor`:** `_pending_symbols`, reserva atómica con `register_open()`, sector sobre `open_positions | pending`, rollback en fallos Binance/DB.
2. **`MT5Executor.try_open_trade`:** misma reserva; todos los `return False` tras reserva limpian `pending` + `register_close` bajo `_positions_lock`.
3. **`utils/operations_tracker.py`:** log JSONL opcional (`TradeState`, `OperationsTracker`).
4. **`tests/test_bot_critical.py`:** regresión race + DB bloqueada.

## Por qué

Sin `_pending_symbols`, el hueco tras `await insert_open_trade` / `order_send` permitía violar “un símbolo” y superar límites lógicos bajo concurrencia asyncio.

## Revertir

```bash
git checkout HEAD~1 -- execution/paper_executor.py execution/mt5_executor.py utils/operations_tracker.py tests/test_bot_critical.py BUGS_REPORT.md CHANGES.md ARCHITECTURE.md OPERATIONS_TRACKER.md
```

(Quién quiera mantener solo docs: revert solo código.)
