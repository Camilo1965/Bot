# Operations tracker

## `utils/operations_tracker.py`

- **Propósito:** línea JSON por evento en `logs/trade_events.jsonl` (auditoría post-mortem).
- **No reemplaza:** TimescaleDB ni `sync_positions_with_exchange`; es complementario.

## Uso

```python
from utils.operations_tracker import OperationsTracker, TradeState

t = OperationsTracker()
t.log_event("tid-1", "ETH/USDT", TradeState.ORDER_SENT, 3200.0, retcode=10009)
```

## Reconciliación real

Seguir usando `MT5Executor.sync_positions_with_exchange()` y tablas en `database/db_manager.py` para fuente de verdad.
