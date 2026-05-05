# ARCHITECTURE — ClawdBot (resumen ejecución)

```mermaid
flowchart LR
  subgraph main_loop [main / loops]
    MC[market_consumer]
    SE[signal_emitter]
  end
  subgraph exec [execution]
    PE[PaperExecutor / MT5Executor]
    L["_positions_lock asyncio.Lock"]
    OP[(open_positions dict)]
    Pending["_pending_symbols set"]
  end
  subgraph io [I/O]
    MT5[MetaTrader5 API]
    DB[(TimescaleDB asyncpg)]
  end
  MC --> SE
  SE --> PE
  PE --> L
  L --> OP
  L --> Pending
  PE --> MT5
  PE --> DB
```

## Sincronización

| Recurso | Mecanismo |
|---------|-----------|
| Mutación posiciones | `asyncio.Lock` (`_positions_lock`) |
| Conteo vs max | `RiskManager._open_count` + `_pending_symbols` hasta persistencia |
| MT5 ↔ libro local | `sync_positions_with_exchange()` (reconcile ghosts / adopt) |
| Estado disco | `state.json` (paper), TimescaleDB `trades` |

## Threads

Proceso único asyncio (AGENTS.md). MT5 API se llama con `asyncio.to_thread` donde aplica; no hay thread Python dedicado al trading clásico.
