# GHOST_SYNC_HISTORY

## Source evidence
User-provided forensic evidence (2026-05-01..2026-05-05) shows repeated `GHOST_SYNC` closures followed by re-entry on same symbols.

## Pattern extracted from evidence
- Single MT5 read failures / reconnect windows were interpreted as definitive absence.
- Local book closed as `GHOST_SYNC`.
- Strategy then reopened on next BUY signal.
- Result: avoidable realized losses and state churn.

## Current local repository snapshot
- `logs/trade_journal.csv` in this workspace currently shows `GHOST_SYNC` count = 0.
- Historical log bundle provided by user remains authoritative for the incident timeline.

## Preventive controls implemented
1. **N-confirmation ghost policy** (`MT5_GHOST_MIN_CONFIRMATIONS`, default 3).
2. **Shutdown/disconnect guard** (skip ghost actions when terminal disconnected or bot shutting down).
3. **Pending-aware adoption filter** (do not adopt broker positions while symbol is in `_pending_symbols`).
4. **Journal idempotency keys** to reduce duplicate event rows from overlapping paths.

## Suggested post-deploy check
- Run one full live session and verify:
  - no `GHOST_SYNC` lines unless position truly absent for >= N confirmations,
  - no burst of repeated ghost logs during process stop,
  - no duplicate BUY rows with same ticket idempotency key.
