# SHUTDOWN_FIX

## Old hazard
`shutdown_mt5()` could happen while sync/reconcile code was still executing, causing terminal calls to return `None` and triggering noisy ghost/error cascades.

## New shutdown contract
1. `MT5Executor.begin_shutdown()` sets internal stop flag.
2. Main loop cancels all running tasks and awaits completion (`gather(..., return_exceptions=True)`).
3. Stop Rich live UI.
4. Close DB (`close_db()`).
5. Close MT5 (`shutdown_mt5()`).

## Code points
- `main.py` `finally` block.
- `execution/mt5_executor.py`:
  - `begin_shutdown()`
  - `_mt5_terminal_connected()`
  - guard checks in `sync_positions_with_exchange`, `_reconcile_stale_tickets`, `_reconcile_ghost_position`.

## Why this stops the loop
- Ghost reconcile now no-ops when shutdown flag set.
- Sync now no-ops when terminal/account session is disconnected.
- MT5 disconnection is treated as “skip reconciliation”, not “position vanished”.
