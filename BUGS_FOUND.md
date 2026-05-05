# BUGS_FOUND

## B1 — Ghost loop during shutdown (CRITICAL)
- **Root cause:** Sync/reconcile paths could continue acting while MT5/DB teardown was in progress; ghost reconcile could still run against disconnected terminal.
- **Code locations:** `execution/mt5_executor.py` (`begin_shutdown`, `_mt5_terminal_connected`, `sync_positions_with_exchange`, `_reconcile_stale_tickets`, `_reconcile_ghost_position`), `main.py` (shutdown order in `finally`).
- **Fix applied:**
  - Added `self._shutting_down` + `begin_shutdown()`.
  - Sync paths return early when shutdown flag active or MT5 not connected.
  - Main shutdown order now: signal executor shutdown ? cancel/wait loops ? stop TUI ? close DB ? shutdown MT5.
- **Risk removed:** repeated ghost close attempts after broker disconnect.

## B2 — False ghost closures from transient MT5 read failures (CRITICAL)
- **Root cause:** single miss (`positions_get(ticket)==[]` or symbol absent once) could trigger immediate ghost reconcile.
- **Code locations:** `execution/mt5_executor.py` (`_reconcile_stale_tickets`, `sync_positions_with_exchange`).
- **Fix applied:**
  - Added miss counter (`_ghost_missing_counts`) and threshold (`MT5_GHOST_MIN_CONFIRMATIONS`, default 3).
  - Ghost close only after N consecutive confirmations.
  - Reset counter when ticket/symbol is visible again.

## B3 — Duplicate/open-journal corruption risk (CRITICAL)
- **Root cause:** journal writes were append-only without idempotency guard; adoption and open paths could emit overlapping BUY entries.
- **Code locations:** `execution/paper_executor.py` (`record_trade`), `execution/mt5_executor.py` (BUY/SELL journal call sites), `sync_positions_with_exchange` untracked adoption logic.
- **Fix applied:**
  - Added `idempotency_key` to `record_trade` + in-memory dedup set + lock.
  - Added safe quantity validation (`BUY` with qty<=0 skipped).
  - Added fsync after write.
  - Adoption now excludes `_pending_symbols` to prevent race-open + adopt duplication.

## B4 — Initial SL may remain unsynced on broker (HIGH)
- **Root cause:** open path assumed broker SL always persisted from BUY request; missing explicit verify/repair.
- **Code locations:** `execution/mt5_executor.py` (`try_open_trade`, `_verify_initial_sl_synced`).
- **Fix applied:**
  - After BUY + ticket resolution, bot verifies broker SL and force-modifies up to 3 attempts.
  - Updates `last_broker_sl_synced` and `last_mt5_modify_mono` on success.

## B5 — Broker symbol mismatch (`BTCUSD-T` not present) (HIGH)
- **Root cause:** static map could fail on broker variants.
- **Code locations:** `execution/mt5_executor.py` (`validate_symbol_mapping`), `main.py` startup after executor creation.
- **Fix applied:**
  - Added startup symbol map audit with auto-variant probe (`-T` removed, `m`, `.r`).
  - On unresolved symbol mapping, startup aborts safely with critical Telegram alert.

## B6 — “paper + mt5 both running” confusion (HIGH)
- **Root cause:** inherited methods log under `execution.paper_executor` even when object is `MT5Executor`; looked like dual executor.
- **Code locations:** `main.py` executor selection block.
- **Fix applied:**
  - Added hard runtime assertions ensuring only MT5Executor in `EXECUTION_MODE=mt5`, and no MT5Executor in paper mode.

## B7 — TP/trailing hint absent when peak==entry (MEDIUM)
- **Root cause:** `compute_dynamic_tp_hint()` returned `None` until peak moved above entry.
- **Code locations:** `execution/paper_executor.py` (`compute_dynamic_tp_hint`).
- **Fix applied:** returns conservative TP baseline even before first new peak.

## B8 — Missing operational ENV defaults (MEDIUM)
- **Root cause:** optional operational envs absent produced implicit behavior.
- **Code locations:** `main.py` (`_apply_safe_env_defaults`).
- **Fix applied:** default/env warning for `DIAGNOSTIC_BUNDLE_INTERVAL_S`, `TELEGRAM_LOG_ALERTS`, `TELEGRAM_LOG_MIN_LEVEL`, explicit warning for model default when `BUY_PROB_THRESHOLD` absent.

## Validation
- Tests: `42 passed` (`pytest tests/ --ignore=tests/manual_order_test.py`).
- Added tests: `tests/test_mt5_sync_guards.py` (ghost threshold + symbol auto-remap).
