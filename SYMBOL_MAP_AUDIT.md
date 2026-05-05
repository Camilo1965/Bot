# SYMBOL_MAP_AUDIT

## Startup validation path
- `MT5Executor.validate_symbol_mapping(watchlist)` now runs during MT5 startup (`main.py`).
- For each internal symbol (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`):
  1. Check mapped broker symbol in `SYMBOL_MAP`.
  2. If missing on broker, probe variants:
     - remove `-T`
     - replace `-T` with `m`
     - append `.r`
     - append `m`
  3. Auto-remap when variant exists.
  4. Abort startup if unresolved after probes.

## Safety behavior
- Unresolved mappings trigger:
  - critical log
  - critical Telegram alert
  - safe startup abort (no trading loop starts)

## Test coverage
- `tests/test_mt5_sync_guards.py::test_validate_symbol_mapping_auto_heals_variant`

## Operational note
This audit runs only with MT5 available/connected. In environments without MT5, method returns empty unresolved list by design.
