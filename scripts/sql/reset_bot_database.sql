-- Reset datos operativos ClawdBot / TimescaleDB (PostgreSQL 15+ / 16).
-- Ejecutar desde la raíz del repo (Windows PowerShell ejemplo abajo).
--
-- Tablas del código: market_data, ml_predictions, htf_trend (database/db_manager.py).
-- trades_history: analytics CEO/web si existe.
-- trades, daily_stats, positions, logs, balance_history: solo si tu DB legacy las tiene.

BEGIN;

TRUNCATE TABLE market_data, ml_predictions, htf_trend RESTART IDENTITY CASCADE;

DO $$
DECLARE
  t text;
BEGIN
  IF to_regclass('public.trades_history') IS NOT NULL THEN
    EXECUTE 'TRUNCATE TABLE trades_history RESTART IDENTITY CASCADE';
  END IF;

  FOREACH t IN ARRAY ARRAY['trades','daily_stats','positions','logs','balance_history']
  LOOP
    IF to_regclass('public.' || t) IS NOT NULL THEN
      EXECUTE format('TRUNCATE TABLE %I RESTART IDENTITY CASCADE', t);
    END IF;
  END LOOP;
END $$;

COMMIT;
