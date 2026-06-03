-- data/db_schema.sql
-- TimescaleDB OHLCV schema for ClawdBot

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        NUMERIC,
    high        NUMERIC,
    low         NUMERIC,
    close       NUMERIC,
    vol         NUMERIC,
    PRIMARY KEY (symbol, ts)
);

SELECT create_hypertable('ohlcv', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS ohlcv_sym_ts ON ohlcv (symbol, ts DESC);
