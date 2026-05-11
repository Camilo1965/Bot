"""
scripts/download_mt5_history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch-download historical OHLCV candles from MetaTrader 5 and persist
them to TimescaleDB for model retraining.

Usage (on the EC2 / Windows host with MT5 running):
    python scripts/download_mt5_history.py --symbols BTC/USDT,ETH/USDT --days 90 --timeframe 15m

Requirements:
    - MetaTrader5 terminal running and logged in
    - TimescaleDB container up (docker compose up -d db)
    - .env configured with MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, DB_*
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import utils.env_bootstrap  # noqa: E402
from utils.env_bootstrap import load_env_file

try:
    load_env_file(_ROOT / ".env")
except Exception:
    pass

logger = logging.getLogger("clawdbot.download_history")

# ── MT5 imports (Windows only) ────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5  # type: ignore[import-untyped]
    import pandas as pd  # type: ignore[import-untyped]

    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False

# ── DB import ─────────────────────────────────────────────────────────────────
from database.db_manager import DatabaseManager, db  # noqa: E402

# ── Timeframe mapping ─────────────────────────────────────────────────────────
_TIMEFRAME_MAP: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 16385,
    "4h": 16388,
    "1d": 16408,
}


def _mt5_tf(tf_str: str) -> int:
    return _TIMEFRAME_MAP.get(tf_str.lower(), 15)


def _resolve_symbol(symbol: str) -> str | None:
    """Map internal symbol to MT5 Market-Watch name."""
    from execution.mt5_executor import SYMBOL_MAP
    normal = symbol.replace("/", "").upper()
    # Try direct mapping first
    if symbol in SYMBOL_MAP:
        return SYMBOL_MAP[symbol]
    # Try without slash
    if normal in SYMBOL_MAP:
        return SYMBOL_MAP[normal]
    # Fallback: try to find a key that ends with the base asset
    for k, v in SYMBOL_MAP.items():
        if k.replace("/", "").upper() == normal:
            return v
    logger.error("No MT5 symbol mapping found for %s", symbol)
    return None


def _init_mt5() -> bool:
    """Initialize MT5 from env vars."""
    if not _MT5_AVAILABLE:
        logger.error("MetaTrader5 library not installed. Run: pip install MetaTrader5")
        return False

    account_raw = os.environ.get("MT5_LOGIN", "").strip()
    password = os.environ.get("MT5_PASSWORD", "").strip()
    server = os.environ.get("MT5_SERVER", "").strip()

    if not account_raw or not password or not server:
        logger.error("MT5 credentials missing from .env (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER).")
        return False

    try:
        account = int(account_raw)
    except ValueError:
        logger.error("MT5_LOGIN must be an integer account number.")
        return False

    if not mt5.initialize(login=account, password=password, server=server):
        err = mt5.last_error()
        logger.error("mt5.initialize() failed: %s", err)
        return False

    info = mt5.account_info()
    if info is None:
        logger.error("MT5 initialized but account_info() returned None.")
        mt5.shutdown()
        return False

    logger.info(
        "[MT5] Connected to %s | Account: %d | Balance: %.2f %s",
        server,
        info.login,
        info.balance,
        info.currency,
    )
    return True


def _fetch_ohlcv_sync(mt5_sym: str, tf: int, count: int) -> pd.DataFrame | None:
    """Synchronous wrapper around mt5.copy_rates_from_pos."""
    if mt5 is None:
        return None
    rates = mt5.copy_rates_from_pos(mt5_sym, tf, 0, count)
    if rates is None or len(rates) == 0:
        logger.warning("MT5 returned no rates for %s (tf=%d, count=%d).", mt5_sym, tf, count)
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


async def _download_and_insert(
    symbol: str,
    days: int,
    tf_str: str,
) -> int:
    """Download candles for *symbol* and insert them into the DB.

    Returns the number of rows inserted.
    """
    mt5_sym = _resolve_symbol(symbol)
    if mt5_sym is None:
        return 0

    tf = _mt5_tf(tf_str)
    bars_per_day = {
        1: 1440, 5: 288, 15: 96, 30: 48,
        16385: 24, 16388: 6, 16408: 1,
    }.get(tf, 96)
    count = days * bars_per_day

    logger.info("Downloading %d %s candles for %s (%s)...", count, tf_str.upper(), symbol, mt5_sym)
    df = await asyncio.get_event_loop().run_in_executor(
        None, _fetch_ohlcv_sync, mt5_sym, tf, count
    )
    if df is None or df.empty:
        return 0

    inserted = 0
    for _, row in df.iterrows():
        try:
            ts = row["time"]
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            await db.insert_market_data(
                bucket=ts,
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("tick_volume", 0) or row.get("real_volume", 0) or 0),
            )
            inserted += 1
        except Exception as exc:
            logger.debug("Insert failed for %s @ %s: %s", symbol, row.get("time"), exc)

    logger.info("Inserted %d/%d rows for %s.", inserted, len(df), symbol)
    return inserted


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Download MT5 historical candles to TimescaleDB")
    parser.add_argument(
        "--symbols",
        default=os.environ.get("WATCHLIST", "BTC/USDT,ETH/USDT"),
        help="Comma-separated list of symbols",
    )
    parser.add_argument("--days", type=int, default=90, help="Days of history to download")
    parser.add_argument("--timeframe", default="15m", help="Candle timeframe (1m,5m,15m,30m,1h,4h,1d)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without inserting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols provided.")
        return 1

    # ── Init MT5 ──────────────────────────────────────────────────────────
    if not _init_mt5():
        return 1

    try:
        # ── Init DB ───────────────────────────────────────────────────────
        try:
            await db.connect()
            await db.initialize()
            logger.info("TimescaleDB connected and initialized.")
        except Exception as exc:
            logger.error("Failed to connect to TimescaleDB: %s", exc)
            return 1

        total_inserted = 0
        for symbol in symbols:
            if args.dry_run:
                mt5_sym = _resolve_symbol(symbol)
                tf = _mt5_tf(args.timeframe)
                bars_per_day = {1: 1440, 5: 288, 15: 96, 30: 48, 16385: 24, 16388: 6, 16408: 1}.get(tf, 96)
                count = args.days * bars_per_day
                logger.info("[DRY-RUN] Would download %d %s candles for %s → %s", count, args.timeframe.upper(), symbol, mt5_sym)
                continue

            n = await _download_and_insert(symbol, args.days, args.timeframe)
            total_inserted += n

        logger.info("=" * 60)
        if args.dry_run:
            logger.info("DRY RUN complete. No data was inserted.")
        else:
            logger.info("TOTAL rows inserted: %d", total_inserted)
        logger.info("=" * 60)

    finally:
        await db.disconnect()
        if _MT5_AVAILABLE:
            mt5.shutdown()
            logger.info("[MT5] Disconnected.")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
