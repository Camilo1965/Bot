#!/usr/bin/env python3
"""
MT5 round-trip smoke test using the same helpers as :class:`~execution.mt5_executor.MT5Executor`
(build BUY with SL/optional TP, ``_send_order_with_retry``, ``close_position_by_ticket``).

Requires Windows + MetaTrader 5 terminal logged in. Configure ``.env`` like ``main.py``:

  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
  Optional: MT5_PATH, MT5_MAGIC (default 20240101), MT5_DEVIATION, MT5_SYMBOL_OVERRIDES

Examples::

  python scripts/mt5_smoke_trade.py
  python scripts/mt5_smoke_trade.py --symbol INJ/USDT --tp-pct 0.02
  python scripts/mt5_smoke_trade.py --open-only
  python scripts/mt5_smoke_trade.py close --ticket 123456789

Environment shortcuts::

  MT5_TEST_SYMBOL=DOGE/USDT
  MT5_TEST_RISK_PCT=0.005
  MT5_TEST_SL_PCT=0.015
  MT5_TEST_TP_PCT=0.02
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import utils.env_bootstrap  # noqa: F401  # patches dotenv for UTF-16 .env

from utils.env_bootstrap import load_env_file

load_env_file(_ROOT / ".env")

import MetaTrader5 as mt5  # noqa: E402

from database.db_manager import db, close_db, init_db  # noqa: E402
from execution.mt5_executor import (  # noqa: E402
    MT5Executor,
    calculate_lot_size,
    fetch_mt5_account_balance,
    initialize_mt5,
    shutdown_mt5,
)
from risk.risk_manager import RiskManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("mt5_smoke")


async def _await_bot_position(mt5_sym: str, magic: int, timeout_s: float = 15.0) -> int | None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        positions = mt5.positions_get(symbol=mt5_sym)
        if positions:
            for p in positions:
                if int(p.magic) == int(magic):
                    return int(p.ticket)
        await asyncio.sleep(0.25)
    return None


def _executor_from_env() -> tuple[MT5Executor, float]:
    balance = fetch_mt5_account_balance()
    if balance is None or balance <= 0:
        raise RuntimeError("Could not read MT5 balance (account_info).")

    risk_pct = float(os.environ.get("MT5_TEST_RISK_PCT", "0.005"))
    magic = int(os.environ.get("MT5_MAGIC", "20240101"))
    deviation = int(os.environ.get("MT5_DEVIATION", "20").strip() or "20")

    rm = RiskManager(initial_balance=balance)
    ex = MT5Executor(
        db=db,
        risk_manager=rm,
        live=True,
        risk_pct=risk_pct,
        magic=magic,
        deviation=deviation,
    )
    return ex, balance


async def cmd_close(args: argparse.Namespace) -> int:
    login_raw = os.environ.get("MT5_LOGIN", "").strip()
    password = os.environ.get("MT5_PASSWORD", "").strip()
    server = os.environ.get("MT5_SERVER", "").strip()
    path = os.environ.get("MT5_PATH", "").strip() or None

    if not login_raw or not password or not server:
        logger.error("Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env")
        return 1

    try:
        await init_db()
    except Exception as exc:
        logger.warning("init_db failed (continuing): %s", exc)

    if not initialize_mt5(int(login_raw), password, server, path):
        return 1

    try:
        ex, _ = _executor_from_env()
    except RuntimeError as exc:
        logger.error("%s", exc)
        shutdown_mt5()
        return 1

    ok = await ex.close_position_by_ticket(int(args.ticket))
    shutdown_mt5()
    try:
        await close_db()
    except Exception:
        pass
    return 0 if ok else 3


async def cmd_smoke(args: argparse.Namespace) -> int:
    try:
        await init_db()
    except Exception as exc:
        logger.warning("init_db failed (continuing without DB): %s", exc)

    login_raw = os.environ.get("MT5_LOGIN", "").strip()
    password = os.environ.get("MT5_PASSWORD", "").strip()
    server = os.environ.get("MT5_SERVER", "").strip()
    path = os.environ.get("MT5_PATH", "").strip() or None

    if not login_raw or not password or not server:
        logger.error("Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env")
        return 1

    if not initialize_mt5(int(login_raw), password, server, path):
        return 1

    try:
        ex, balance = _executor_from_env()
    except RuntimeError as exc:
        logger.error("%s", exc)
        shutdown_mt5()
        return 1

    # Override risk_pct from CLI when provided
    risk_pct = args.risk_pct
    ex._risk_pct = risk_pct  # noqa: SLF001 — smoke harness matches user's intent

    sym = args.symbol.strip()
    bad = ex.validate_symbol_mapping([sym])
    if bad:
        logger.error("Symbol mapping failed: %s", bad)
        shutdown_mt5()
        return 1

    mt5_sym = ex._resolve_symbol(sym)  # noqa: SLF001
    if mt5_sym is None:
        logger.error("Could not resolve symbol for %s", sym)
        shutdown_mt5()
        return 1

    if not mt5.symbol_select(mt5_sym, True):
        logger.error("symbol_select(%s) failed — enable symbol in Market Watch.", mt5_sym)
        shutdown_mt5()
        return 1

    tick = mt5.symbol_info_tick(mt5_sym)
    if tick is None or tick.ask <= 0 or tick.bid <= 0:
        logger.error("No tick for %s", mt5_sym)
        shutdown_mt5()
        return 1

    info = mt5.symbol_info(mt5_sym)
    digits = int(info.digits) if info else 5

    ask = float(tick.ask)
    bid = float(tick.bid)

    sl_raw = bid * (1.0 - args.sl_pct)
    sl_price, adjusted = ex._clamp_stop_loss_buy(mt5_sym, sl_raw, ask, tick, digits)  # noqa: SLF001
    if sl_price is None:
        logger.error("Could not clamp SL — abort.")
        shutdown_mt5()
        return 1
    if adjusted:
        logger.info("SL adjusted by broker rules → %.5f", sl_price)

    sl_distance = max(ask - sl_price, 1e-8)
    acct = mt5.account_info()
    equity = float(acct.equity) if acct else balance

    if args.lots is not None:
        lots = args.lots
    else:
        lots = calculate_lot_size(mt5_sym, equity, sl_distance, risk_pct=risk_pct)

    logger.info(
        "Opening BUY %s lots=%.4f ask=%.5f sl=%.5f dist=%.5f risk_pct=%.4f magic=%s",
        mt5_sym,
        lots,
        ask,
        sl_price,
        sl_distance,
        risk_pct,
        ex._magic,  # noqa: SLF001
    )

    if args.dry_run:
        logger.info("Dry run — not sending.")
        shutdown_mt5()
        return 0

    if not ex._check_margin_available(mt5_sym, lots, ask):  # noqa: SLF001
        logger.error("Margin check failed.")
        shutdown_mt5()
        return 1

    req = ex._build_buy_request(mt5_sym, lots, ask, sl_price, comment="smoke BUY")  # noqa: SLF001

    if args.tp_pct > 0.0:
        tp_price = ex._normalize_price(ask * (1.0 + args.tp_pct), digits)  # noqa: SLF001
        req["tp"] = tp_price
        logger.info("Take profit set to %.5f (+%.2f%% vs ask)", tp_price, args.tp_pct * 100.0)

    result = await ex._send_order_with_retry(req)  # noqa: SLF001
    if result is None:
        logger.error("order_send failed.")
        shutdown_mt5()
        return 1

    rc = int(getattr(result, "retcode", -1))
    logger.info(
        "order_send retcode=%s deal=%s order=%s comment=%s",
        rc,
        getattr(result, "deal", None),
        getattr(result, "order", None),
        getattr(result, "comment", ""),
    )
    done = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
    if rc != done:
        logger.warning("Unexpected retcode (expected TRADE_RETCODE_DONE=%s)", done)

    ticket = await _await_bot_position(mt5_sym, ex._magic)  # noqa: SLF001
    if ticket is None:
        logger.warning(
            "Could not find position with bot magic=%s — check MT5 (manual magic / overlap).",
            ex._magic,  # noqa: SLF001
        )
        shutdown_mt5()
        return 2

    logger.info("Position ticket=%s", ticket)
    positions = mt5.positions_get(ticket=ticket)
    if positions:
        p0 = positions[0]
        logger.info(
            "Broker snapshot: vol=%.4f open=%.5f sl=%.5f tp=%.5f profit=%.2f",
            p0.volume,
            p0.price_open,
            p0.sl,
            p0.tp,
            p0.profit,
        )

    if args.open_only:
        logger.info("--open-only: leaving position open.")
        shutdown_mt5()
        try:
            await close_db()
        except Exception:
            pass
        return 0

    await asyncio.sleep(args.pause)

    ok = await ex.close_position_by_ticket(ticket)
    if not ok:
        logger.error("close_position_by_ticket failed — close manually ticket=%s", ticket)
        shutdown_mt5()
        try:
            await close_db()
        except Exception:
            pass
        return 3

    logger.info("Closed ticket=%s OK.", ticket)
    shutdown_mt5()
    try:
        await close_db()
    except Exception:
        pass
    return 0


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "close":
        pc = argparse.ArgumentParser(prog="mt5_smoke_trade close", description="Close by ticket.")
        pc.add_argument("--ticket", type=int, required=True, help="MT5 position ticket")
        args = pc.parse_args(argv[1:])
        rc = asyncio.run(cmd_close(args))
        raise SystemExit(rc)

    parser = argparse.ArgumentParser(
        description="MT5 smoke test (same order stack as the bot).",
    )
    parser.add_argument(
        "--symbol",
        default=os.environ.get("MT5_TEST_SYMBOL", "DOGE/USDT"),
        help="Internal pair (see SYMBOL_MAP), default DOGE/USDT",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=float(os.environ.get("MT5_TEST_RISK_PCT", "0.005")),
        help="Risk fraction for lot sizing (default 0.5%%)",
    )
    parser.add_argument(
        "--sl-pct",
        type=float,
        default=float(os.environ.get("MT5_TEST_SL_PCT", "0.015")),
        help="SL distance below bid as fraction (default 1.5%%)",
    )
    parser.add_argument(
        "--tp-pct",
        type=float,
        default=float(os.environ.get("MT5_TEST_TP_PCT", "0")),
        help="Optional TP above ask as fraction (0 = off)",
    )
    parser.add_argument("--lots", type=float, default=None, help="Fixed lots (skip risk sizing)")
    parser.add_argument("--pause", type=float, default=3.0, help="Seconds before auto-close")
    parser.add_argument("--open-only", action="store_true", help="Do not close automatically")
    parser.add_argument("--dry-run", action="store_true", help="Print parameters only")

    args = parser.parse_args(argv)
    rc = asyncio.run(cmd_smoke(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
