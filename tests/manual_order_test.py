"""
Prueba única de ejecución MT5: compra de mercado y cierre pocos segundos después.

Requiere MT5 abierto, mismas credenciales que en .env, y símbolo en Market Watch.
Envía órdenes reales (demo o real según la cuenta conectada).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
from dotenv import load_dotenv

from execution.mt5_executor import MT5Executor
from risk.risk_manager import RiskManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


class MockDB:
    """Solo insert_open_trade es necesario para el flujo de apertura del executor."""

    async def insert_open_trade(self, *args: object, **kwargs: object) -> int:
        return 1

    def __getattr__(self, name: str):  # noqa: ANN001
        async def _noop(*args: object, **kwargs: object) -> None:
            return None

        return _noop


BROKER_SYMBOL = "BTCUSD-T"
INTERNAL_SYMBOL = "BTC/USDT"


async def _run_round_trip() -> int:
    print("Prueba MT5: compra + venta (market)")
    print("-" * 50)

    if not mt5.initialize():
        print("ERROR: mt5.initialize() falló.")
        return 1

    login_raw = os.getenv("MT5_LOGIN", "").strip()
    password = os.getenv("MT5_PASSWORD", "").strip()
    server = os.getenv("MT5_SERVER", "").strip()
    if not login_raw or not password or not server:
        print("ERROR: define MT5_LOGIN, MT5_PASSWORD y MT5_SERVER en .env")
        return 1

    try:
        login = int(login_raw)
    except ValueError:
        print("ERROR: MT5_LOGIN debe ser un número entero.")
        return 1

    if not mt5.login(login, password=password, server=server):
        print(f"ERROR: mt5.login falló: {mt5.last_error()}")
        return 1

    if not mt5.symbol_select(BROKER_SYMBOL, True):
        print(f"ERROR: no se pudo seleccionar {BROKER_SYMBOL} en Market Watch.")
        return 1
    print(f"OK: símbolo {BROKER_SYMBOL} activo.")
    time.sleep(1.0)

    tick = mt5.symbol_info_tick(BROKER_SYMBOL)
    if tick is None or tick.ask <= 0:
        print("ERROR: sin tick válido para el símbolo.")
        return 1

    acct = mt5.account_info()
    if acct is None:
        print("ERROR: mt5.account_info() es None.")
        return 1

    risk = RiskManager(initial_balance=float(acct.equity))
    executor = MT5Executor(db=MockDB(), risk_manager=risk, live=True)

    entry = float(tick.ask)
    datos = {
        "symbol": INTERNAL_SYMBOL,
        "entry_price": entry,
        "win_probability": 0.99,
        "current_atr": 5000.0,
        "timestamp": datetime.now(tz=timezone.utc),
    }

    print(f"Enviando BUY vía executor (precio referencia ask={entry:.2f})...")
    opened = await executor.try_open_trade(**datos)

    if opened is not True:
        print("ERROR: try_open_trade devolvió False (orden no completada o chequeos fallaron).")
        print(f"  mt5.last_error() = {mt5.last_error()}")
        print("  Revisa también mensajes [MT5] en la consola y bot_debug.log.")
        return 1

    await asyncio.sleep(0.5)
    positions = executor.get_open_positions(INTERNAL_SYMBOL)
    if not positions:
        positions = executor.get_open_positions()

    ticket: int | None = None
    for p in positions:
        if p.get("symbol") == BROKER_SYMBOL and p.get("type") == "BUY":
            ticket = int(p["ticket"])
            break

    if ticket is None:
        print(
            "ADVERTENCIA: la apertura devolvió True pero no hay posición con magic del bot "
            "en get_open_positions. Comprueba la pestaña Operaciones en MT5."
        )
        return 1

    print(f"OK: posición abierta, ticket MT5 = {ticket}")
    print("Esperando 5 s antes de cerrar...")
    await asyncio.sleep(5.0)

    closed = await executor.close_position_by_ticket(ticket)
    if not closed:
        print(f"ERROR: cierre fallido. mt5.last_error() = {mt5.last_error()}")
        return 1

    print("OK: orden de cierre enviada. Verifica el historial en MT5.")
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run_round_trip()))
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
