import MetaTrader5 as mt5
import os
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
from execution.mt5_executor import MT5Executor

load_dotenv()

# --- MOCKS NECESARIOS ---
class MockRiskManager:
    def __getattr__(self, name):
        def magic_method(*args, **kwargs):
            if name.startswith('is_'): return False
            if name == 'get_position_size': return 0.1 # 0.1 que funcionó en el test raw
            return True
        return magic_method

class MockDB:
    def __getattr__(self, name):
        async def async_magic(*args, **kwargs): return True
        return async_magic

async def test_bypass():
    print("🚀 TEST DE CONEXIÓN FORZADA")
    print("-" * 50)

    # 1. Inicialización y Login manual ANTES que nada
    if not mt5.initialize():
        print("❌ Error al inicializar MT5")
        return

    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    
    if not mt5.login(login, password=password, server=server):
        print(f"❌ Error de Login: {mt5.last_error()}")
        return

    # 2. Obligamos a MT5 a seleccionar el símbolo y esperamos 2 segundos
    simbolo_real = "BTCUSD-T"
    if mt5.symbol_select(simbolo_real, True):
        print(f"✅ Símbolo {simbolo_real} seleccionado y activo.")
        time.sleep(2) # Pausa para que el terminal reciba datos
    else:
        print(f"❌ No se pudo seleccionar {simbolo_real}")
        return

    # 3. Ahora sí, creamos el Executor
    executor = MT5Executor(db=MockDB(), risk_manager=MockRiskManager(), live=True)

    # 4. Datos de la IA
    # IMPORTANTE: Si el bot falla con "BTC/USDT", usa "BTCUSD-T" directamente aquí
    # para probar si el problema es el traductor.
    datos_ia = {
        "symbol": "BTCUSD-T", # <--- Probamos con el nombre directo del broker
        "entry_price": mt5.symbol_info_tick(simbolo_real).ask,
        "win_probability": 0.99,
        "current_atr": 5000.0,
        "timestamp": datetime.now()
    }

    print(f"🤖 Enviando orden de {simbolo_real} al Executor...")
    
    try:
        ticket = await executor.try_open_trade(**datos_ia)
        
        if ticket and not isinstance(ticket, bool):
            print(f"🔥 ¡ORDEN REAL ABIERTA! Ticket: {ticket}")
            print("Cerrando en 5 segundos...")
            await asyncio.sleep(5)
            await executor.close_position_by_ticket(ticket)
        else:
            print("\n❌ El bot sigue devolviendo None.")
            print("TIP: Si el 'Expertos' sigue vacío, el bot tiene un 'return None' escondido.")
            
    except Exception as e:
        print(f"💥 Error durante la ejecución: {e}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(test_bypass())