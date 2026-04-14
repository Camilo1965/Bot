import MetaTrader5 as mt5

# 1. Iniciar el motor
print("🚀 Iniciando conexión con Admirals...")
if not mt5.initialize():
    print("❌ Falló la inicialización")
    quit()

# 2. Definir el símbolo EXACTO que vimos en tu captura
symbol = "BTCUSD-T"

# 3. Forzar que el símbolo aparezca en el Market Watch
if not mt5.symbol_select(symbol, True):
    print(f"❌ No se pudo seleccionar el símbolo {symbol}. ¿Está en dorado en MT5?")
    mt5.shutdown()
    quit()

# 4. Obtener datos en tiempo real
tick = mt5.symbol_info_tick(symbol)

if tick:
    print(f"\n✅ ¡CONEXIÓN EXITOSA A BITCOIN!")
    print(f"----------------------------------")
    print(f"Símbolo:    {symbol}")
    print(f"Precio Compra (Ask): {tick.ask}")
    print(f"Precio Venta (Bid):  {tick.bid}")
    print(f"Spread actual:       {round(tick.ask - tick.bid, 2)}")
    print(f"----------------------------------")
    print("¡El bot ya puede leer el mercado!")
else:
    print(f"⚠️ No se recibieron ticks para {symbol}. Revisa si el mercado está abierto.")

# 5. Apagar
mt5.shutdown()