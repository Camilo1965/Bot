
import asyncio
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

# Mock de lo que necesitamos para probar
@dataclass
class MockOpenPosition:
    symbol: str
    entry_price: float
    stop_loss_price: float
    current_stop_loss: float
    peak_price: float
    trailing_stop_active: bool = False
    activation_pct: float = 0.01  # 1% como pusimos hoy

async def simulate_atr_and_sl_logic():
    print("\n" + "="*60)
    print("🔬 TEST DE LÓGICA: ATR, ACTIVACIÓN (1%) Y SENTIMIENTO")
    print("="*60)

    # 1. Configuración inicial
    entry_price = 100.0
    initial_sl = 98.5 # -1.5%
    pos = MockOpenPosition(
        symbol="ETH/USDT",
        entry_price=entry_price,
        stop_loss_price=initial_sl,
        current_stop_loss=initial_sl,
        peak_price=entry_price
    )

    # Multiplicadores (lo que hace Gemini)
    sentiment_multipliers = [0.8, 1.0, 1.8] # [Bajista/Scalp, Neutral, Muy Alcista/Swing]
    
    # Simulación de volatilidad (ATR)
    atr_values = [0.5, 1.0, 2.5] # [Baja, Normal, Alta]

    for sentiment in sentiment_multipliers:
        print(f"\n--- Probando con Multiplicador de Sentimiento: {sentiment} ---")
        
        for atr in atr_values:
            # Lógica simplificada del bot
            # Distancia base = ATR * Multiplicador ATR (usamos 2.5 que es el default del bot)
            base_distance = atr * 2.5
            
            # Aplicamos el "espacio" de Gemini
            dynamic_distance = base_distance * sentiment
            
            # Nueva parada si el precio subiera a 101.5 (ya pasó el 1% de activación)
            current_price = 101.5
            new_sl = current_price - dynamic_distance
            
            print(f"ATR: {atr:.2f} | Distancia SL: {dynamic_distance:.2f} | Si precio es {current_price} -> Nuevo SL sería: {new_sl:.2f}")
            
            if new_sl > entry_price:
                print(f"   ✅ RESULTADO: Trade Protegido (SL arriba de entrada)")
            else:
                print(f"   ⚠️ RESULTADO: SL aún por debajo de entrada (Dando aire)")

    print("\n" + "="*60)
    print("CONCLUSIÓN: El ATR y Gemini dictan qué tan lejos está el SL.")
    print("Con ATR alto y Sentimiento alto, el SL se aleja para NO sacarte en retrocesos.")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(simulate_atr_and_sl_logic())
