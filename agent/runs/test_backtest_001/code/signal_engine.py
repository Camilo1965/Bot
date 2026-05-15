import pandas as pd
import numpy as np

class SignalEngine:
    """Simple 20/50 MA crossover strategy."""
    
    def __init__(self, config: dict):
        self.config = config
        self.fast_window = config.get("fast_window", 20)
        self.slow_window = config.get("slow_window", 50)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ma_fast'] = df['close'].rolling(window=self.fast_window).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_window).mean()
        
        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
        
        df['position'] = df['signal'].shift(1).fillna(0)
        return df
