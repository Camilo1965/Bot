import sys

with open('bot/signal_emitter.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = """                ohlcv_list = []
                n = min(len(prices), len(highs) if highs else len(prices), len(lows) if lows else len(prices), len(volumes) if volumes else len(prices))
                for i in range(n):
                    o_val = prices[i]
                    h_val = highs[i] if highs else o_val
                    l_val = lows[i] if lows else o_val
                    v_val = volumes[i] if volumes else 0.0
                    ohlcv_list.append({"open": o_val, "high": h_val, "low": l_val, "close": o_val, "volume": v_val})"""

new = """                ohlcv_list = []
                n = min(len(prices), len(highs) if highs else len(prices), len(lows) if lows else len(prices), len(volumes) if volumes else len(prices))
                compress_ohlcv = n > 200
                for i in range(n):
                    o_val = prices[i]
                    v_val = volumes[i] if volumes else 0.0
                    if compress_ohlcv:
                        ohlcv_list.append({"close": o_val, "volume": v_val})
                    else:
                        h_val = highs[i] if highs else o_val
                        l_val = lows[i] if lows else o_val
                        ohlcv_list.append({"open": o_val, "high": h_val, "low": l_val, "close": o_val, "volume": v_val})"""

content = content.replace(old, new)
with open('bot/signal_emitter.py', 'w', encoding='utf-8') as f:
    f.write(content)
