import sys

with open('bot/signal_emitter.py', 'r', encoding='utf-8') as f:
    content = f.read()

import_aiohttp = '''import aiohttp

from bot.constants import DEBUG_LOG_HINT'''

content = content.replace('from bot.constants import DEBUG_LOG_HINT', import_aiohttp)

mcp_vars = '''from vibe.feature_bridge import extract_vibe_features

_VIBE_MCP_URL: str = os.environ.get("VIBE_MCP_URL", "http://localhost:5000/predict")
_VIBE_MCP_TIMEOUT: float = float(os.environ.get("VIBE_MCP_TIMEOUT_S", "30"))
_VIBE_MCP_ENABLED: bool = os.environ.get("VIBE_MCP_ENABLED", "1").strip().lower() in ("1", "true", "yes")

async def _fetch_vibe_mcp_decision(
    session: aiohttp.ClientSession,
    features: list[float],
    ohlcv: list[dict[str, Any]],
    symbol: str,
) -> dict[str, Any] | None:
    payload = {
        "features": features,
        "ohlcv": ohlcv,
        "symbol": symbol,
    }
    try:
        async with session.post(
            _VIBE_MCP_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=_VIBE_MCP_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            text = await resp.text()
            logging.getLogger("clawdbot.signal").warning(
                "[VIBE MCP] HTTP %d: %s", resp.status, text[:200]
            )
            return None
    except asyncio.TimeoutError:
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Timeout after %.0fs", _VIBE_MCP_TIMEOUT
        )
        return None
    except Exception as exc:
        logging.getLogger("clawdbot.signal").warning(
            "[VIBE MCP] Request failed: %s", exc
        )
        return None
'''

content = content.replace('from vibe.feature_bridge import extract_vibe_features', mcp_vars)

# In order to not mess up indentation of 300 lines, I will just create the ClientSession inside the while loop,
# OR I'll just use a small wrapper function or instantiate it dynamically on the class. 
# Better yet: Instantiate one global session per execution? No, let's just put it outside the while True loop.

loop_start_old = '''    logger = logging.getLogger("clawdbot.signal")
    while True:'''

loop_start_new = '''    logger = logging.getLogger("clawdbot.signal")
    
    # We create a persistent session manually to avoid shifting the entire indentation of the while-loop
    http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=_VIBE_MCP_TIMEOUT))
    try:
        while True:'''

content = content.replace(loop_start_old, loop_start_new)

# Re-indent ONLY the body by finding it. Wait, if I do `try: while True:`, I still have to shift `while True:` and its body!
# Let's just create it WITHOUT async with!
# `http_session = aiohttp.ClientSession()`
# and just leave it open. aiohttp will warn on exit, but it's fine for an infinite loop.

loop_start_new_no_indent = '''    logger = logging.getLogger("clawdbot.signal")
    
    http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=_VIBE_MCP_TIMEOUT))
    while True:'''

content = content.replace(loop_start_new, loop_start_new_no_indent)


vibe_mcp_call = '''            # [VIBE FASE 4] Extraer features numéricas para el modelo XGBoost
            vibe_vec = None
            if os.environ.get("VIBE_FEATURES_ENABLED") == "1":
                vibe_vec = extract_vibe_features(state, symbol)

            if _VIBE_MCP_ENABLED:
                ohlcv_list = []
                n = min(len(prices), len(highs) if highs else len(prices), len(lows) if lows else len(prices), len(volumes) if volumes else len(prices))
                for i in range(n):
                    o_val = prices[i]
                    h_val = highs[i] if highs else o_val
                    l_val = lows[i] if lows else o_val
                    v_val = volumes[i] if volumes else 0.0
                    ohlcv_list.append({"open": o_val, "high": h_val, "low": l_val, "close": o_val, "volume": v_val})
                
                q_feat = predictor._compute_features(
                    prices, 0.0, highs=highs or None, lows=lows or None,
                    obi_ratio=obi_ratio, volumes=volumes or None, symbol=symbol
                )
                full_features = (q_feat or []) + (vibe_vec or [0.0, 0.0, 1.0, 0.5])
                
                asyncio.create_task(
                    _fetch_vibe_mcp_decision(http_session, full_features, ohlcv_list, symbol)
                )

            # Single-pass prediction'''

old_vibe = '''            # [VIBE FASE 4] Extraer features numéricas para el modelo XGBoost
            vibe_vec = None
            if os.environ.get("VIBE_FEATURES_ENABLED") == "1":
                vibe_vec = extract_vibe_features(state, symbol)

            # Single-pass prediction'''

content = content.replace(old_vibe, vibe_mcp_call)

with open('bot/signal_emitter.py', 'w', encoding='utf-8') as f:
    f.write(content)
