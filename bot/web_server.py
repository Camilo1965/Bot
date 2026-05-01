import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from aiohttp import web

from bot.dashboard_helpers import compute_rsi, mt5_dashboard_mark
from bot import state as dash_state
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD, BUY_SENTIMENT_THRESHOLD

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClawdBot Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .neon-text { text-shadow: 0 0 10px rgba(56, 189, 248, 0.5); }
        .neon-green { color: #4ade80; text-shadow: 0 0 8px rgba(74, 222, 128, 0.6); }
        .neon-red { color: #f87171; text-shadow: 0 0 8px rgba(248, 113, 113, 0.6); }
        .card-enter { animation: fadein 0.3s ease-in; }
        @keyframes fadein { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-4xl mx-auto space-y-6">
        
        <!-- Header -->
        <header class="glass rounded-2xl p-6 flex flex-col md:flex-row justify-between items-center shadow-lg">
            <div>
                <h1 class="text-3xl font-bold text-sky-400 neon-text flex items-center gap-3">
                    🤖 ClawdBot <span class="text-sm font-normal text-slate-400 bg-slate-800 px-3 py-1 rounded-full border border-slate-700" id="uptime">--:--:--</span>
                </h1>
                <p class="text-slate-400 mt-2 text-sm">XGBoost_v3 + Gemini-2.5-Flash-Lite</p>
            </div>
            <div class="mt-4 md:mt-0 flex gap-4 text-center">
                <div class="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
                    <div class="text-xs text-slate-400 uppercase">Lectura IA</div>
                    <div class="text-xl font-bold" id="sentiment">--</div>
                    <div class="text-[11px] text-slate-400 mt-1" id="sentiment-detail">--</div>
                </div>
                <div class="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
                    <div class="text-xs text-slate-400 uppercase">Estado Bot</div>
                    <div class="text-xl font-bold" id="bot-status">--</div>
                </div>
            </div>
        </header>

        <!-- Riesgo y Billetera -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass rounded-xl p-4">
                <div class="text-xs text-slate-400">Balance</div>
                <div class="text-xl font-bold mt-1" id="balance">--</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-xs text-slate-400">Dinero Disponible</div>
                <div class="text-xl font-bold mt-1" id="margin">--</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-xs text-slate-400">Ganancia Sesión</div>
                <div class="text-xl font-bold mt-1" id="session-pnl">--</div>
            </div>
            <div class="glass rounded-xl p-4">
                <div class="text-xs text-slate-400">Peor Caída Sesión</div>
                <div class="text-xl font-bold mt-1" id="drawdown">--</div>
            </div>
        </div>

        <!-- Market Data -->
        <h2 class="text-xl font-semibold text-slate-200 mt-8 mb-4 px-2">📊 Mercado en Vivo</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="market-cards">
            <!-- Renderizado por JS -->
        </div>

        <!-- Eventos -->
        <h2 class="text-xl font-semibold text-slate-200 mt-8 mb-4 px-2">📋 Últimos Eventos</h2>
        <div class="glass rounded-xl p-4 space-y-2 text-sm text-slate-300 font-mono" id="events-log">
            <!-- Renderizado por JS -->
        </div>

    </div>

    <script>
        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                render(data);
            } catch (err) {
                console.error("Error fetching state", err);
            }
        }

        function render(data) {
            document.getElementById('uptime').innerText = data.uptime;
            
            const sentEl = document.getElementById('sentiment');
            sentEl.innerText = data.sentiment_label;
            sentEl.className = `text-xl font-bold ${data.sentiment_num >= 0.60 ? 'neon-green' : data.sentiment_num >= 0.45 ? 'text-yellow-400' : 'neon-red'}`;
            document.getElementById('sentiment-detail').innerText = data.sentiment_detail;

            const botStatusEl = document.getElementById('bot-status');
            botStatusEl.innerHTML = data.global_hold ? '⛔ En pausa' : '✅ Operando';
            botStatusEl.className = `text-xl font-bold ${data.global_hold ? 'neon-red' : 'neon-green'}`;

            document.getElementById('balance').innerText = data.balance;
            document.getElementById('margin').innerText = data.available_margin;
            
            const pnlEl = document.getElementById('session-pnl');
            pnlEl.innerText = data.session_pnl;
            pnlEl.className = `text-xl font-bold mt-1 ${data.session_pnl_num >= 0 ? 'neon-green' : 'neon-red'}`;

            const ddEl = document.getElementById('drawdown');
            ddEl.innerText = data.max_drawdown;
            ddEl.className = `text-xl font-bold mt-1 ${data.max_drawdown_num < 0 ? 'neon-red' : 'text-slate-200'}`;

            // Render Market Cards
            const marketContainer = document.getElementById('market-cards');
            marketContainer.innerHTML = '';
            data.market.forEach(item => {
                let actionColor = item.action.includes('Gestionando') ? 'text-sky-400' : item.action.includes('Comprar') ? 'neon-green' : 'text-yellow-400';
                let bgStyle = item.has_position ? 'border-sky-500/50 bg-sky-900/20' : 'border-slate-700/50 bg-slate-800/30';
                
                let html = `
                    <div class="glass rounded-2xl p-5 border ${bgStyle} card-enter">
                        <div class="flex justify-between items-start mb-3">
                            <h3 class="text-2xl font-bold">${item.symbol}</h3>
                            <span class="text-xs font-semibold px-2 py-1 rounded bg-slate-900/50 ${actionColor}">${item.action}</span>
                        </div>
                        <div class="text-3xl font-mono tracking-tight mb-1">${item.price}</div>
                        <div class="text-sm ${item.change_num >= 0 ? 'neon-green' : 'neon-red'} mb-4">${item.change} en 24h</div>
                        
                        <div class="grid grid-cols-2 gap-y-3 gap-x-2 text-sm border-t border-slate-700/50 pt-4">
                            <div class="text-slate-400">Confianza IA: <span class="text-slate-200 font-semibold float-right">${item.ml_conf}</span></div>
                            <div class="text-slate-400">RSI: <span class="text-slate-200 font-semibold float-right">${item.rsi}</span></div>
                            <div class="text-slate-400 col-span-2">Estado Mercado: <span class="text-slate-200 font-semibold float-right">${item.trend}</span></div>
                        </div>
                `;

                if (item.has_position) {
                    let pnlColor = item.unrealized_pnl_num >= 0 ? 'neon-green' : 'neon-red';
                    html += `
                        <div class="mt-4 bg-slate-900/50 rounded-lg p-3 border border-slate-700">
                            <div class="flex justify-between text-sm mb-1">
                                <span class="text-sky-400 font-semibold">${item.position_str}</span>
                                <span class="${pnlColor} font-bold font-mono">${item.unrealized_pnl}</span>
                            </div>
                        </div>
                    `;
                }

                html += `</div>`;
                marketContainer.innerHTML += html;
            });

            // Render Events
            const eventsContainer = document.getElementById('events-log');
            if (data.events.length > 0) {
                eventsContainer.innerHTML = data.events.map(e => `<div class="border-b border-slate-800 pb-2 last:border-0">${e}</div>`).join('');
            } else {
                eventsContainer.innerHTML = '<div class="text-slate-500 italic">Sin eventos recientes...</div>';
            }
        }

        // Init
        fetchState();
        setInterval(fetchState, 1000); // Refresca cada 1 segundo
    </script>
</body>
</html>
"""

async def start_web_dashboard(
    state: dict[str, Any],
    paper_executor: PaperExecutor,
    risk_manager: RiskManager,
    watchlist: list[str],
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Run an aiohttp web server providing a modern real-time mobile dashboard."""
    
    async def handle_html(request: web.Request) -> web.Response:
        return web.Response(text=HTML_TEMPLATE, content_type="text/html")

    async def handle_api(request: web.Request) -> web.Response:
        now_utc = datetime.now(tz=timezone.utc)
        
        # Uptime
        if dash_state.bot_start_time is not None:
            elapsed = now_utc - dash_state.bot_start_time
            total_secs = int(elapsed.total_seconds())
            h, rem = divmod(total_secs, 3600)
            m, s = divmod(rem, 60)
            uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            uptime_str = "00:00:00"

        # Sentiment
        sentiment = state.get("sentiment")
        news_hold_until = state.get("news_hold_until")
        global_hold = (
            (sentiment is not None and sentiment < BUY_SENTIMENT_THRESHOLD)
            or (news_hold_until is not None and now_utc < news_hold_until)
        )
        
        # Sentiment label
        if sentiment is None:
            sentiment_label = "Sin datos"
            sentiment_detail = "Esperando lectura IA"
        elif sentiment >= 0.60:
            sentiment_label = "🟢 Favorable compra"
            sentiment_detail = "Mercado con sesgo positivo"
        elif sentiment >= 0.45:
            sentiment_label = "🟡 Neutral / esperar"
            sentiment_detail = "Señal mixta, mejor esperar confirmación"
        else:
            sentiment_label = "🔴 Riesgo alto"
            sentiment_detail = "Evitar compras por ahora"

        # Wallet
        mt5_wallet = state.get("mt5_wallet")
        use_mt5_wallet = isinstance(mt5_wallet, dict) and "balance" in mt5_wallet
        if use_mt5_wallet:
            balance = float(mt5_wallet["balance"])
            avail = float(mt5_wallet["margin_free"])
        else:
            balance = risk_manager.balance
            avail = balance - sum(p.position_size / 5.0 for p in paper_executor.open_positions.values())

        # Market Data
        market_data = []
        ml_probs = state.get("ml_probs", {})
        htf_trend = state.get("htf_trend", {})
        
        for sym in watchlist:
            prices = list(state["prices"].get(sym, []))
            candle_last = prices[-1] if prices else None
            price = mt5_dashboard_mark(state, sym, candle_last)
            
            if price is None:
                continue

            # 24h change
            pct = 0.0
            if len(prices) >= 96:
                p24 = prices[-96]
                if p24 > 0:
                    pct = (price - p24) / p24 * 100
                    
            rsi = compute_rsi(prices) if len(prices) >= 15 else None
            prob = ml_probs.get(sym, 0.0)
            t1h = htf_trend.get(sym, {}).get("1h", "neut")
            
            pos = paper_executor.open_positions.get(sym)
            has_pos = bool(pos)
            unrl = 0.0
            pos_str = ""
            
            if has_pos:
                qty = pos.position_size / pos.entry_price
                unrl = (price - pos.entry_price) * qty
                pos_str = f"Entrada {pos.entry_price:,.2f}"
                action = "Gestionando posición"
            else:
                action = "Comprar" if prob >= BUY_PROB_THRESHOLD else "Esperar"
                
            market_data.append({
                "symbol": sym.split("/")[0],
                "price": f"{price:,.2f}",
                "change": f"{'+' if pct >=0 else ''}{pct:.1f}%",
                "change_num": pct,
                "rsi": f"{rsi:.1f}" if rsi is not None else "--",
                "ml_conf": f"{prob*100:.0f}%",
                "trend": (
                    "Alcista" if t1h == "bullish"
                    else "Bajista" if t1h == "bearish"
                    else "Neutral"
                ),
                "action": action,
                "has_position": has_pos,
                "position_str": pos_str,
                "unrealized_pnl": f"{'+' if unrl >=0 else ''}{unrl:.2f} USDT",
                "unrealized_pnl_num": unrl
            })

        # Events
        events = list(dash_state.dashboard_events)[-3:]
        # Strip rich tags like [dim], [red], etc for HTML
        import re
        clean_events = [re.sub(r'\[.*?\]', '', e) for e in events]

        resp = {
            "uptime": uptime_str,
            "sentiment": f"{sentiment:.4f}" if sentiment is not None else "--",
            "sentiment_label": sentiment_label,
            "sentiment_detail": sentiment_detail,
            "sentiment_num": sentiment if sentiment is not None else 0,
            "global_hold": global_hold,
            "balance": f"{balance:,.2f}",
            "available_margin": f"{avail:,.2f}",
            "session_pnl": f"{paper_executor.total_pnl:+.2f}",
            "session_pnl_num": paper_executor.total_pnl,
            "max_drawdown": f"{state.get('max_drawdown', 0.0):+.2f}",
            "max_drawdown_num": state.get("max_drawdown", 0.0),
            "market": market_data,
            "events": clean_events
        }
        
        return web.json_response(resp)

    app = web.Application()
    app.add_routes([
        web.get("/", handle_html),
        web.get("/api/state", handle_api)
    ])

    # ... existing aiohttp runner code ...
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    logger.info("🌐 Web dashboard server listening on http://%s:%d", host, port)
    await site.start()
    
    while True:
        await asyncio.sleep(3600)
