import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from aiohttp import web

from bot.dashboard_helpers import compute_rsi, mt5_dashboard_mark
from bot import state as dash_state
from database.db_manager import db
from execution.paper_executor import PaperExecutor
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD, BUY_SENTIMENT_THRESHOLD

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClawdBot Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { 
            background: radial-gradient(circle at top right, #1e293b, #0f172a 40%, #020617 100%);
            color: #f8fafc; 
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
        }
        .glass { 
            background: rgba(30, 41, 59, 0.4); 
            backdrop-filter: blur(16px); 
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.05); 
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .glass-card {
            background: linear-gradient(145deg, rgba(30,41,59,0.4) 0%, rgba(15,23,42,0.6) 100%);
            border: 1px solid rgba(255,255,255,0.05);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
        }
        .glass-card:hover {
            border-color: rgba(255,255,255,0.15);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        .neon-text { text-shadow: 0 0 15px rgba(56, 189, 248, 0.4); }
        .text-glow-green { text-shadow: 0 0 10px rgba(74, 222, 128, 0.5); color: #4ade80; }
        .text-glow-red { text-shadow: 0 0 10px rgba(248, 113, 113, 0.5); color: #f87171; }
        
        .pulse-dot {
            height: 8px; width: 8px; border-radius: 50%; display: inline-block;
            box-shadow: 0 0 8px currentColor;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .value-up { animation: flash-green 1s ease-out; }
        .value-down { animation: flash-red 1s ease-out; }
        @keyframes flash-green { 0% { color: #4ade80; text-shadow: 0 0 15px #4ade80; } 100% { color: inherit; text-shadow: none; } }
        @keyframes flash-red { 0% { color: #f87171; text-shadow: 0 0 15px #f87171; } 100% { color: inherit; text-shadow: none; } }
        
        .progress-bar-bg { background: rgba(0,0,0,0.3); border-radius: 999px; overflow: hidden; height: 6px; }
        .progress-bar-fill { height: 100%; transition: width 0.5s ease-in-out; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-5xl mx-auto space-y-6">
        
        <!-- Header & System Health -->
        <header class="glass rounded-3xl p-6 flex flex-col md:flex-row justify-between items-start md:items-center">
            <div>
                <h1 class="text-3xl font-bold text-sky-400 neon-text flex items-center gap-3">
                    ClawdBot <span class="text-xs font-medium text-slate-400 bg-black/30 px-3 py-1 rounded-full border border-white/5 tracking-wider uppercase">Pro</span>
                </h1>
                <div class="flex items-center gap-4 mt-3 text-xs text-slate-400">
                    <span class="flex items-center gap-2"><span class="pulse-dot text-emerald-400"></span> MT5 Activo</span>
                    <span class="flex items-center gap-2"><span class="pulse-dot text-emerald-400" style="animation-delay: 0.5s"></span> Feed OK</span>
                    <span class="flex items-center gap-2"><span class="pulse-dot text-sky-400" style="animation-delay: 1s"></span> <span id="uptime">--:--:--</span></span>
                </div>
            </div>
            
            <div class="mt-6 md:mt-0 flex gap-3 w-full md:w-auto">
                <div class="bg-black/20 p-4 rounded-2xl border border-white/5 flex-1 md:w-48">
                    <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Lectura IA</div>
                    <div class="text-lg font-bold" id="sentiment">--</div>
                    <div class="text-xs text-slate-400 mt-1" id="sentiment-detail">--</div>
                </div>
                <div class="bg-black/20 p-4 rounded-2xl border border-white/5 flex-1 md:w-40">
                    <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Estado Bot</div>
                    <div class="text-lg font-bold" id="bot-status">--</div>
                </div>
            </div>
        </header>

        <!-- Fintech Wallet Panel -->
        <div class="glass rounded-3xl p-6 relative overflow-hidden">
            <div class="absolute -right-20 -top-20 w-64 h-64 bg-sky-500/10 rounded-full blur-3xl"></div>
            
            <div class="text-[11px] text-slate-400 uppercase tracking-widest font-semibold mb-6">Resumen Financiero</div>
            
            <div class="flex flex-col md:flex-row divide-y md:divide-y-0 md:divide-x divide-white/5">
                <div class="flex-1 pb-6 md:pb-0 md:pr-8">
                    <div class="text-sm text-slate-400 mb-1">Balance Total</div>
                    <div class="text-4xl font-light tracking-tight"><span class="text-slate-500">$</span><span id="balance" class="font-medium text-white">--</span></div>
                </div>
                <div class="flex-1 py-6 md:py-0 md:px-8">
                    <div class="text-sm text-slate-400 mb-1">Dinero Disponible</div>
                    <div class="text-2xl font-light mt-2"><span class="text-slate-500">$</span><span id="margin" class="font-medium text-white">--</span></div>
                </div>
                <div class="flex-1 py-6 md:py-0 md:px-8">
                    <div class="text-sm text-slate-400 mb-1">Ganancia Sesión</div>
                    <div class="text-2xl font-medium mt-2" id="session-pnl">--</div>
                </div>
                <div class="flex-1 pt-6 md:pt-0 md:pl-8">
                    <div class="text-sm text-slate-400 mb-1">Peor Caída Sesión</div>
                    <div class="text-2xl font-medium mt-2" id="drawdown">--</div>
                </div>
            </div>
        </div>

        <!-- CEO Snapshot -->
        <div class="glass rounded-3xl p-6">
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-2 mb-4">
                <h2 class="text-lg font-medium text-slate-300 tracking-wide">CEO SNAPSHOT</h2>
                <div class="text-xs text-slate-500" id="ceo-updated">Actualizado: --</div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
                <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
                    <div class="text-[10px] uppercase tracking-widest text-slate-500 mb-1">PnL 7D</div>
                    <div class="text-lg font-semibold" id="ceo-pnl-7d">--</div>
                </div>
                <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
                    <div class="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Winrate 7D</div>
                    <div class="text-lg font-semibold" id="ceo-winrate-7d">--</div>
                </div>
                <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
                    <div class="text-[10px] uppercase tracking-widest text-slate-500 mb-1">PnL 30D</div>
                    <div class="text-lg font-semibold" id="ceo-pnl-30d">--</div>
                </div>
                <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
                    <div class="text-[10px] uppercase tracking-widest text-slate-500 mb-1">PF 30D</div>
                    <div class="text-lg font-semibold" id="ceo-pf-30d">--</div>
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
                    <div class="text-xs text-slate-400 mb-2">Rendimiento por símbolo (mes)</div>
                    <div class="space-y-2 text-sm" id="ceo-symbols"></div>
                </div>
                <div class="bg-black/20 rounded-2xl p-4 border border-white/5">
                    <div class="text-xs text-slate-400 mb-2">Últimos trades cerrados</div>
                    <div class="flex flex-wrap gap-2 mb-3" id="ceo-trades-filters">
                        <button type="button" data-filter="today" class="px-2 py-1 text-xs rounded border border-white/10 text-slate-300 hover:border-sky-400/50">Hoy</button>
                        <button type="button" data-filter="7d" class="px-2 py-1 text-xs rounded border border-white/10 text-slate-300 hover:border-sky-400/50">7D</button>
                        <button type="button" data-filter="30d" class="px-2 py-1 text-xs rounded border border-white/10 text-slate-300 hover:border-sky-400/50">30D</button>
                        <button type="button" data-filter="all" class="px-2 py-1 text-xs rounded border border-sky-400/50 text-sky-300">Todo</button>
                    </div>
                    <div class="space-y-2 text-sm" id="ceo-trades"></div>
                    <button id="ceo-trades-more" class="mt-3 text-xs text-sky-400 hover:text-sky-300 hidden" type="button">Cargar más</button>
                </div>
            </div>
        </div>

        <!-- Market Cards -->
        <div class="flex justify-between items-end mt-10 mb-4 px-2">
            <h2 class="text-lg font-medium text-slate-300 tracking-wide">MERCADO EN VIVO</h2>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6" id="market-cards">
            <!-- JS Renders Here -->
        </div>

        <!-- Events -->
        <div class="mt-8">
            <h2 class="text-lg font-medium text-slate-300 tracking-wide mb-4 px-2">LÍNEA DE TIEMPO</h2>
            <div class="glass rounded-2xl p-6 space-y-4" id="events-log">
                <!-- JS Renders Here -->
            </div>
        </div>
    </div>

    <script>
        const stateCache = {};
        let ceoTradeVisibleCount = 10;
        let ceoTradeRows = [];
        let ceoTradeFilter = 'all';

        function filterCeoTrades(rows) {
            if (ceoTradeFilter === 'all') return rows;
            const now = new Date();
            const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const dayMs = 24 * 60 * 60 * 1000;
            return rows.filter((t) => {
                if (!t.exit_time_iso) return false;
                const d = new Date(t.exit_time_iso);
                if (Number.isNaN(d.getTime())) return false;
                if (ceoTradeFilter === 'today') {
                    return d >= startToday;
                }
                if (ceoTradeFilter === '7d') {
                    return (now - d) <= (7 * dayMs);
                }
                if (ceoTradeFilter === '30d') {
                    return (now - d) <= (30 * dayMs);
                }
                return true;
            });
        }

        function updateCeoFilterButtons() {
            document.querySelectorAll('#ceo-trades-filters button[data-filter]').forEach((btn) => {
                const active = btn.dataset.filter === ceoTradeFilter;
                btn.className = active
                    ? 'px-2 py-1 text-xs rounded border border-sky-400/50 text-sky-300'
                    : 'px-2 py-1 text-xs rounded border border-white/10 text-slate-300 hover:border-sky-400/50';
            });
        }

        function updateValue(id, newValue, className = '') {
            const el = document.getElementById(id);
            if (!el) return;
            
            if (stateCache[id] !== newValue) {
                el.innerText = newValue;
                if (className) el.className = className;
                
                // Flash effect
                if (stateCache[id] !== undefined) {
                    const oldNum = parseFloat(stateCache[id].toString().replace(/[^0-9.-]+/g, ""));
                    const newNum = parseFloat(String(newValue).replace(/[^0-9.-]+/g, ""));
                    
                    if (!isNaN(oldNum) && !isNaN(newNum) && oldNum !== newNum) {
                        el.classList.remove('value-up', 'value-down');
                        void el.offsetWidth; // trigger reflow
                        el.classList.add(newNum > oldNum ? 'value-up' : 'value-down');
                    }
                }
                stateCache[id] = newValue;
            }
        }

        async function fetchState() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                window.__lastData = data;
                render(data);
            } catch (err) {
                console.error("Error fetching state", err);
            }
        }

        function render(data) {
            document.getElementById('uptime').innerText = data.uptime;
            
            const sentEl = document.getElementById('sentiment');
            sentEl.innerText = data.sentiment_label;
            sentEl.className = `text-lg font-bold ${data.sentiment_num >= 0.60 ? 'text-glow-green' : data.sentiment_num >= 0.45 ? 'text-yellow-400' : 'text-glow-red'}`;
            document.getElementById('sentiment-detail').innerText = data.sentiment_detail;

            const botStatusEl = document.getElementById('bot-status');
            botStatusEl.innerHTML = data.global_hold ? '<span class="text-glow-red">En pausa</span>' : '<span class="text-glow-green">Operando</span>';

            updateValue('balance', data.balance);
            updateValue('margin', data.available_margin);
            
            updateValue('session-pnl', data.session_pnl, `text-2xl font-medium mt-2 ${data.session_pnl_num >= 0 ? 'text-glow-green' : 'text-glow-red'}`);
            updateValue('drawdown', data.max_drawdown, `text-2xl font-medium mt-2 ${data.max_drawdown_num < 0 ? 'text-glow-red' : 'text-slate-200'}`);

            // CEO metrics
            const ceo = data.ceo || {};
            const asMoneyClass = (v) => v >= 0 ? 'text-glow-green' : 'text-glow-red';
            const pnl7 = Number(ceo.pnl_7d_num || 0);
            const pnl30 = Number(ceo.pnl_30d_num || 0);
            updateValue('ceo-pnl-7d', ceo.pnl_7d || '--', `text-lg font-semibold ${asMoneyClass(pnl7)}`);
            updateValue('ceo-winrate-7d', ceo.winrate_7d || '--', 'text-lg font-semibold text-slate-100');
            updateValue('ceo-pnl-30d', ceo.pnl_30d || '--', `text-lg font-semibold ${asMoneyClass(pnl30)}`);
            updateValue('ceo-pf-30d', ceo.profit_factor_30d || '--', 'text-lg font-semibold text-slate-100');
            document.getElementById('ceo-updated').innerText = `Actualizado: ${ceo.last_updated_local || '--'}`;

            const symbolsEl = document.getElementById('ceo-symbols');
            const symbols = ceo.symbols_month || [];
            if (symbols.length === 0) {
                symbolsEl.innerHTML = '<div class="text-slate-500">Sin datos</div>';
            } else {
                symbolsEl.innerHTML = symbols.slice(0, 6).map(s => `
                    <div class="flex justify-between">
                        <span class="text-slate-300">${s.symbol}</span>
                        <span class="${Number(s.pnl_total) >= 0 ? 'text-emerald-400' : 'text-red-400'}">${s.pnl_label}</span>
                    </div>
                `).join('');
            }

            const tradesEl = document.getElementById('ceo-trades');
            const trades = ceo.recent_trades || [];
            ceoTradeRows = trades;
            const filteredTrades = filterCeoTrades(ceoTradeRows);
            const renderCount = Math.min(ceoTradeVisibleCount, filteredTrades.length);
            if (filteredTrades.length === 0) {
                tradesEl.innerHTML = '<div class="text-slate-500">Sin trades cerrados</div>';
            } else {
                tradesEl.innerHTML = filteredTrades.slice(0, renderCount).map(t => `
                    <div class="flex justify-between">
                        <span class="text-slate-300">${t.symbol} <span class="text-slate-500">(${t.exit_time_local})</span></span>
                        <span class="${Number(t.pnl_num) >= 0 ? 'text-emerald-400' : 'text-red-400'}">${t.pnl}</span>
                    </div>
                `).join('');
            }
            updateCeoFilterButtons();
            const moreBtn = document.getElementById('ceo-trades-more');
            if (filteredTrades.length > renderCount) {
                moreBtn.classList.remove('hidden');
                moreBtn.innerText = `Cargar más (${filteredTrades.length - renderCount})`;
            } else {
                moreBtn.classList.add('hidden');
            }

            // Render Market Cards
            const marketContainer = document.getElementById('market-cards');
            let cardsHtml = '';
            
            data.market.forEach(item => {
                let actionBadge = '';
                if (item.action.includes('Gestionando')) {
                    actionBadge = '<span class="bg-sky-500/20 text-sky-400 border border-sky-500/30 px-2 py-1 rounded text-xs font-semibold tracking-wide">ACTIVA</span>';
                } else if (item.action.includes('Comprar')) {
                    actionBadge = '<span class="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-1 rounded text-xs font-semibold tracking-wide">COMPRAR</span>';
                } else {
                    actionBadge = '<span class="bg-slate-700/50 text-slate-400 border border-slate-600 px-2 py-1 rounded text-xs font-semibold tracking-wide">ESPERAR</span>';
                }

                let priceClass = item.has_position ? 'text-white' : 'text-slate-200';
                
                let mlColor = item.ml_conf.replace('%','') > 55 ? 'bg-emerald-400' : 'bg-sky-400';
                let rsiNum = parseFloat(item.rsi);
                let rsiColor = rsiNum > 70 ? 'bg-red-400' : (rsiNum < 30 ? 'bg-emerald-400' : 'bg-slate-400');
                
                cardsHtml += `
                    <div class="glass-card rounded-3xl p-6 transition-all relative overflow-hidden" style="border-color: ${item.has_position ? 'rgba(14,165,233,0.3)' : ''}">
                        ${item.has_position ? '<div class="absolute top-0 left-0 w-full h-1 bg-sky-500 shadow-[0_0_10px_rgba(14,165,233,0.8)]"></div>' : ''}
                        
                        <div class="flex justify-between items-start mb-4">
                            <h3 class="text-xl font-bold tracking-tight text-white">${item.symbol}</h3>
                            ${actionBadge}
                        </div>
                        
                        <div class="mb-6">
                            <div class="text-3xl font-light tracking-tight ${priceClass} font-mono">${item.price}</div>
                            <div class="text-sm font-medium mt-1 ${item.change_num >= 0 ? 'text-emerald-400' : 'text-red-400'}">
                                ${item.change_num >= 0 ? '↗' : '↘'} ${item.change} <span class="text-slate-500 font-normal text-xs ml-1">24h</span>
                            </div>
                        </div>
                        
                        <div class="space-y-4 mb-2">
                            <div>
                                <div class="flex justify-between text-xs mb-1.5">
                                    <span class="text-slate-400">Confianza IA</span>
                                    <span class="text-white font-medium">${item.ml_conf}</span>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill ${mlColor}" style="width: ${item.ml_conf}"></div>
                                </div>
                            </div>
                            
                            <div>
                                <div class="flex justify-between text-xs mb-1.5">
                                    <span class="text-slate-400">RSI (Fuerza)</span>
                                    <span class="text-white font-medium">${item.rsi}</span>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill ${rsiColor}" style="width: ${rsiNum}%"></div>
                                </div>
                            </div>
                            
                            <div class="flex justify-between text-xs pt-2">
                                <span class="text-slate-400">Tendencia (HTF)</span>
                                <span class="text-white font-medium">${item.trend}</span>
                            </div>
                        </div>
                `;

                if (item.has_position) {
                    let pnlColor = item.unrealized_pnl_num >= 0 ? 'text-glow-green' : 'text-glow-red';
                    cardsHtml += `
                        <div class="mt-5 bg-black/40 rounded-2xl p-4 border border-sky-500/20">
                            <div class="text-[10px] uppercase tracking-widest text-sky-400/80 mb-2 font-semibold">Trade Activo</div>
                            <div class="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-1 sm:gap-0">
                                <div>
                                    <div class="text-xs text-slate-400 mb-0.5">${item.position_str}</div>
                                    <div class="text-xs text-slate-500">${item.unrealized_pnl_label}</div>
                                </div>
                                <div class="${pnlColor} font-bold text-lg font-mono tracking-tight">${item.unrealized_pnl}</div>
                            </div>
                        </div>
                    `;
                }

                cardsHtml += `</div>`;
            });
            
            marketContainer.innerHTML = cardsHtml;

            // Render Events
            const eventsContainer = document.getElementById('events-log');
            if (data.events.length > 0) {
                let eventsHtml = '';
                data.events.forEach((e, i) => {
                    eventsHtml += `
                        <div class="flex items-start gap-4">
                            <div class="mt-1">
                                <div class="w-2 h-2 rounded-full ${i === 0 ? 'bg-sky-400 shadow-[0_0_8px_rgba(56,189,248,0.8)]' : 'bg-slate-600'}"></div>
                                ${i !== data.events.length - 1 ? '<div class="w-[1px] h-full bg-white/5 mx-auto mt-1"></div>' : ''}
                            </div>
                            <div class="text-sm text-slate-300 pb-4">${e}</div>
                        </div>
                    `;
                });
                eventsContainer.innerHTML = eventsHtml;
            } else {
                eventsContainer.innerHTML = '<div class="text-slate-500 text-sm py-2">Sin eventos en la línea de tiempo.</div>';
            }
        }

        fetchState();
        setInterval(fetchState, 1000);
        document.getElementById('ceo-trades-more').addEventListener('click', () => {
            ceoTradeVisibleCount += 20;
            render({ ...window.__lastData, ceo: { ...(window.__lastData?.ceo || {}), recent_trades: ceoTradeRows } });
        });
        document.querySelectorAll('#ceo-trades-filters button[data-filter]').forEach((btn) => {
            btn.addEventListener('click', () => {
                ceoTradeFilter = btn.dataset.filter || 'all';
                ceoTradeVisibleCount = 10;
                render({ ...window.__lastData, ceo: { ...(window.__lastData?.ceo || {}), recent_trades: ceoTradeRows } });
            });
        });
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
    report_tz = os.environ.get("REPORT_TIMEZONE", "America/Bogota")
    ceo_cache: dict[str, Any] = {"expires_mono": 0.0, "data": None}

    async def build_ceo_snapshot() -> dict[str, Any]:
        now_mono = asyncio.get_event_loop().time()
        if ceo_cache["data"] is not None and now_mono < float(ceo_cache["expires_mono"]):
            return ceo_cache["data"]
        week = await db.fetch_period_summary("week", tz_name=report_tz)
        month = await db.fetch_period_summary("month", tz_name=report_tz)
        symbols = await db.fetch_symbol_performance("month", tz_name=report_tz)
        recent = await db.fetch_recent_closed_trades(limit=250)
        tz_obj = ZoneInfo(report_tz)
        recent_rows: list[dict[str, Any]] = []
        for row in recent:
            exit_time = row.get("exit_time")
            if isinstance(exit_time, datetime):
                exit_local = exit_time.astimezone(tz_obj).strftime("%m-%d %H:%M")
            else:
                exit_local = "--"
            pnl_num = float(row.get("pnl_net") if row.get("pnl_net") is not None else (row.get("pnl") or 0.0))
            recent_rows.append(
                {
                    "symbol": str(row.get("symbol", "-")),
                    "pnl": f"{pnl_num:+.2f} USDT",
                    "pnl_num": pnl_num,
                    "exit_time_local": exit_local,
                    "exit_time_iso": exit_time.isoformat() if isinstance(exit_time, datetime) else "",
                }
            )
        ceo_payload = {
            "pnl_7d": f"{float(week['pnl_total']):+,.2f} USDT",
            "pnl_7d_num": float(week["pnl_total"]),
            "winrate_7d": f"{float(week['winrate']):.1f}%",
            "trades_7d": int(week["total_trades"]),
            "pnl_30d": f"{float(month['pnl_total']):+,.2f} USDT",
            "pnl_30d_num": float(month["pnl_total"]),
            "profit_factor_30d": (
                "N/A (sin pérdidas)"
                if int(month["total_trades"]) > 0 and int(month["losses"]) == 0 and int(month["wins"]) > 0
                else "N/A"
                if int(month["total_trades"]) == 0
                else f"{float(month['profit_factor']):.2f}"
            ),
            "symbols_month": [
                {
                    "symbol": s["symbol"],
                    "pnl_total": float(s["pnl_total"]),
                    "pnl_label": f"{float(s['pnl_total']):+,.2f} USDT",
                }
                for s in symbols
            ],
            "recent_trades": recent_rows,
            "last_updated_local": datetime.now(tz=tz_obj).strftime("%Y-%m-%d %H:%M:%S"),
        }
        ceo_cache["data"] = ceo_payload
        ceo_cache["expires_mono"] = now_mono + 45.0
        return ceo_payload
    
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
        mt5_profit_by_ticket: dict[int, float] = {}

        # MT5 live mode: prefer broker-reported unrealized PnL so dashboard matches terminal.
        get_positions = getattr(paper_executor, "get_open_positions", None)
        if callable(get_positions):
            try:
                mt5_positions = get_positions()
                if isinstance(mt5_positions, list):
                    mt5_profit_by_ticket = {
                        int(p.get("ticket")): float(p.get("profit", 0.0))
                        for p in mt5_positions
                        if p.get("ticket") is not None
                    }
            except Exception:
                mt5_profit_by_ticket = {}
        
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
                mt5_ticket = getattr(pos, "mt5_position_ticket", None)
                if isinstance(mt5_ticket, int) and mt5_ticket in mt5_profit_by_ticket:
                    unrl = mt5_profit_by_ticket[mt5_ticket]
                else:
                    # Fallback for paper mode or when ticket/profit is unavailable.
                    qty = pos.position_size / pos.entry_price
                    unrl = (price - pos.entry_price) * qty
                pos_str = f"Comprado en {pos.entry_price:,.2f}"
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
                "unrealized_pnl_label": "Ganancia flotante" if unrl >= 0 else "Pérdida flotante",
                "unrealized_pnl_num": unrl
            })

        # Events
        events = list(dash_state.dashboard_events)[-3:]
        # Strip rich tags like [dim], [red], etc for HTML
        import re
        clean_events = [re.sub(r'\[.*?\]', '', e) for e in events]

        ceo = await build_ceo_snapshot()
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
            "events": clean_events,
            "ceo": ceo,
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
