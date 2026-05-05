import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from aiohttp import web

from bot import state as dash_state
from bot.constants import DEBUG_LOG_HINT
from bot.dashboard_helpers import (
    compute_rsi,
    dashboard_rsi_label,
    dashboard_rsi_timeframe,
    mt5_dashboard_mark,
    pct_change_24h_vs_h1,
)
from database.db_manager import db
from execution.mt5_executor import MT5Executor
from execution.paper_executor import PaperExecutor, compute_dynamic_tp_hint
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD

logger = logging.getLogger(__name__)


def _dashboard_client_ip(request: web.Request) -> str:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote or "unknown"


def _make_api_rate_middleware(per_minute: int, window_s: float = 60.0):
    """Return aiohttp middleware limiting /api/state requests per client IP."""
    hits: dict[str, list[float]] = {}

    @web.middleware
    async def _middleware(request: web.Request, handler):
        if per_minute <= 0 or request.path != "/api/state":
            return await handler(request)
        ip = _dashboard_client_ip(request)
        now = asyncio.get_running_loop().time()
        bucket = hits.setdefault(ip, [])
        cutoff = now - window_s
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if not bucket:
            hits.pop(ip, None)
            bucket = hits.setdefault(ip, [])
        if len(bucket) >= per_minute:
            return web.json_response(
                {"error": "rate_limited", "retry_after_s": int(window_s)},
                status=429,
                headers={"Retry-After": str(int(window_s))},
            )
        bucket.append(now)
        return await handler(request)

    return _middleware


def _htf_trend_letters(t15: str, t1h: str, t4h: str) -> str:
    def _one(v: str) -> str:
        if v == "bullish":
            return "B"
        if v == "bearish":
            return "S"
        return "N"

    return f"{_one(t15)}/{_one(t1h)}/{_one(t4h)}"


def _fallback_ceo_payload(report_tz: str) -> dict[str, Any]:
    """Return safe CEO block when DB is unavailable."""
    try:
        tz_obj = ZoneInfo(report_tz)
    except Exception:
        tz_obj = timezone.utc  # type: ignore[assignment]
    return {
        "pnl_7d": "+0.00 USDT",
        "pnl_7d_num": 0.0,
        "winrate_7d": "0.0%",
        "trades_7d": 0,
        "pnl_30d": "+0.00 USDT",
        "pnl_30d_num": 0.0,
        "profit_factor_30d": "N/A",
        "symbols_month": [],
        "recent_trades": [],
        "last_updated_local": datetime.now(tz=tz_obj).strftime("%Y-%m-%d %H:%M:%S"),
    }


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
    <div class="max-w-7xl mx-auto space-y-6">
        
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
                    <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Estrategia</div>
                    <div class="text-lg font-bold text-sky-300" id="sentiment">—</div>
                    <div class="text-xs text-slate-400 mt-1 leading-snug" id="sentiment-detail">—</div>
                </div>
                <div class="bg-black/20 p-4 rounded-2xl border border-white/5 flex-1 md:w-40">
                    <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Estado Bot</div>
                    <div class="text-lg font-bold" id="bot-status">--</div>
                </div>
                <div class="bg-black/20 p-4 rounded-2xl border border-white/5 flex-1 md:w-40">
                    <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Posiciones</div>
                    <div class="text-lg font-bold text-sky-400" id="pos-count">--</div>
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

        <!-- Session Performance -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass-card rounded-2xl p-5 text-center">
                <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-2">Win / Loss</div>
                <div class="text-2xl font-bold"><span class="text-emerald-400" id="s-wins">0</span><span class="text-slate-600"> / </span><span class="text-red-400" id="s-losses">0</span></div>
            </div>
            <div class="glass-card rounded-2xl p-5 text-center">
                <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-2">Winrate Sesión</div>
                <div class="text-2xl font-bold text-slate-100" id="s-winrate">--%</div>
            </div>
            <div class="glass-card rounded-2xl p-5 text-center">
                <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-2">Latencia API</div>
                <div class="text-2xl font-bold text-sky-400" id="api-latency">--</div>
            </div>
            <div class="glass-card rounded-2xl p-5 text-center">
                <div class="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-2">Umbral compra</div>
                <div class="text-2xl font-bold text-slate-200" id="sent-num">70%</div>
                <div class="progress-bar-bg mt-2"><div class="progress-bar-fill bg-emerald-500/60" id="sent-bar" style="width:70%"></div></div>
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
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full" id="market-cards">
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
            const pair = data.primary_pair || (Array.isArray(data.watchlist) && data.watchlist[0]) || '—';
            sentEl.innerText = pair.replace('/', ' / ');
            sentEl.className = 'text-lg font-bold text-sky-300';
            document.getElementById('sentiment-detail').innerText = data.strategy_blurb || '';

            const botStatusEl = document.getElementById('bot-status');
            if (data.global_hold) {
                botStatusEl.innerHTML = '<span class="text-glow-red">⛔ Pausa</span>';
            } else if (data.open_count > 0) {
                botStatusEl.innerHTML = '<span class="text-sky-400">⚡ Operando</span>';
            } else {
                botStatusEl.innerHTML = '<span class="text-glow-green">✅ Listo</span>';
            }

            updateValue('pos-count', `${data.open_count || 0}/${data.max_positions || 3}`);
            updateValue('s-wins', data.session_wins || 0);
            updateValue('s-losses', data.session_losses || 0);
            updateValue('s-winrate', data.session_winrate || '--%');

            const th = Number.isFinite(data.buy_prob_threshold_pct) ? data.buy_prob_threshold_pct : 70;
            const sentEl2 = document.getElementById('sent-num');
            if (sentEl2) {
                sentEl2.innerText = th + '%';
                sentEl2.className = 'text-2xl font-bold text-slate-200';
            }
            const sentBar = document.getElementById('sent-bar');
            if (sentBar) sentBar.style.width = Math.min(100, Math.max(0, th)) + '%';

            const latEl = document.getElementById('api-latency');
            if (latEl) {
                const ms = Number(data.api_latency_ms || 0);
                latEl.innerText = ms > 0 ? (Math.round(ms) + ' ms') : '—';
            }

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
            
            const buyTh = Number.isFinite(data.buy_prob_threshold_pct) ? data.buy_prob_threshold_pct : 70;
            data.market.forEach(item => {
                let actionBadge = '';
                if (item.action.includes('Gestionando')) {
                    actionBadge = '<span class="bg-sky-500/20 text-sky-400 border border-sky-500/30 px-2 py-1 rounded text-xs font-semibold tracking-wide">ACTIVA</span>';
                } else if (item.can_enter === true) {
                    actionBadge = '<span class="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-1 rounded text-xs font-semibold tracking-wide">COMPRAR</span>';
                } else {
                    actionBadge = '<span class="bg-slate-700/50 text-slate-400 border border-slate-600 px-2 py-1 rounded text-xs font-semibold tracking-wide">ESPERAR</span>';
                }

                let priceClass = item.has_position ? 'text-white' : 'text-slate-200';
                
                const mlPct = Math.min(100, Math.max(0, parseFloat(String(item.ml_conf).replace(/[%\\s]/g, '')) || 0));
                let mlColor = mlPct >= buyTh ? 'bg-emerald-400' : 'bg-sky-400';
                const rsiRaw = item.rsi;
                const rsiNum = (rsiRaw === '--' || rsiRaw === undefined) ? NaN : parseFloat(rsiRaw);
                let rsiColor = (!Number.isFinite(rsiNum)) ? 'bg-slate-600' : (rsiNum > 70 ? 'bg-red-400' : (rsiNum < 30 ? 'bg-emerald-400' : 'bg-slate-400'));
                const rsiBarW = Number.isFinite(rsiNum) ? Math.min(100, Math.max(0, rsiNum)) : 0;
                const symTitle = item.symbol_pair ? String(item.symbol_pair).replace('/', ' / ') : item.symbol;
                const chOk = item.change_ok === true;
                const chClass = !chOk ? 'text-slate-500' : (item.change_num >= 0 ? 'text-emerald-400' : 'text-red-400');
                const chGlyph = !chOk ? '•' : (item.change_num >= 0 ? '↗' : '↘');
                const rsiLbl = item.rsi_label || 'RSI (14)';
                
                cardsHtml += `
                    <div class="glass-card rounded-3xl p-6 transition-all relative overflow-hidden" style="border-color: ${item.has_position ? 'rgba(14,165,233,0.3)' : ''}">
                        ${item.has_position ? '<div class="absolute top-0 left-0 w-full h-1 bg-sky-500 shadow-[0_0_10px_rgba(14,165,233,0.8)]"></div>' : ''}
                        
                        <div class="flex justify-between items-start mb-4 gap-2">
                            <div>
                                <h3 class="text-xl font-bold tracking-tight text-white">${symTitle}</h3>
                                <div class="text-[10px] text-slate-500 mt-0.5 font-mono">${item.symbol}</div>
                            </div>
                            ${actionBadge}
                        </div>
                        
                        <div class="mb-6">
                            <div class="text-3xl font-light tracking-tight ${priceClass} font-mono">${item.price}</div>
                            <div class="text-sm font-medium mt-1 ${chClass}">
                                ${chGlyph} ${item.change} <span class="text-slate-500 font-normal text-xs ml-1">${chOk ? '≈24h (vs H1)' : 'sin datos H1'}</span>
                            </div>
                        </div>
                        
                        <div class="space-y-4 mb-2">
                            <div>
                                <div class="flex justify-between text-xs mb-1.5">
                                    <span class="text-slate-400">Prob. modelo (XGB)</span>
                                    <span class="text-white font-medium">${item.ml_conf}</span>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill ${mlColor}" style="width: ${mlPct}%"></div>
                                </div>
                            </div>
                            
                            <div>
                                <div class="flex justify-between text-xs mb-1.5">
                                    <span class="text-slate-400">${rsiLbl}</span>
                                    <span class="text-white font-medium">${item.rsi}</span>
                                </div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill ${rsiColor}" style="width: ${rsiBarW}%"></div>
                                </div>
                            </div>
                            
                            <div class="flex justify-between text-xs pt-2">
                                <span class="text-slate-400">Tendencia 15m/1h/4h</span>
                                <span class="text-white font-medium">${item.trend}</span>
                            </div>
                        </div>
                `;
                if (!item.has_position) {
                    cardsHtml += `
                        <div class="mt-2 pt-2 border-t border-white/5 space-y-1">
                            <div class="flex justify-between text-[10px]">
                                <span class="text-slate-500">Señal estrategia</span>
                                <span class="text-slate-300 font-mono">${item.strategy_signal}</span>
                            </div>
                            ${item.entry_hint ? `<div class="text-[10px] text-slate-500 leading-snug">${item.entry_hint}</div>` : ''}
                        </div>`;
                }

                if (item.has_position) {
                    let pnlColor = item.unrealized_pnl_num >= 0 ? 'text-glow-green' : 'text-glow-red';
                    const slVal = item.stop_loss || '—';
                    const tpVal = item.take_profit || '—';
                    const trailBadge = item.trailing_active ? '<span class="text-[10px] text-amber-400/90 ml-2">Trailing ON</span>' : '';
                    cardsHtml += `
                        <div class="mt-5 bg-black/40 rounded-2xl p-4 border border-sky-500/20">
                            <div class="text-[10px] uppercase tracking-widest text-sky-400/80 mb-2 font-semibold">Trade Activo${trailBadge}</div>
                            <div class="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-1 sm:gap-0">
                                <div>
                                    <div class="text-xs text-slate-400 mb-0.5">${item.position_str}</div>
                                    <div class="text-xs text-slate-500">${item.unrealized_pnl_label}</div>
                                </div>
                                <div class="${pnlColor} font-bold text-lg font-mono tracking-tight">${item.unrealized_pnl}</div>
                            </div>
                            <div class="grid grid-cols-2 gap-3 mt-3 pt-3 border-t border-white/5 text-xs">
                                <div>
                                    <div class="text-slate-500 mb-0.5">SL activo (bot)</div>
                                    <div class="font-mono text-slate-100">${slVal}</div>
                                </div>
                                <div>
                                    <div class="text-slate-500 mb-0.5">TP objetivo (hint)</div>
                                    <div class="font-mono text-emerald-400/90">${tpVal}</div>
                                </div>
                                <div>
                                    <div class="text-slate-500 mb-0.5">Pico máximo</div>
                                    <div class="font-mono text-sky-200/90">${item.peak_price || '—'}</div>
                                </div>
                                <div class="col-span-2">
                                    <div class="text-slate-500 mb-0.5">Estado trailing</div>
                                    <div class="text-slate-300 leading-snug">${item.trail_progress || '—'}</div>
                                </div>
                                <div>
                                    <div class="text-slate-500 mb-0.5">SL en MT5</div>
                                    <div class="font-mono text-amber-200/90">${item.broker_stop_loss || '—'}</div>
                                </div>
                                <div>
                                    <div class="text-slate-500 mb-0.5">TP en MT5</div>
                                    <div class="font-mono text-amber-200/90">${item.broker_take_profit || '—'}</div>
                                </div>
                            </div>
                            <div class="text-[10px] text-slate-600 mt-2">El bot sube el SL con el pico (tras umbral). TP hint sube con el pico; MT5 puede ir un tick atrás hasta el próximo sync.</div>
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
        setInterval(fetchState, 2500);
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
    host: str = "0.0.0.0",  # nosec B104
    port: int = 8080,
):
    """Run an aiohttp web server providing a modern real-time mobile dashboard."""
    report_tz = os.environ.get("REPORT_TIMEZONE", "America/Bogota")
    ceo_cache: dict[str, Any] = {"expires_mono": 0.0, "data": None}
    api_mt5_sync_last_mono = 0.0
    # Min seconds between full MT5↔memory syncs on /api/state. 0 = sync every request
    # (matches SPA poll ~2.5s ⇒ dashboard rarely >~3s stale vs broker).
    _raw_api_iv = os.environ.get(
        "WEB_API_MT5_SYNC_MIN_INTERVAL_S",
        os.environ.get("WEB_API_MT5_SYNC_EVERY_S", "0"),
    ).strip()
    try:
        api_mt5_sync_iv = float(_raw_api_iv) if _raw_api_iv else 0.0
    except ValueError:
        api_mt5_sync_iv = 0.0

    async def build_ceo_snapshot() -> dict[str, Any]:
        now_mono = asyncio.get_event_loop().time()
        if ceo_cache["data"] is not None and now_mono < float(ceo_cache["expires_mono"]):
            return ceo_cache["data"]
        try:
            week = await db.fetch_period_summary("week", tz_name=report_tz)
            month = await db.fetch_period_summary("month", tz_name=report_tz)
            symbols = await db.fetch_symbol_performance("month", tz_name=report_tz)
            recent = await db.fetch_recent_closed_trades(limit=250)
        except Exception as exc:  # noqa: BLE001
            # Keep API alive while DB reconnects/shuts down.
            logger.warning("Web CEO snapshot fallback (DB unavailable): %s", exc)
            fallback = _fallback_ceo_payload(report_tz)
            ceo_cache["data"] = fallback
            ceo_cache["expires_mono"] = now_mono + 5.0
            return fallback
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
        nonlocal api_mt5_sync_last_mono
        now_utc = datetime.now(tz=timezone.utc)

        if isinstance(paper_executor, MT5Executor) and paper_executor._live:
            now_mono = asyncio.get_running_loop().time()
            if api_mt5_sync_iv <= 0 or (
                now_mono - api_mt5_sync_last_mono
            ) >= api_mt5_sync_iv:
                api_mt5_sync_last_mono = now_mono
                try:
                    await paper_executor.sync_positions_with_exchange()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Web /api/state MT5 sync failed: %s %s",
                        exc,
                        DEBUG_LOG_HINT,
                    )
        
        # Uptime
        if dash_state.bot_start_time is not None:
            elapsed = now_utc - dash_state.bot_start_time
            total_secs = int(elapsed.total_seconds())
            h, rem = divmod(total_secs, 3600)
            m, s = divmod(rem, 60)
            uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            uptime_str = "00:00:00"

        sentiment = state.get("sentiment")
        global_hold = False
        sentiment_label = "INJ/USDT ML-only"
        sentiment_detail = "Sin Gemini / noticias — solo XGB ≥ 70%"

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
        ml_signals = state.get("ml_signals", {})
        htf_trend = state.get("htf_trend", {})
        mt5_profit_by_ticket: dict[int, float] = {}
        mt5_by_ticket: dict[int, dict[str, Any]] = {}
        mt5_symbol_open_any: set[str] = set()
        mt5_symbol_open_bot: set[str] = set()

        # MT5 live mode: prefer broker-reported unrealized PnL so dashboard matches terminal.
        get_positions = getattr(paper_executor, "get_open_positions", None)
        if callable(get_positions):
            try:
                mt5_positions = get_positions(include_foreign=True)
                if isinstance(mt5_positions, list):
                    for p in mt5_positions:
                        t = p.get("ticket")
                        if t is None:
                            continue
                        tid = int(t)
                        mt5_profit_by_ticket[tid] = float(p.get("profit", 0.0))
                        mt5_by_ticket[tid] = p
                        br_sym = str(p.get("symbol", ""))
                        local_sym = paper_executor._local_symbol_from_broker(br_sym) if hasattr(paper_executor, "_local_symbol_from_broker") else br_sym
                        if local_sym:
                            mt5_symbol_open_any.add(local_sym)
                            if int(p.get("magic", 0) or 0) == int(getattr(paper_executor, "_magic", -1)):
                                mt5_symbol_open_bot.add(local_sym)
            except Exception:
                mt5_profit_by_ticket = {}
                mt5_by_ticket = {}
                mt5_symbol_open_any = set()
                mt5_symbol_open_bot = set()
        
        for sym in watchlist:
            prices = list(state["prices"].get(sym, []))
            candle_last = prices[-1] if prices else None
            price = mt5_dashboard_mark(state, sym, candle_last)
            
            if price is None:
                continue

            h1_hist = list(state.get("htf_closes", {}).get(sym, {}).get("1h", []) or [])
            pct = pct_change_24h_vs_h1(h1_hist, float(price))

            rsi_series = list(state.get("dashboard_rsi_closes", {}).get(sym, []) or [])
            rsi = compute_rsi(rsi_series) if len(rsi_series) >= 15 else None
            prob = ml_probs.get(sym, 0.0)
            t15 = htf_trend.get(sym, {}).get("15m", "neutral")
            t1h = htf_trend.get(sym, {}).get("1h", "neutral")
            t4h = htf_trend.get(sym, {}).get("4h", "neutral")

            pos = paper_executor.open_positions.get(sym)
            has_pos_local = bool(pos)
            has_pos_broker = sym in mt5_symbol_open_any
            has_pos = has_pos_local or has_pos_broker
            unrl = 0.0
            pos_str = ""
            sl_display = ""
            tp_display = ""
            trailing_active = False
            sl_num: float | None = None
            tp_num: float | None = None
            peak_display = "—"
            trail_progress = "—"
            broker_sl_display = "—"
            broker_tp_display = "—"
            strategy_signal = "HOLD"
            can_buy = False
            entry_hint = ""

            if has_pos_local and pos is not None:
                mt5_ticket = getattr(pos, "mt5_position_ticket", None)
                if isinstance(mt5_ticket, int) and mt5_ticket in mt5_profit_by_ticket:
                    unrl = mt5_profit_by_ticket[mt5_ticket]
                else:
                    # Fallback for paper mode or when ticket/profit is unavailable.
                    qty = pos.position_size / pos.entry_price
                    unrl = (price - pos.entry_price) * qty
                pos_str = f"Comprado en {pos.entry_price:,.2f}"
                action = "Gestionando posición"
                trailing_active = bool(getattr(pos, "trailing_stop_active", False))
                sl_num = float(pos.current_stop_loss)
                sl_display = f"{sl_num:,.2f}"
                peak_display = f"{pos.peak_price:,.2f}"
                if pos.entry_price and pos.entry_price > 0:
                    pcp = (float(price) - pos.entry_price) / pos.entry_price
                    if trailing_active:
                        trail_progress = f"Trailing ON (pico {peak_display})"
                    else:
                        trail_progress = (
                            f"Beneficio {pcp * 100:.2f}% → activar ≥ {pos.activation_pct * 100:.2f}%"
                        )
                tp_raw = compute_dynamic_tp_hint(pos)
                if tp_raw is not None and tp_raw > 0:
                    tp_num = float(tp_raw)
                    tp_display = f"{tp_num:,.2f}"
                else:
                    tp_display = "—"
                if isinstance(mt5_ticket, int) and mt5_ticket in mt5_by_ticket:
                    br = mt5_by_ticket[mt5_ticket]
                    slb = float(br.get("sl") or 0.0)
                    tpb = float(br.get("tp") or 0.0)
                    broker_sl_display = f"{slb:,.2f}" if slb > 0.0 else "—"
                    broker_tp_display = f"{tpb:,.2f}" if tpb > 0.0 else "—"
            elif has_pos_broker:
                pos_str = "Abierta en broker (manual/externa)"
                action = "Gestionando en broker"
                strategy_signal = "MANUAL/BROKER"
                trail_progress = "No gestionada por libro local"
                br_positions = [x for x in mt5_by_ticket.values() if (paper_executor._local_symbol_from_broker(str(x.get("symbol", ""))) if hasattr(paper_executor, "_local_symbol_from_broker") else str(x.get("symbol", ""))) == sym]
                if br_positions:
                    p0 = br_positions[0]
                    entry_br = float(p0.get("price_open", 0.0) or 0.0)
                    if entry_br > 0:
                        pos_str = f"Broker @ {entry_br:,.2f}"
                    unrl = float(p0.get("profit", 0.0) or 0.0)
                    slb = float(p0.get("sl") or 0.0)
                    tpb = float(p0.get("tp") or 0.0)
                    broker_sl_display = f"{slb:,.2f}" if slb > 0.0 else "—"
                    broker_tp_display = f"{tpb:,.2f}" if tpb > 0.0 else "—"
            else:
                strategy_signal = ml_signals.get(sym, "HOLD")
                can_buy = (
                    not global_hold
                    and prob >= BUY_PROB_THRESHOLD
                    and strategy_signal == "BUY"
                )
                action = "Comprar" if can_buy else "Esperar"
                if global_hold:
                    entry_hint = "Pausa global — no se abren entradas."
                elif prob >= BUY_PROB_THRESHOLD and strategy_signal != "BUY":
                    entry_hint = "Prob. ≥ umbral pero símbolo no operable o modelo en HOLD."
                elif prob < BUY_PROB_THRESHOLD:
                    entry_hint = (
                        f"Prob. modelo por debajo del umbral de compra "
                        f"({int(BUY_PROB_THRESHOLD * 100)}%)."
                    )
                else:
                    entry_hint = ""

            market_data.append({
                "symbol": sym.split("/")[0],
                "symbol_pair": sym,
                "price": f"{price:,.2f}",
                "change": (
                    f"{'+' if pct >= 0 else ''}{pct:.1f}%"
                    if pct is not None
                    else "—"
                ),
                "change_num": float(pct) if pct is not None else 0.0,
                "change_ok": pct is not None,
                "rsi": f"{rsi:.1f}" if rsi is not None else "--",
                "rsi_num": rsi,
                "rsi_label": dashboard_rsi_label(),
                "ml_conf": f"{prob*100:.0f}%",
                "trend": _htf_trend_letters(t15, t1h, t4h),
                "trend_detail": {"15m": t15, "1h": t1h, "4h": t4h},
                "action": action,
                "strategy_signal": strategy_signal if not has_pos else "—",
                "can_enter": can_buy if not has_pos else False,
                "entry_hint": entry_hint if not has_pos else "",
                "has_position": has_pos,
                "position_str": pos_str,
                "stop_loss": sl_display,
                "take_profit": tp_display,
                "stop_loss_num": sl_num,
                "take_profit_num": tp_num,
                "trailing_active": trailing_active,
                "peak_price": peak_display,
                "trail_progress": trail_progress,
                "broker_stop_loss": broker_sl_display,
                "broker_take_profit": broker_tp_display,
                "unrealized_pnl": f"{'+' if unrl >=0 else ''}{unrl:.2f} USDT",
                "unrealized_pnl_label": "Ganancia flotante" if unrl >= 0 else "Pérdida flotante",
                "unrealized_pnl_num": unrl
            })

        # Events
        events = list(dash_state.dashboard_events)[-6:]
        import re
        clean_events = [re.sub(r'\[.*?\]', '', e) for e in events]

        # Session win/loss stats
        wins = dash_state.session_wins
        losses = dash_state.session_losses
        total_trades_session = wins + losses
        winrate_session = (wins / total_trades_session * 100) if total_trades_session > 0 else 0.0

        try:
            ceo = await build_ceo_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web CEO snapshot failed unexpectedly: %s", exc)
            ceo = _fallback_ceo_payload(report_tz)
        _th_pct = int(BUY_PROB_THRESHOLD * 100)
        _pair = watchlist[0] if watchlist else ""
        resp = {
            "uptime": uptime_str,
            "global_hold": global_hold,
            "api_latency_ms": float(state.get("api_latency_ms", 0.0)),
            "watchlist": list(watchlist),
            "primary_pair": _pair,
            "buy_prob_threshold": BUY_PROB_THRESHOLD,
            "buy_prob_threshold_pct": _th_pct,
            "strategy_blurb": (
                f"Modelo XGB · umbral {_th_pct}% · RSI en velas {dashboard_rsi_timeframe()}"
            ),
            "dashboard_rsi_tf": dashboard_rsi_timeframe(),
            "balance": f"{balance:,.2f}",
            "available_margin": f"{avail:,.2f}",
            "session_pnl": f"{paper_executor.total_pnl:+.2f}",
            "session_pnl_num": paper_executor.total_pnl,
            "max_drawdown": f"{state.get('max_drawdown', 0.0):+.2f}",
            "max_drawdown_num": state.get("max_drawdown", 0.0),
            "open_count": len(mt5_symbol_open_any) if mt5_symbol_open_any else len(paper_executor.open_positions),
            "open_count_local": len(paper_executor.open_positions),
            "open_count_broker": len(mt5_symbol_open_any),
            "server_time_utc": now_utc.isoformat(),
            "max_positions": risk_manager.max_positions,
            "session_wins": wins,
            "session_losses": losses,
            "session_winrate": f"{winrate_session:.0f}%",
            "market": market_data,
            "events": clean_events,
            "ceo": ceo,
        }
        
        return web.json_response(resp)

    _rpm_raw = os.environ.get("DASH_API_RATE_PER_MIN", "180").strip()
    try:
        _rpm = max(0, int(_rpm_raw))
    except ValueError:
        _rpm = 180
    _middlewares = [_make_api_rate_middleware(_rpm)] if _rpm > 0 else []
    app = web.Application(middlewares=_middlewares)
    app.add_routes([
        web.get("/", handle_html),
        web.get("/api/state", handle_api)
    ])

    # ... existing aiohttp runner code ...
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        logger.info("🌐 Web dashboard server listening on http://%s:%d", host, port)
        await site.start()
        while True:
            await asyncio.sleep(3600)
    finally:
        await runner.cleanup()
