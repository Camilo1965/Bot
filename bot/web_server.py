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
from bot.monitoring import get_health_metrics
from database.db_manager import db
from execution.mt5_executor import MT5Executor
from execution.paper_executor import PaperExecutor, compute_dynamic_tp_hint
from risk.risk_manager import RiskManager
from strategy.ml_predictor import BUY_PROB_THRESHOLD

logger = logging.getLogger(__name__)


@web.middleware
async def _security_and_error_guard(request: web.Request, handler):
    """Suppress malformed requests and catch-all exceptions to avoid server crashes."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.warning(
            "Bad request suppressed from %s: %s (%s)",
            request.remote,
            type(exc).__name__,
            str(exc)[:200],
        )
        return web.Response(status=400, text="Bad Request")


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


def _extract_vibe_text(data: Any) -> str:
    """Extract readable text from a Vibe MCP tool result dict."""
    try:
        if isinstance(data, dict) and "content" in data:
            parts = []
            for item in data["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
            return " ".join(parts).strip()[:300]
        return str(data)[:300] if data else ""
    except Exception:
        return str(data)[:300] if data else ""


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
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root{--bg-base:#06060b;--bg-surface:#0d0d14;--bg-elevated:#13131d;--bg-hover:#1a1a28;--border:#1c1c2e;--border-active:#2d2d44;--text-primary:#eaeaf0;--text-secondary:#6b7280;--text-dim:#3a3a50;--accent:#00d4ff;--accent-dim:rgba(0,212,255,.12);--positive:#00ff88;--positive-dim:rgba(0,255,136,.1);--negative:#ff3366;--negative-dim:rgba(255,51,102,.1);--warning:#ffaa00;--warning-dim:rgba(255,170,0,.1);--vibe:#a855f7;--vibe-dim:rgba(168,85,247,.1)}
        *{box-sizing:border-box}body{background:var(--bg-base);color:var(--text-primary);font-family:'Inter',-apple-system,sans-serif;min-height:100vh;margin:0;padding:16px}
        .mono{font-family:'JetBrains Mono',monospace}
        .surface{background:var(--bg-surface);border:1px solid var(--border);border-radius:12px}
        .elevated{background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px}
        .badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:.05em;font-family:'JetBrains Mono',monospace}
        .badge-buy{background:var(--positive-dim);color:var(--positive);border:1px solid rgba(0,255,136,.2)}
        .badge-sell{background:var(--negative-dim);color:var(--negative);border:1px solid rgba(255,51,102,.2)}
        .badge-hold{background:var(--warning-dim);color:var(--warning);border:1px solid rgba(255,170,0,.2)}
        .badge-active{background:var(--accent-dim);color:var(--accent);border:1px solid rgba(0,212,255,.2)}
        .badge-wait{background:rgba(100,100,120,.1);color:var(--text-secondary);border:1px solid rgba(100,100,120,.2)}
        .badge-vibe{background:var(--vibe-dim);color:var(--vibe);border:1px solid rgba(168,85,247,.2)}
        .kpi-card{background:var(--bg-surface);border:1px solid var(--border);border-radius:10px;padding:16px 20px;transition:border-color .2s}
        .kpi-card:hover{border-color:var(--border-active)}
        .bar-track{height:6px;border-radius:3px;background:rgba(255,255,255,.06);overflow:hidden}
        .bar-fill{height:100%;border-radius:3px;transition:width .5s ease}
        .pulse-dot{width:8px;height:8px;border-radius:50%;display:inline-block;animation:pulse 2s infinite}
        @keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 6px currentColor}50%{opacity:.6;box-shadow:0 0 12px currentColor}}
        .value-up{animation:flash-green .8s ease-out}
        .value-down{animation:flash-red .8s ease-out}
        @keyframes flash-green{0%{color:var(--positive);text-shadow:0 0 8px var(--positive)}100%{color:inherit;text-shadow:none}}
        @keyframes flash-red{0%{color:var(--negative);text-shadow:0 0 8px var(--negative)}100%{color:inherit;text-shadow:none}}
        .mkt-row{transition:background .15s}
        .mkt-row:hover{background:var(--bg-hover)}
        .section-title{font-size:11px;text-transform:uppercase;letter-spacing:.1em;font-weight:600;color:var(--text-secondary);margin-bottom:12px}
        .vibe-item{padding:10px 12px;border-left:2px solid var(--vibe);background:var(--vibe-dim);border-radius:0 6px 6px 0}
        @media(max-width:768px){body{padding:8px}.hide-mobile{display:none}}
    </style>
</head>
<body>
    <div id="error-banner" class="hidden" style="position:fixed;top:0;left:0;right:0;z-index:50;background:var(--negative-dim);color:var(--negative);border-bottom:1px solid rgba(255,51,102,.3);padding:8px 16px;text-align:center;font-size:12px;font-weight:600"></div>
    <div class="max-w-7xl mx-auto space-y-5">
        <header class="surface p-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
            <div class="flex items-center gap-4">
                <h1 class="mono text-lg font-bold tracking-tight" style="color:var(--accent)">CLAWDBOT</h1>
                <span class="text-xs text-gray-500 font-medium tracking-wider">PRO</span>
            </div>
            <div class="flex items-center gap-5 text-xs text-gray-400">
                <span class="flex items-center gap-1.5"><span class="pulse-dot" id="dot-mt5" style="color:var(--positive)"></span>MT5</span>
                <span class="flex items-center gap-1.5"><span class="pulse-dot" id="dot-feed" style="color:var(--positive)"></span>Feed</span>
                <span class="flex items-center gap-1.5"><span class="pulse-dot" id="dot-vibe" style="color:var(--vibe)"></span>Vibe</span>
                <span class="mono font-medium" id="uptime">00:00:00</span>
            </div>
            <div class="flex gap-3">
                <div class="kpi-card text-center min-w-[100px]"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Estrategia</div><div class="mono text-sm font-semibold" style="color:var(--accent)" id="sentiment">—</div><div class="text-[10px] text-gray-500 mt-0.5 leading-snug" id="sentiment-detail">—</div></div>
                <div class="kpi-card text-center min-w-[90px]"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Estado</div><div class="text-sm font-bold" id="bot-status">—</div></div>
                <div class="kpi-card text-center min-w-[80px]"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Pos</div><div class="mono text-sm font-bold" id="pos-count">—</div></div>
            </div>
        </header>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div class="kpi-card"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Balance</div><div class="mono text-xl font-medium text-white" id="balance">—</div></div>
            <div class="kpi-card"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">PnL Sesión</div><div class="mono text-xl font-medium" id="session-pnl">—</div></div>
            <div class="kpi-card"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Max Drawdown</div><div class="mono text-xl font-medium" id="drawdown">—</div></div>
            <div class="kpi-card"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Win Rate</div><div class="flex items-baseline gap-2"><span class="mono text-xl font-bold text-white" id="s-winrate">0%</span><span class="text-xs text-gray-500"><span id="s-wins" style="color:var(--positive)">0</span>W / <span id="s-losses" style="color:var(--negative)">0</span>L</span></div></div>
            <div class="kpi-card"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold mb-1">Umbral</div><div class="mono text-xl font-bold text-white" id="sent-num">70%</div><div class="bar-track mt-1.5"><div class="bar-fill" id="sent-bar" style="width:70%;background:var(--positive)"></div></div></div>
        </div>
        <div class="surface p-5">
            <div class="flex flex-col md:flex-row gap-4 md:gap-8">
                <div class="flex-1"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Balance Total</div><div class="mono text-3xl font-light tracking-tight mt-1"><span class="text-gray-500">$</span><span id="balance-lg" class="font-medium text-white">—</span></div></div>
                <div class="flex-1"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Disponible</div><div class="mono text-2xl font-light mt-1"><span class="text-gray-500">$</span><span id="margin" class="font-medium text-white">—</span></div></div>
                <div class="flex-1"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Riesgo Port.</div><div class="mono text-lg font-medium mt-1" id="risk-pct">—</div></div>
                <div class="flex-1"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Latencia</div><div class="mono text-lg font-medium mt-1" style="color:var(--accent)" id="api-latency">—</div></div>
            </div>
        </div>
        <div>
            <div class="section-title px-1">MERCADO EN VIVO</div>
            <div class="surface overflow-x-auto">
                <table class="w-full text-sm">
                    <thead><tr class="border-b" style="border-color:var(--border)">
                        <th class="px-4 py-3 text-left text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Símbolo</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Precio</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold hide-mobile">24h</th>
                        <th class="px-4 py-3 text-center text-[10px] text-gray-500 uppercase tracking-widest font-semibold hide-mobile">RSI</th>
                        <th class="px-4 py-3 text-center text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Prob.</th>
                        <th class="px-4 py-3 text-center text-[10px] text-gray-500 uppercase tracking-widest font-semibold hide-mobile">Trend</th>
                        <th class="px-4 py-3 text-center text-[10px] text-gray-500 uppercase tracking-widest font-semibold">PnL</th>
                        <th class="px-4 py-3 text-center text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Señal</th>
                    </tr></thead>
                    <tbody id="market-tbody"></tbody>
                </table>
            </div>
        </div>
        <div id="positions-section" class="hidden">
            <div class="section-title px-1">POSICIONES ACTIVAS</div>
            <div class="surface overflow-x-auto">
                <table class="w-full text-sm">
                    <thead><tr class="border-b" style="border-color:var(--border)">
                        <th class="px-4 py-3 text-left text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Símbolo</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Entrada</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Actual</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold">SL</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold">TP</th>
                        <th class="px-4 py-3 text-right text-[10px] text-gray-500 uppercase tracking-widest font-semibold">PnL</th>
                        <th class="px-4 py-3 text-center text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Trailing</th>
                    </tr></thead>
                    <tbody id="pos-tbody"></tbody>
                </table>
            </div>
        </div>
        <div>
            <div class="section-title px-1 flex items-center gap-2"><span style="color:var(--vibe)">&#x1f52c;</span> VIBE-TRADING AI <span id="vibe-status-badge" class="badge badge-wait">Desconectado</span></div>
            <div class="surface p-5 space-y-3" id="vibe-content"><div class="text-gray-500 text-sm">Esperando datos del modelo de análisis…</div></div>
        </div>
        <div>
            <div class="section-title px-1">CEO SNAPSHOT</div>
            <div class="surface p-5">
                <div class="flex justify-between items-center mb-4"><span class="text-xs text-gray-500" id="ceo-updated">Actualizado: —</span></div>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div class="elevated p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">PnL 7D</div><div class="mono text-lg font-semibold" id="ceo-pnl-7d">—</div></div>
                    <div class="elevated p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Winrate 7D</div><div class="mono text-lg font-semibold text-white" id="ceo-winrate-7d">—</div></div>
                    <div class="elevated p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">PnL 30D</div><div class="mono text-lg font-semibold" id="ceo-pnl-30d">—</div></div>
                    <div class="elevated p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">PF 30D</div><div class="mono text-lg font-semibold text-white" id="ceo-pf-30d">—</div></div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div class="elevated p-3"><div class="text-xs text-gray-400 mb-2">Rendimiento por símbolo (mes)</div><div class="space-y-1.5 text-sm" id="ceo-symbols"></div></div>
                    <div class="elevated p-3">
                        <div class="text-xs text-gray-400 mb-2">Últimos trades cerrados</div>
                        <div class="flex flex-wrap gap-2 mb-2" id="ceo-trades-filters">
                            <button type="button" data-filter="today" class="px-2 py-1 text-xs rounded border text-gray-400" style="border-color:var(--border)">Hoy</button>
                            <button type="button" data-filter="7d" class="px-2 py-1 text-xs rounded border text-gray-400" style="border-color:var(--border)">7D</button>
                            <button type="button" data-filter="30d" class="px-2 py-1 text-xs rounded border text-gray-400" style="border-color:var(--border)">30D</button>
                            <button type="button" data-filter="all" class="px-2 py-1 text-xs rounded border" style="border-color:var(--accent);color:var(--accent)">Todo</button>
                        </div>
                        <div class="space-y-1.5 text-sm" id="ceo-trades"></div>
                        <button id="ceo-trades-more" class="mt-2 text-xs hidden" style="color:var(--accent)" type="button">Cargar más</button>
                    </div>
                </div>
            </div>
        </div>
        <div>
            <div class="section-title px-1">LÍNEA DE TIEMPO</div>
            <div class="surface p-5" id="events-log"><div class="text-gray-500 text-sm">Cargando dashboard...</div></div>
        </div>
    </div>
    <script>
const stateCache={};
let ceoTradeVisibleCount=10;
let ceoTradeRows=[];
let ceoTradeFilter='all';

function filterCeoTrades(rows) {
  if (ceoTradeFilter==='all') return rows;
  const now=new Date();
  const st=new Date(now.getFullYear(),now.getMonth(),now.getDate());
  const dMs=864e5;
  return rows.filter(t=>{
    if (!t.exit_time_iso) return false;
    const d=new Date(t.exit_time_iso);
    if (isNaN(d)) return false;
    if (ceoTradeFilter==='today') return d>=st;
    if (ceoTradeFilter==='7d') return (now-d)<=7*dMs;
    if (ceoTradeFilter==='30d') return (now-d)<=30*dMs;
    return true;
  });
}

function updateCeoFilterButtons() {
  document.querySelectorAll('#ceo-trades-filters button[data-filter]').forEach(b=>{
    const a=b.dataset.filter===ceoTradeFilter;
    b.style.borderColor=a?'var(--accent)':'var(--border)';
    b.style.color=a?'var(--accent)':'';
    b.className='px-2 py-1 text-xs rounded border'+(a?'':' text-gray-400');
  });
}

function updateValue(id,v,cls) {
  const el=document.getElementById(id);
  if (!el) return;
  if (stateCache[id]!==v) {
    el.innerText=v;
    if (cls) el.className=cls;
    const oN=parseFloat((stateCache[id]||'').toString().replace(/[^0-9.\\-]+/g,''));
    const nN=parseFloat(String(v).replace(/[^0-9.\\-]+/g,''));
    if (!isNaN(oN) && !isNaN(nN) && oN!==nN) {
      el.classList.remove('value-up','value-down');
      void el.offsetWidth;
      el.classList.add(nN>oN?'value-up':'value-down');
    }
    stateCache[id]=v;
  }
}

function rsiC(v) {
  return v>=70?'var(--negative)':v<=30?'var(--positive)':'var(--accent)';
}

function tB(t) {
  return t==='bullish'?'<span style="color:var(--positive);font-weight:600">B</span>':t==='bearish'?'<span style="color:var(--negative);font-weight:600">S</span>':'<span style="color:var(--text-secondary)">N</span>';
}

function sB(i) {
  if (i.has_position) return '<span class="badge badge-active">● ACTIVA</span>';
  if (i.can_enter) return '<span class="badge badge-buy">▲ COMPRAR</span>';
  return '<span class="badge badge-hold">● ESPERAR</span>';
}

function renderVibe(v) {
  const b=document.getElementById('vibe-status-badge');
  if (v.mcp_available) {
    b.className='badge badge-vibe';
    b.textContent='Conectado';
  } else {
    b.className='badge badge-wait';
    b.textContent='Desconectado';
  }
  const d=document.getElementById('dot-vibe');
  if (d) d.style.color=v.mcp_available?'var(--vibe)':'var(--text-dim)';
  const c=document.getElementById('vibe-content');
  let h='';
  const p=v.patterns||{};
  const pk=Object.keys(p);
  if (pk.length) {
    for (const s of pk) {
      const t=p[s]||'';
      h+=`<div class="vibe-item"><span class="mono text-xs" style="color:var(--vibe);font-weight:600">${s}</span> <span class="text-sm text-gray-300">${t.substring(0,150)}</span></div>`;
    }
  } else {
    h+='<div class="text-gray-500 text-sm">🔍 Patrones: esperando datos…</div>';
  }
  const j=v.journal||'';
  if (j) {
    h+=`<div class="vibe-item"><span class="text-xs" style="color:var(--vibe);font-weight:600">📋 Journal</span> <span class="text-sm text-gray-300">${j.substring(0,150)}</span></div>`;
  } else {
    h+='<div class="text-gray-500 text-sm">📋 Journal: sin analizar</div>';
  }
  const bt=v.backtest||{};
  const bk=Object.keys(bt);
  if (bk.length) {
    for (const s of bk) {
      const t=bt[s]||'';
      h+=`<div class="vibe-item"><span class="text-xs" style="color:var(--vibe);font-weight:600">📊 Backtest ${s}</span> <span class="text-sm text-gray-300">${t.substring(0,150)}</span></div>`;
    }
  } else {
    h+='<div class="text-gray-500 text-sm">📊 Backtest: semanal…</div>';
  }
  const f=v.factors||{};
  const fk=Object.keys(f);
  if (fk.length) {
    for (const s of fk) {
      const t=f[s]||'';
      h+=`<div class="vibe-item"><span class="text-xs" style="color:var(--vibe);font-weight:600">🧬 Factors ${s}</span> <span class="text-sm text-gray-300">${t.substring(0,150)}</span></div>`;
    }
  } else {
    h+='<div class="text-gray-500 text-sm">🧬 Factors: mensual…</div>';
  }
  const sh=v.shadow||'';
  if (sh) {
    h+=`<div class="vibe-item"><span class="text-xs" style="color:var(--vibe);font-weight:600">🌙 Shadow</span> <span class="text-sm text-gray-300">${sh.substring(0,150)}</span></div>`;
  } else {
    h+='<div class="text-gray-500 text-sm">🌙 Shadow: semanal…</div>';
  }
  c.innerHTML=h;
}

async function fetchState() {
  try {
    const r=await fetch('/api/state');
    if (!r.ok) {
      console.error('API error:',r.status);
      return;
    }
    const d=await r.json();
    window.__lastData=d;
    render(d);
  } catch(e) {
    console.error('Error fetching state:',e);
  }
}

function render(data) {
  document.getElementById('uptime').innerText=data.uptime;
  const pair=data.primary_pair||(Array.isArray(data.watchlist)&&data.watchlist[0])||'—';
  document.getElementById('sentiment').innerText=pair.replace('/',' / ');
  document.getElementById('sentiment-detail').innerText=data.strategy_blurb||'';
  const bs=document.getElementById('bot-status');
  if (data.global_hold) {
    bs.innerHTML='<span style="color:var(--negative)">⛔ Pausa</span>';
  } else if (data.open_count>0) {
    bs.innerHTML='<span style="color:var(--accent)">⚡ Operando</span>';
  } else {
    bs.innerHTML='<span style="color:var(--positive)">✅ Listo</span>';
  }
  updateValue('pos-count',`${data.open_count||0}/${data.max_positions||3}`);
  updateValue('s-wins',data.session_wins||0);
  updateValue('s-losses',data.session_losses||0);
  updateValue('s-winrate',data.session_winrate||'0%');
  
  const th=Number.isFinite(data.buy_prob_threshold_pct)?data.buy_prob_threshold_pct:70;
  updateValue('sent-num',th+'%');
  const sb=document.getElementById('sent-bar');
  if (sb) sb.style.width=Math.min(100,Math.max(0,th))+'%';
  
  const la=document.getElementById('api-latency');
  if (la) {
    const ms=Number(data.api_latency_ms||0);
    la.innerText=ms>0?(Math.round(ms)+' ms'):'—';
  }
  updateValue('balance',data.balance);
  updateValue('balance-lg',data.balance);
  updateValue('margin',data.available_margin);
  
  const pnlEl=document.getElementById('session-pnl');
  updateValue('session-pnl',data.session_pnl);
  if (pnlEl) pnlEl.style.color=data.session_pnl_num>=0?'var(--positive)':'var(--negative)';
  
  const ddEl=document.getElementById('drawdown');
  updateValue('drawdown',data.max_drawdown);
  if (ddEl) ddEl.style.color=data.max_drawdown_num<0?'var(--negative)':'var(--text-primary)';
  
  const buyTh=Number.isFinite(data.buy_prob_threshold_pct)?data.buy_prob_threshold_pct:70;
  
  let rh='';
  data.market.forEach(i=>{
    const ml=Math.min(100,Math.max(0,parseFloat(String(i.ml_conf).replace(/[%\\s]/g,''))||0));
    const mc=ml>=buyTh?'var(--positive)':'var(--accent)';
    const rn=(i.rsi==='--'||i.rsi===undefined)?NaN:parseFloat(i.rsi);
    const rs=Number.isFinite(rn)?rn.toFixed(0):'—';
    const rv=Number.isFinite(rn)?Math.min(100,Math.max(0,rn)):0;
    const cc=i.change_ok===true;
    const chC=!cc?'color:var(--text-dim)':(i.change_num>=0?'color:var(--positive)':'color:var(--negative)');
    const chG=!cc?'•':(i.change_num>=0?'↗':'↘');
    const st=i.symbol_pair?String(i.symbol_pair).replace('/',' / '):i.symbol;
    const rb=i.has_position?'background:rgba(0,212,255,.04)':'';
    let ph='<span style="color:var(--text-dim)">—</span>';
    if (i.has_position && i.unrealized_pnl_num!==undefined) {
      const c=i.unrealized_pnl_num>=0?'var(--positive)':'var(--negative)';
      ph=`<span class="mono font-medium" style="${c}">${i.unrealized_pnl_num>=0?'+':''}${i.unrealized_pnl_num.toFixed(2)}</span>`;
    }
    rh+=`<tr class="mkt-row border-b" style="border-color:var(--border);${rb}"><td class="px-4 py-3"><span class="font-semibold text-white">${st}</span><div class="text-[10px] text-gray-500 mono">${i.symbol}</div></td><td class="px-4 py-3 text-right mono font-medium text-white">${i.price}</td><td class="px-4 py-3 text-right hide-mobile" style="${chC}">${chG} ${i.change}</td><td class="px-4 py-3 text-center hide-mobile"><span class="mono" style="color:${rsiC(Number.isFinite(rn)?rn:50)}">${rs}</span><div class="bar-track mt-1"><div class="bar-fill" style="width:${rv}%;background:${rsiC(Number.isFinite(rn)?rn:50)}"></div></div></td><td class="px-4 py-3 text-center"><span class="mono" style="color:${mc}">${i.ml_conf}</span><div class="bar-track mt-1"><div class="bar-fill" style="width:${ml}%;background:${mc}"></div></div></td><td class="px-4 py-3 text-center hide-mobile">${tB(i.trend_detail?.['15m']||'neutral')}/${tB(i.trend_detail?.['1h']||'neutral')}/${tB(i.trend_detail?.['4h']||'neutral')}</td><td class="px-4 py-3 text-center">${ph}</td><td class="px-4 py-3 text-center">${sB(i)}</td></tr>`;
  });
  document.getElementById('market-tbody').innerHTML=rh;
  
  let ph2='';
  let hp=false;
  data.market.forEach(i=>{
    if (!i.has_position) return;
    hp=true;
    const pc=i.unrealized_pnl_num>=0?'var(--positive)':'var(--negative)';
    ph2+=`<tr class="border-b" style="border-color:var(--border)"><td class="px-4 py-2 font-semibold text-white">${i.symbol_pair||i.symbol}</td><td class="px-4 py-2 text-right mono">${i.position_str||'—'}</td><td class="px-4 py-2 text-right mono text-white">${i.price}</td><td class="px-4 py-2 text-right mono" style="color:var(--warning)">${i.stop_loss||'—'}</td><td class="px-4 py-2 text-right mono" style="color:var(--positive)">${i.take_profit||'—'}</td><td class="px-4 py-2 text-right mono font-medium" style="${pc}">${i.unrealized_pnl||'—'}</td><td class="px-4 py-2 text-center">${i.trailing_active?'<span class="badge badge-buy" style="font-size:10px">🔒 TRAIL</span>':'<span class="text-gray-500 text-xs">OFF</span>'}</td></tr>`;
  });
  document.getElementById('pos-tbody').innerHTML=ph2||'<tr><td colspan="7" class="text-center text-gray-500 py-4">Sin posiciones</td></tr>';
  document.getElementById('positions-section').classList.toggle('hidden',!hp);
  
  renderVibe(data.vibe||{});
  
  const ceo=data.ceo||{};
  const ms=v=>v>=0?'color:var(--positive);font-weight:500':'color:var(--negative);font-weight:500';
  const p7n=Number(ceo.pnl_7d_num||0);
  const p30n=Number(ceo.pnl_30d_num||0);
  updateValue('ceo-pnl-7d',ceo.pnl_7d||'—');
  const p7e=document.getElementById('ceo-pnl-7d');
  if (p7e) p7e.style.cssText=ms(p7n);
  updateValue('ceo-winrate-7d',ceo.winrate_7d||'—');
  updateValue('ceo-pf-30d',ceo.profit_factor_30d||'—');
  updateValue('ceo-pnl-30d',ceo.pnl_30d||'—');
  const p30e=document.getElementById('ceo-pnl-30d');
  if (p30e) p30e.style.cssText=ms(p30n);
  document.getElementById('ceo-updated').innerText=`Actualizado: ${ceo.last_updated_local||'—'}`;
  
  const syEl=document.getElementById('ceo-symbols');
  const sy=ceo.symbols_month||[];
  syEl.innerHTML=sy.length===0?'<div class="text-gray-500">Sin datos</div>':sy.slice(0,6).map(s=>`<div class="flex justify-between"><span class="text-gray-300">${s.symbol}</span><span class="mono" style="${Number(s.pnl_total)>=0?'color:var(--positive)':'color:var(--negative)'}">${s.pnl_label}</span></div>`).join('');
  
  const trEl=document.getElementById('ceo-trades');
  const tr=ceo.recent_trades||[];
  ceoTradeRows=tr;
  const ft=filterCeoTrades(ceoTradeRows);
  const rc=Math.min(ceoTradeVisibleCount,ft.length);
  trEl.innerHTML=ft.length===0?'<div class="text-gray-500">Sin trades cerrados</div>':ft.slice(0,rc).map(t=>`<div class="flex justify-between"><span class="text-gray-300">${t.symbol} <span class="text-gray-500">(${t.exit_time_local})</span></span><span class="mono" style="${Number(t.pnl_num)>=0?'color:var(--positive)':'color:var(--negative)'}">${t.pnl}</span></div>`).join('');
  updateCeoFilterButtons();
  const mb=document.getElementById('ceo-trades-more');
  mb.classList.toggle('hidden',ft.length<=rc);
  mb.innerText=`Cargar más (${ft.length-rc})`;
  
  const evEl=document.getElementById('events-log');
  if (data.events.length>0) {
    evEl.innerHTML=data.events.map((e,i)=>`<div class="flex items-start gap-3 py-1.5"><div class="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0" style="background:${i===0?'var(--accent)':'var(--text-dim)'}"></div><div class="text-sm text-gray-300">${e}</div></div>`).join('');
  } else {
    evEl.innerHTML='<div class="text-gray-500 text-sm">Sin eventos recientes…</div>';
  }
}

fetchState();
setInterval(fetchState,2500);

document.getElementById('ceo-trades-more').addEventListener('click',()=>{
  ceoTradeVisibleCount+=20;
  render({...window.__lastData,ceo:{...(window.__lastData?.ceo||{}),recent_trades:ceoTradeRows}});
});

document.querySelectorAll('#ceo-trades-filters button[data-filter]').forEach(b=>{
  b.addEventListener('click',()=>{
    ceoTradeFilter=b.dataset.filter||'all';
    ceoTradeVisibleCount=10;
    render({...window.__lastData,ceo:{...(window.__lastData?.ceo||{}),recent_trades:ceoTradeRows}});
  });
});
    </script>
</body>
</html>"""

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
    # Min seconds between full MT5↔memory syncs on /api/state.
    # Default 5s avoids hammering MT5 on every browser poll (~2.5s).
    _raw_api_iv = os.environ.get(
        "WEB_API_MT5_SYNC_MIN_INTERVAL_S",
        os.environ.get("WEB_API_MT5_SYNC_EVERY_S", "5"),
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
        logger.debug("Web GET / from %s", request.remote)
        try:
            return web.Response(
                text=HTML_TEMPLATE,
                content_type="text/html",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        except Exception as exc:
            logger.error("Web failed to serve HTML: %s", exc)
            raise

    async def handle_health(request: web.Request) -> web.Response:
        logger.debug("Web GET /health from %s", request.remote)
        # Use a dummy queue with qsize=0 since web server does not hold the market queue
        class _DummyQueue:
            def qsize(self) -> int:
                return 0
        metrics = get_health_metrics(state, _DummyQueue(), paper_executor, risk_manager)
        return web.json_response(metrics)

    async def handle_api(request: web.Request) -> web.Response:
        logger.debug("Web GET /api/state from %s", request.remote)
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
        global_hold = bool(state.get("global_hold", False))
        sentiment_label = "ETH/USDT ML-only"
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
            prices = list(state.get("prices", {}).get(sym, []))
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
            "vibe": {
                "mcp_available": bool(state.get("vibe_client") and state["vibe_client"].available),
                "patterns": {s: _extract_vibe_text(d) for s, d in state.get("vibe_patterns", {}).items()},
                "journal": _extract_vibe_text(state.get("vibe_journal_analysis")),
                "backtest": {s: _extract_vibe_text(d) for s, d in state.get("vibe_backtest", {}).items()},
                "factors": {s: _extract_vibe_text(d) for s, d in state.get("vibe_factors", {}).items()},
                "shadow": _extract_vibe_text(state.get("vibe_shadow_report")),
            },
        }
        
        return web.json_response(resp)

    async def handle_performance(request: web.Request) -> web.Response:
        try:
            from bot.performance_analyzer import analyzer
            metrics = await analyzer.get_metrics()
            return web.json_response(metrics)
        except Exception as exc:
            logger.warning("Web /api/performance failed: %s", exc)
            return web.json_response(
                {"error": "service_unavailable", "detail": str(exc)},
                status=503,
            )

    _rpm_raw = os.environ.get("DASH_API_RATE_PER_MIN", "180").strip()
    try:
        _rpm = max(0, int(_rpm_raw))
    except ValueError:
        _rpm = 180

    _dash_token = os.environ.get("DASHBOARD_TOKEN", "").strip()

    @web.middleware
    async def _auth_middleware(request: web.Request, handler):
        """Require DASHBOARD_TOKEN on sensitive endpoints if set."""
        if _dash_token and request.path not in ("/", "/health"):
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() != _dash_token:
                return web.json_response(
                    {"error": "unauthorized"},
                    status=401,
                    headers={"WWW-Authenticate": 'Bearer realm="dashboard"'},
                )
        return await handler(request)

    @web.middleware
    async def _cors_middleware(request: web.Request, handler):
        """Add CORS headers for API endpoints."""
        response = await handler(request)
        if request.path.startswith("/api/"):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    _middlewares = [_security_and_error_guard, _auth_middleware, _cors_middleware]
    if _rpm > 0:
        _middlewares.append(_make_api_rate_middleware(_rpm))

    # ── Vibe-Trading analysis endpoints ──────────────────────────────────
    async def _handle_vibe_journal(request: web.Request) -> web.Response:
        analysis = state.get("vibe_journal_analysis")
        if not analysis:
            return web.json_response({"status": "no_data"}, status=404)
        return web.json_response(analysis)

    async def _handle_vibe_backtest(request: web.Request) -> web.Response:
        backtests = state.get("vibe_backtest", {})
        if not backtests:
            return web.json_response({"status": "no_data"}, status=404)
        return web.json_response(backtests)

    async def _handle_vibe_patterns(request: web.Request) -> web.Response:
        patterns = state.get("vibe_patterns", {})
        if not patterns:
            return web.json_response({"status": "no_data"}, status=404)
        return web.json_response(patterns)

    async def _handle_vibe_factors(request: web.Request) -> web.Response:
        factors = state.get("vibe_factors", {})
        if not factors:
            return web.json_response({"status": "no_data"}, status=404)
        return web.json_response(factors)

    async def _handle_vibe_shadow(request: web.Request) -> web.Response:
        shadow = state.get("vibe_shadow_report")
        if not shadow:
            return web.json_response({"status": "no_data"}, status=404)
        return web.json_response(shadow)

    async def _handle_vibe_status(request: web.Request) -> web.Response:
        vibe_c = state.get("vibe_client")
        return web.json_response({
            "available": vibe_c.available if vibe_c else False,
            "tools": [
                "analyze_trade_journal", "backtest", "pattern_recognition",
                "factor_analysis", "extract_shadow_strategy",
                "run_shadow_backtest", "render_shadow_report",
                "get_market_data", "list_skills",
            ] if (vibe_c and vibe_c.available) else [],
        })

    app = web.Application(middlewares=_middlewares)
    app.add_routes([
        web.get("/", handle_html),
        web.get("/health", handle_health),
        web.get("/api/state", handle_api),
        web.get("/api/performance", handle_performance),
        web.get("/api/vibe/journal", _handle_vibe_journal),
        web.get("/api/vibe/backtest", _handle_vibe_backtest),
        web.get("/api/vibe/patterns", _handle_vibe_patterns),
        web.get("/api/vibe/factors", _handle_vibe_factors),
        web.get("/api/vibe/shadow", _handle_vibe_shadow),
        web.get("/api/vibe/status", _handle_vibe_status),
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
