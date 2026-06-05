"use client";

import { useEffect, useState } from "react";
import { TABULAR } from "@/lib/format";

import { API_BASE as API } from "@/lib/api";

type Tab = "symbols" | "config" | "access";

const TABS: { key: Tab; label: string }[] = [
  { key: "symbols", label: "Symbols" },
  { key: "config", label: "Config" },
  { key: "access", label: "Acceso" },
];

interface SymbolConfig {
  symbol: string;
  timeframe: string;
  prob_threshold: number;
  fixed_sl_pct: number;
  fixed_tp_pct: number;
  enabled: boolean;
  max_position_pct?: number;
}

interface BacktestState { symbol: string; logs: string[]; running: boolean; done: boolean; error: boolean }

function SymbolsTab() {
  const [configs, setConfigs] = useState<SymbolConfig[]>([]);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [bt, setBt] = useState<BacktestState | null>(null);

  useEffect(() => {
    fetch(`${API}/api/symbols/config`)
      .then((r) => r.json())
      .then((data) => { setConfigs(Array.isArray(data) ? data : []); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  function setThreshold(symbol: string, val: string) {
    const n = parseFloat(val);
    if (!isNaN(n) && n > 0 && n < 1) setOverrides((o) => ({ ...o, [symbol]: n }));
  }

  async function runQuickBacktest(symbol: string) {
    const threshold = overrides[symbol] ?? configs.find((c) => c.symbol === symbol)?.prob_threshold ?? 0.55;
    setBt({ symbol, logs: [], running: true, done: false, error: false });
    try {
      const res = await fetch(`${API}/api/backtest/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols: symbol, days: 30, mode: "disk", prob_threshold_override: threshold }),
      });
      if (!res.ok || !res.body) { setBt((b) => b ? { ...b, running: false, error: true } : b); return; }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.replace(/^data:\s*/, "").trim();
          if (!line) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.type === "log") setBt((b) => b ? { ...b, logs: [...b.logs, ev.line] } : b);
            if (ev.type === "done") setBt((b) => b ? { ...b, running: false, done: ev.exit_code === 0, error: ev.exit_code !== 0 } : b);
          } catch {}
        }
      }
    } catch (e) {
      setBt((b) => b ? { ...b, running: false, error: true, logs: [...(b?.logs ?? []), String(e)] } : b);
    }
  }

  if (loading) return <div className="text-[#9CA3AF] text-sm text-center py-6">Loading…</div>;
  if (error) return <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>;
  if (configs.length === 0) return <div className="text-[#9CA3AF] text-sm text-center py-6">No symbol configs found.</div>;

  return (
    <div className="space-y-4">
      <div className="text-xs text-[#9CA3AF]">
        Edit thresholds to preview, then click <span className="text-[#F3F4F6]">Quick BT</span> for a 30-day disk backtest.
        Changes here are local — update SYMBOL_CONFIG in source to persist.
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="text-[10px] uppercase tracking-widest text-[#9CA3AF] border-b border-[#374151]">
              {["Symbol", "TF", "Threshold", "SL%", "TP%", "Enabled", ""].map((h, i) => (
                <th key={i} className="pb-2 pr-4 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {configs.map((c) => {
              const thr = overrides[c.symbol] ?? c.prob_threshold;
              const dirty = overrides[c.symbol] !== undefined && overrides[c.symbol] !== c.prob_threshold;
              const isBtActive = bt?.symbol === c.symbol && bt.running;
              return (
                <tr key={c.symbol} className="border-b border-[#374151]/40 hover:bg-white/5 transition-colors">
                  <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{c.symbol}</td>
                  <td className="py-2 pr-4 font-mono text-[#9CA3AF]">{c.timeframe}</td>
                  <td className="py-2 pr-3">
                    <input
                      type="number" step="0.01" min="0.30" max="0.99"
                      value={thr.toFixed(2)}
                      onChange={(e) => setThreshold(c.symbol, e.target.value)}
                      className={`w-20 bg-[#0a0e15] border rounded px-2 py-0.5 text-xs font-mono focus:outline-none ${dirty ? "border-[#F59E0B] text-[#F59E0B]" : "border-[#374151] text-[#F3F4F6]"} focus:border-[#3B82F6]`}
                    />
                  </td>
                  <td className="py-2 pr-4 font-mono text-[#9CA3AF]">{(c.fixed_sl_pct * 100).toFixed(1)}%</td>
                  <td className="py-2 pr-4 font-mono text-[#9CA3AF]">{(c.fixed_tp_pct * 100).toFixed(1)}%</td>
                  <td className="py-2 pr-4">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${c.enabled ? "bg-[#10B981]/20 text-[#10B981]" : "bg-[#374151] text-[#9CA3AF]"}`}>
                      {c.enabled ? "ON" : "OFF"}
                    </span>
                  </td>
                  <td className="py-2">
                    <button
                      onClick={() => runQuickBacktest(c.symbol)}
                      disabled={isBtActive}
                      className={`px-3 py-1 rounded text-[11px] font-medium transition-colors ${isBtActive ? "bg-[#F59E0B]/20 text-[#F59E0B] cursor-wait" : "bg-[#374151] text-[#9CA3AF] hover:bg-[#3B82F6]/20 hover:text-[#3B82F6] cursor-pointer"}`}
                    >
                      {isBtActive ? "Running…" : "Quick BT"}
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {bt && (
        <div className="bg-[#0a0e15] border border-[#374151] rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs uppercase tracking-widest text-[#9CA3AF]">Backtest — {bt.symbol} (30d)</span>
            <span className={`text-[11px] font-semibold ${bt.running ? "text-[#F59E0B]" : bt.done ? "text-[#10B981]" : bt.error ? "text-[#EF4444]" : "text-[#9CA3AF]"}`}>
              {bt.running ? "Running…" : bt.done ? "Done" : bt.error ? "Error" : ""}
            </span>
          </div>
          <div className="h-48 overflow-y-auto font-mono text-xs text-[#10B981] space-y-0.5">
            {bt.logs.map((line, i) => (
              <div key={i} className="leading-5 whitespace-pre-wrap break-all">{line}</div>
            ))}
            {bt.logs.length === 0 && bt.running && <div className="text-[#9CA3AF]">Connecting…</div>}
          </div>
        </div>
      )}
    </div>
  );
}

interface ConfigRow { group: string; label: string; value: string | number; hint?: string }

function ConfigTab() {
  const [rows, setRows] = useState<ConfigRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/api/risk/state`)
      .then((r) => r.json())
      .then((s) => {
        const lim = s.limits ?? {};
        const fil = s.filters ?? {};
        const built: ConfigRow[] = [
          // Risk limits
          { group: "Risk Limits", label: "Max daily loss", value: `${lim.max_daily_loss_pct ?? "—"}%`, hint: "Kill switch fires if exceeded" },
          { group: "Risk Limits", label: "Max 7d drawdown", value: `${lim.max_drawdown_7d_pct ?? "—"}%`, hint: "Kill switch fires if exceeded" },
          { group: "Risk Limits", label: "Max consecutive losses", value: lim.max_consecutive_losses ?? "—", hint: "Kill switch fires if exceeded" },
          { group: "Risk Limits", label: "Max open positions", value: lim.max_positions ?? "—" },
          { group: "Risk Limits", label: "Risk per trade", value: `${lim.risk_per_trade_pct ?? "—"}%`, hint: "Of account balance" },
          // Entry filters
          { group: "Entry Filters", label: "Hour filter", value: fil.trade_hour_filter ? `ON — UTC ${fil.trade_hour_start_utc}:00–${fil.trade_hour_end_utc}:00` : "OFF" },
          { group: "Entry Filters", label: "SMA200 (1H)", value: fil.sma200_filter ? "ON" : "OFF", hint: "Blocks LONG when price < 1H SMA200" },
          { group: "Entry Filters", label: "Regime filter", value: fil.regime_filter_enabled ? `ON — threshold ${((fil.regime_trending_threshold ?? 0.65) * 100).toFixed(0)}%` : "OFF", hint: "Blocks entry in RANGING regime" },
          { group: "Entry Filters", label: "Max spread", value: `${fil.max_spread_pct ?? "—"}%`, hint: "Skip if spread wider" },
          // Execution
          { group: "Execution", label: "ATR SL multiplier", value: `${fil.atr_sl_mult ?? "—"}×` },
          { group: "Execution", label: "ATR TP multiplier", value: `${fil.atr_tp_mult ?? "—"}×` },
          { group: "Execution", label: "Base SL", value: `${fil.base_sl_pct ?? "—"}%` },
        ];
        setRows(built);
        setLoading(false);
      })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  if (loading) return <div className="text-[#9CA3AF] text-sm text-center py-6">Loading…</div>;
  if (error) return <div className="text-[#EF4444] text-sm py-4 text-center">Failed: {error}</div>;

  const groups = [...new Set(rows.map((r) => r.group))];

  return (
    <div className="space-y-5">
      <div className="text-xs text-[#9CA3AF]">
        Live values read from running bot. Change via <code className="text-[#F3F4F6]">.env</code> and restart.
      </div>
      {groups.map((group) => (
        <div key={group}>
          <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold mb-2">{group}</div>
          <div className="bg-[#0a0e15] border border-[#374151] rounded-lg divide-y divide-[#374151]/60">
            {rows.filter((r) => r.group === group).map((row) => (
              <div key={row.label} className="flex items-center justify-between px-4 py-3">
                <div>
                  <span className="text-xs text-[#F3F4F6]">{row.label}</span>
                  {row.hint && <span className="text-[10px] text-[#6B7280] ml-2">{row.hint}</span>}
                </div>
                <span className={`font-mono ${TABULAR} text-sm text-[#10B981] font-semibold`}>
                  {row.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function AccessTab() {
  const [info, setInfo] = useState<{ lan_ip?: string; dashboard_url?: string; api_url?: string; hostname?: string } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/system/info`)
      .then((r) => r.json())
      .then((d) => { setInfo(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const rows = info ? [
    { label: "Dashboard (browser)", value: info.dashboard_url, copy: true, highlight: true },
    { label: "API", value: info.api_url, copy: true, highlight: false },
    { label: "IP local (LAN)", value: info.lan_ip, copy: false, highlight: false },
    { label: "Hostname", value: info.hostname, copy: false, highlight: false },
  ] : [];

  return (
    <div className="space-y-4">
      <p className="text-xs text-[#9CA3AF]">
        URLs para acceder al dashboard desde cualquier dispositivo en la misma red.
        Para acceso desde internet cambia <code className="text-[#F3F4F6]">DASHBOARD_API_URL</code> en <code className="text-[#F3F4F6]">.env</code> a tu IP pública y abre los puertos 3000/8000 en el firewall.
      </p>

      {loading ? (
        <div className="text-[#9CA3AF] text-xs text-center py-6">Detectando IPs…</div>
      ) : !info ? (
        <div className="text-[#EF4444] text-xs py-4">No se pudo conectar al API.</div>
      ) : (
        <div className="space-y-2">
          {rows.map((r) => (
            <div key={r.label} className={`flex items-center justify-between border rounded-lg px-4 py-3 ${r.highlight ? "border-[#3B82F6]/40 bg-[#3B82F6]/5" : "border-[#374151]"}`}>
              <div>
                <div className={`text-xs font-medium ${r.highlight ? "text-[#F3F4F6]" : "text-[#9CA3AF]"}`}>{r.label}</div>
                <code className={`text-sm font-mono ${r.highlight ? "text-[#3B82F6]" : "text-[#F3F4F6]"}`}>{r.value ?? "—"}</code>
              </div>
              {r.copy && r.value && (
                <button
                  onClick={() => navigator.clipboard.writeText(r.value!)}
                  className="text-[10px] px-2 py-1 rounded bg-[#374151] text-[#9CA3AF] hover:text-[#F3F4F6] transition-colors ml-3 shrink-0"
                >
                  Copiar
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="border border-[#374151] rounded-lg px-4 py-3 space-y-1">
        <div className="text-xs font-medium text-[#F3F4F6]">Para VPS / acceso global</div>
        <div className="text-[11px] text-[#9CA3AF] space-y-1">
          <p>1. En <code className="text-[#F3F4F6]">.env</code> establece: <code className="text-[#F59E0B]">DASHBOARD_API_URL=http://&lt;TU_IP_PUBLICA&gt;:8000</code></p>
          <p>2. Abre puertos <code className="text-[#F3F4F6]">3000</code> y <code className="text-[#F3F4F6]">8000</code> en el firewall del servidor.</p>
          <p>3. Reinicia el bot — los servidores arrancan solos con la nueva IP.</p>
        </div>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("symbols");

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Settings</h1>
        <p className="text-xs text-[#9CA3AF] mt-0.5">Symbol configs, live bot config, and access URLs.</p>
      </div>

      <div className="flex gap-1 border-b border-[#374151] overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px whitespace-nowrap ${
              tab === t.key
                ? "border-[#3B82F6] text-[#F3F4F6]"
                : "border-transparent text-[#9CA3AF] hover:text-[#F3F4F6]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-5">
        {tab === "symbols" && <SymbolsTab />}
        {tab === "config" && <ConfigTab />}
        {tab === "access" && <AccessTab />}
      </div>
    </div>
  );
}
