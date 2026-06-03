"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface RiskState {
  kill_switch_active: boolean;
  consecutive_losses: number;
  daily_pnl_pct: number;
  drawdown_7d_pct: number;
  demoted_symbols: string[];
  reason?: string;
}

interface KillSwitchEvent {
  ts: string;
  action: "triggered" | "reset";
  reason?: string;
  triggered_by?: string;
}

export default function RiskPage() {
  const [state, setState] = useState<RiskState | null>(null);
  const [history, setHistory] = useState<KillSwitchEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [resetLoading, setResetLoading] = useState(false);
  const [resetMsg, setResetMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/risk/state`).then((r) => r.json()),
      fetch(`${API}/api/risk/killswitch/history`).then((r) => r.json()),
    ])
      .then(([s, h]) => {
        setState(s);
        setHistory(Array.isArray(h) ? h : []);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  async function handleReset() {
    setResetLoading(true);
    setResetMsg(null);
    try {
      const r = await fetch(`${API}/api/risk/killswitch/reset`, { method: "POST" });
      const body = await r.json();
      setResetMsg(r.ok ? "Kill switch reset successfully." : body?.detail ?? "Reset failed.");
      if (r.ok) setState((prev) => prev ? { ...prev, kill_switch_active: false } : prev);
    } catch (e) {
      setResetMsg(`Error: ${e}`);
    } finally {
      setResetLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Risk</h1>

      {loading && (
        <div className="text-[#9CA3AF] text-sm py-8 text-center">Loading risk state…</div>
      )}
      {error && (
        <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>
      )}

      {!loading && !error && state && (
        <>
          {/* Kill switch status */}
          <div
            className={`rounded-lg border p-4 ${
              state.kill_switch_active
                ? "bg-[#EF4444]/10 border-[#EF4444]"
                : "bg-[#10B981]/10 border-[#10B981]"
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-1">
                  Kill Switch
                </div>
                <div
                  className={`text-2xl font-bold font-mono ${
                    state.kill_switch_active ? "text-[#EF4444]" : "text-[#10B981]"
                  }`}
                >
                  {state.kill_switch_active ? "ACTIVE" : "OFF"}
                </div>
                {state.kill_switch_active && state.reason && (
                  <div className="text-sm text-[#9CA3AF] mt-1">Reason: {state.reason}</div>
                )}
              </div>
              <div className="flex flex-col gap-2 items-end">
                <div className="relative group">
                  <button
                    disabled
                    className="px-3 py-1.5 rounded bg-[#374151] text-[#9CA3AF] text-sm cursor-not-allowed"
                  >
                    Trigger Kill Switch
                  </button>
                  <div className="absolute right-0 top-full mt-1 hidden group-hover:block bg-[#374151] text-[#9CA3AF] text-xs rounded px-2 py-1 whitespace-nowrap z-10">
                    Use API directly
                  </div>
                </div>
                <button
                  onClick={handleReset}
                  disabled={resetLoading || !state.kill_switch_active}
                  className="px-3 py-1.5 rounded bg-[#3B82F6] text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#2563EB] transition-colors"
                >
                  {resetLoading ? "Resetting…" : "Reset"}
                </button>
              </div>
            </div>
            {resetMsg && (
              <div className="mt-2 text-sm text-[#F3F4F6] border-t border-[#374151]/50 pt-2">
                {resetMsg}
              </div>
            )}
          </div>

          {/* Risk metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
              <div className="text-[11px] text-[#9CA3AF] uppercase tracking-wider mb-1">
                Consecutive Losses
              </div>
              <div
                className={`font-mono text-xl font-bold ${
                  state.consecutive_losses >= 3 ? "text-[#EF4444]" : "text-[#F3F4F6]"
                }`}
              >
                {state.consecutive_losses}
              </div>
            </div>
            <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
              <div className="text-[11px] text-[#9CA3AF] uppercase tracking-wider mb-1">
                Daily PnL
              </div>
              <div
                className={`font-mono text-xl font-bold ${
                  state.daily_pnl_pct >= 0 ? "text-[#10B981]" : "text-[#EF4444]"
                }`}
              >
                {state.daily_pnl_pct >= 0 ? "+" : ""}
                {state.daily_pnl_pct.toFixed(2)}%
              </div>
            </div>
            <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
              <div className="text-[11px] text-[#9CA3AF] uppercase tracking-wider mb-1">
                Drawdown 7d
              </div>
              <div className="font-mono text-xl font-bold text-[#EF4444]">
                -{state.drawdown_7d_pct.toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Demoted symbols */}
          <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
            <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">
              Demoted Symbols
            </div>
            {state.demoted_symbols.length === 0 ? (
              <div className="text-[#9CA3AF] text-sm">None — all symbols active.</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {state.demoted_symbols.map((sym) => (
                  <span
                    key={sym}
                    className="px-2 py-1 rounded bg-[#EF4444]/20 text-[#EF4444] text-xs font-mono font-semibold"
                  >
                    {sym}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Kill switch history */}
          <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
            <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">
              Kill Switch History
            </div>
            {history.length === 0 ? (
              <div className="text-[#9CA3AF] text-sm">No history recorded.</div>
            ) : (
              <ul className="space-y-2">
                {history.map((ev, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-3 text-sm border-b border-[#374151]/40 pb-2"
                  >
                    <span
                      className={`mt-0.5 px-2 py-0.5 rounded text-[11px] font-semibold shrink-0 ${
                        ev.action === "triggered"
                          ? "bg-[#EF4444]/20 text-[#EF4444]"
                          : "bg-[#10B981]/20 text-[#10B981]"
                      }`}
                    >
                      {ev.action.toUpperCase()}
                    </span>
                    <div className="flex-1 min-w-0">
                      <span className="font-mono text-[#9CA3AF] text-xs">
                        {new Date(ev.ts).toLocaleString()}
                      </span>
                      {ev.reason && (
                        <span className="ml-2 text-[#F3F4F6] text-xs">{ev.reason}</span>
                      )}
                      {ev.triggered_by && (
                        <span className="ml-2 text-[#9CA3AF] text-xs">({ev.triggered_by})</span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </>
      )}
    </div>
  );
}
