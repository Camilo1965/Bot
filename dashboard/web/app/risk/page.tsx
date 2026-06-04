"use client";

import { useEffect, useState } from "react";
import { KpiCard } from "@/components/ui/KpiCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonKpi } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { useWsEvent } from "@/hooks/WsProvider";
import { fmtPct, fmtDateTime, pnlColor, TABULAR } from "@/lib/format";

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
  const [error, setError] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const toast = useToast();

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

  // Push: kill switch state changes
  useWsEvent("kill_switch.changed", (msg) => {
    setState((prev) => prev ? { ...prev, kill_switch_active: msg.active, reason: msg.reason } : prev);
    setHistory((prev) => [{
      ts: msg.ts,
      action: msg.active ? "triggered" : "reset",
      reason: msg.reason,
    }, ...prev]);
  });

  async function handleReset() {
    setConfirmOpen(false);
    setResetLoading(true);
    try {
      const r = await fetch(`${API}/api/risk/killswitch/reset`, { method: "POST" });
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        toast.push({ type: "success", title: "Kill switch reset" });
        setState((prev) => (prev ? { ...prev, kill_switch_active: false } : prev));
      } else {
        toast.push({ type: "error", title: "Reset failed", body: body?.detail });
      }
    } catch (e) {
      toast.push({ type: "error", title: "Reset error", body: String(e) });
    } finally {
      setResetLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Risk</h1>
        <p className="text-xs text-[#9CA3AF] mt-0.5">Kill switch state, exposure, and demoted symbols.</p>
      </div>

      {error && <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>}

      {/* Kill switch hero card */}
      {loading || !state ? (
        <SkeletonKpi />
      ) : (
        <div
          className={`rounded-xl border p-5 flex items-center justify-between flex-wrap gap-4 ${
            state.kill_switch_active
              ? "bg-[#EF4444]/10 border-[#EF4444]/40"
              : "bg-[#10B981]/10 border-[#10B981]/40"
          }`}
        >
          <div className="flex items-center gap-4">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold ${
                state.kill_switch_active ? "bg-[#EF4444]/20 text-[#EF4444]" : "bg-[#10B981]/20 text-[#10B981]"
              }`}
            >
              {state.kill_switch_active ? "!" : "✓"}
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF]">Kill Switch</div>
              <div className={`text-2xl font-bold font-mono ${TABULAR} ${state.kill_switch_active ? "text-[#EF4444]" : "text-[#10B981]"}`}>
                {state.kill_switch_active ? "ACTIVE" : "OFF"}
              </div>
              {state.kill_switch_active && state.reason && (
                <div className="text-xs text-[#F3F4F6] mt-1">{state.reason}</div>
              )}
            </div>
          </div>
          {state.kill_switch_active && (
            <button
              onClick={() => setConfirmOpen(true)}
              disabled={resetLoading}
              className="px-4 py-2 rounded bg-[#3B82F6] text-white text-sm font-medium hover:bg-[#2563EB] disabled:opacity-50 transition-colors"
            >
              {resetLoading ? "Resetting…" : "Reset Kill Switch"}
            </button>
          )}
        </div>
      )}

      {/* Risk metrics */}
      {loading || !state ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {Array.from({ length: 3 }).map((_, i) => <SkeletonKpi key={i} />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <KpiCard
            label="Consecutive Losses"
            value={String(state.consecutive_losses)}
            valueClass={state.consecutive_losses >= 3 ? "text-[#EF4444]" : "text-[#F3F4F6]"}
            hint="Kill switch triggers at MAX_CONSECUTIVE_LOSSES"
          />
          <KpiCard
            label="Daily PnL"
            value={fmtPct(state.daily_pnl_pct)}
            valueClass={pnlColor(state.daily_pnl_pct)}
            hint="Kill switch triggers at MAX_DAILY_LOSS_PCT"
          />
          <KpiCard
            label="Drawdown 7d"
            value={`-${state.drawdown_7d_pct.toFixed(2)}%`}
            valueClass="text-[#EF4444]"
            hint="Kill switch triggers at MAX_DRAWDOWN_7D_PCT"
          />
        </div>
      )}

      {/* Demoted symbols */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Demoted Symbols</div>
        {!state || state.demoted_symbols.length === 0 ? (
          <div className="text-[#9CA3AF] text-sm">None — all symbols active.</div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {state.demoted_symbols.map((sym) => (
              <span key={sym} className="px-2.5 py-1 rounded bg-[#EF4444]/15 text-[#EF4444] text-xs font-mono font-semibold border border-[#EF4444]/30">
                {sym}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* History */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Kill Switch History</div>
        {history.length === 0 ? (
          <EmptyState icon="≣" title="No kill switch events" body="Triggers and resets will appear here." />
        ) : (
          <ul className="space-y-2">
            {history.map((ev, i) => (
              <li key={i} className="flex items-start gap-3 text-sm border-b border-[#374151]/40 pb-2 last:border-0">
                <span className={`mt-0.5 px-2 py-0.5 rounded text-[10px] font-semibold shrink-0 ${
                  ev.action === "triggered" ? "bg-[#EF4444]/15 text-[#EF4444]" : "bg-[#10B981]/15 text-[#10B981]"
                }`}>
                  {ev.action.toUpperCase()}
                </span>
                <div className="flex-1 min-w-0">
                  <div className={`font-mono ${TABULAR} text-[#9CA3AF] text-xs`}>{fmtDateTime(ev.ts)}</div>
                  {ev.reason && <div className="text-[#F3F4F6] text-xs mt-0.5">{ev.reason}</div>}
                  {ev.triggered_by && <div className="text-[#6B7280] text-[10px] mt-0.5">via {ev.triggered_by}</div>}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Confirm dialog */}
      {confirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setConfirmOpen(false)}>
          <div className="bg-[#10151D] border border-[#EF4444]/40 rounded-xl shadow-2xl p-5 w-full max-w-sm mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="text-[#F3F4F6] font-semibold mb-2">Reset Kill Switch?</div>
            <div className="text-xs text-[#9CA3AF] mb-4">
              This re-enables new entries. Make sure you've reviewed the trigger reason before resetting.
            </div>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setConfirmOpen(false)} className="px-3 py-1.5 rounded border border-[#374151] text-[#9CA3AF] text-xs hover:text-[#F3F4F6]">
                Cancel
              </button>
              <button onClick={handleReset} className="px-3 py-1.5 rounded bg-[#EF4444] text-white text-xs font-medium hover:bg-[#DC2626]">
                Yes, reset
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
