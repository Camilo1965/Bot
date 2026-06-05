"use client";

import { useEffect, useState } from "react";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonKpi } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { useWsEvent } from "@/hooks/WsProvider";
import { fmtDateTime, TABULAR } from "@/lib/format";

import { API_BASE as API } from "@/lib/api";

interface Limits {
  max_daily_loss_pct: number;
  max_consecutive_losses: number;
  max_drawdown_7d_pct: number;
  max_positions: number;
  risk_per_trade_pct: number;
}

interface Filters {
  trade_hour_filter: boolean;
  trade_hour_start_utc: number;
  trade_hour_end_utc: number;
  max_spread_pct: number;
  regime_filter_enabled: boolean;
  regime_trending_threshold: number;
  sma200_filter: boolean;
  atr_sl_mult: number;
  atr_tp_mult: number;
  base_sl_pct: number;
}

interface RiskState {
  kill_switch_active: boolean;
  reason?: string;
  since?: string;
  until?: string;
  consecutive_losses: number;
  daily_pnl_pct: number;
  drawdown_7d_pct: number;
  demoted_symbols: string[] | Record<string, unknown>;
  limits?: Limits;
  filters?: Filters;
}

interface KillSwitchEvent {
  ts: string;
  action: "triggered" | "reset" | "demoted" | "other";
  reason?: string;
  triggered_by?: string;
  duration_hours?: number;
  symbol?: string;
}

function toDemotedList(d: RiskState["demoted_symbols"] | undefined): string[] {
  if (!d) return [];
  if (Array.isArray(d)) return d;
  if (typeof d === "object") return Object.keys(d);
  return [];
}

function normalizeEvent(raw: any): KillSwitchEvent {
  const event = String(raw?.event ?? raw?.action ?? "").toLowerCase();
  let action: KillSwitchEvent["action"] = "other";
  if (event.includes("activ") || event.includes("trigger")) action = "triggered";
  else if (event.includes("deactiv") || event.includes("reset")) action = "reset";
  else if (event.includes("demot")) action = "demoted";
  return {
    ts: String(raw?.ts ?? raw?.timestamp ?? ""),
    action,
    reason: raw?.reason ?? undefined,
    triggered_by: raw?.triggered_by ?? undefined,
    duration_hours: typeof raw?.duration_hours === "number" ? raw.duration_hours : undefined,
    symbol: raw?.symbol ?? undefined,
  };
}

function Gauge({ label, current, limit, unit = "%", invert = false }: {
  label: string; current: number; limit: number; unit?: string; invert?: boolean;
}) {
  const pct = limit > 0 ? Math.min(Math.abs(current) / limit, 1) : 0;
  const danger = pct >= 0.8;
  const warn = pct >= 0.5;
  const barColor = danger ? "#EF4444" : warn ? "#F59E0B" : "#10B981";
  const valColor = danger ? "text-[#EF4444]" : warn ? "text-[#F59E0B]" : "text-[#10B981]";

  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">{label}</span>
        <span className={`font-mono ${TABULAR} text-sm font-bold ${valColor}`}>
          {invert && current < 0 ? "" : ""}{Math.abs(current).toFixed(2)}{unit}
        </span>
      </div>
      <div className="relative h-2 bg-[#0a0e15] rounded-full overflow-hidden">
        <div
          className="absolute h-2 rounded-full transition-all duration-500"
          style={{ width: `${pct * 100}%`, backgroundColor: barColor }}
        />
      </div>
      <div className={`flex justify-between text-[10px] font-mono ${TABULAR} text-[#6B7280]`}>
        <span>0{unit}</span>
        <span className="text-[#F59E0B]">limit {limit}{unit}</span>
      </div>
    </div>
  );
}

function FilterBadge({ label, active, detail }: { label: string; active: boolean; detail?: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#374151]/40 last:border-0">
      <div>
        <span className="text-xs text-[#F3F4F6]">{label}</span>
        {detail && <span className="text-[10px] text-[#6B7280] ml-2">{detail}</span>}
      </div>
      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold font-mono ${
        active ? "bg-[#10B981]/15 text-[#10B981]" : "bg-[#374151] text-[#9CA3AF]"
      }`}>
        {active ? "ON" : "OFF"}
      </span>
    </div>
  );
}

function utcNow(): number {
  return new Date().getUTCHours();
}

export default function RiskPage() {
  const [state, setState] = useState<RiskState | null>(null);
  const [history, setHistory] = useState<KillSwitchEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [resetLoading, setResetLoading] = useState(false);
  const [triggerLoading, setTriggerLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [showTrigger, setShowTrigger] = useState(false);
  const [triggerReason, setTriggerReason] = useState("");
  const [triggerDuration, setTriggerDuration] = useState("24");
  const toast = useToast();

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/risk/state`).then((r) => r.json()),
      fetch(`${API}/api/risk/killswitch/history`).then((r) => r.json()),
    ])
      .then(([s, h]) => {
        setState(s);
        setHistory(Array.isArray(h) ? h.map(normalizeEvent) : []);
        setLoading(false);
      })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  useWsEvent("kill_switch.changed", (msg) => {
    setState((prev) => prev ? { ...prev, kill_switch_active: msg.active, reason: msg.reason } : prev);
    setHistory((prev) => [{
      ts: msg.ts, action: msg.active ? "triggered" : "reset", reason: msg.reason,
    } as KillSwitchEvent, ...prev]);
  });

  async function handleReset() {
    setConfirmReset(false);
    setResetLoading(true);
    try {
      const r = await fetch(`${API}/api/risk/killswitch/reset`, { method: "POST" });
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        toast.push({ type: "success", title: "Kill switch reset" });
        setState((prev) => prev ? { ...prev, kill_switch_active: false, reason: undefined } : prev);
      } else {
        toast.push({ type: "error", title: "Reset failed", body: body?.detail });
      }
    } catch (e) {
      toast.push({ type: "error", title: "Reset error", body: String(e) });
    } finally {
      setResetLoading(false);
    }
  }

  async function handleTrigger() {
    if (!triggerReason.trim()) { toast.push({ type: "error", title: "Enter a reason" }); return; }
    setTriggerLoading(true);
    try {
      const r = await fetch(`${API}/api/risk/killswitch/trigger`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: triggerReason.trim(), duration_hours: parseFloat(triggerDuration) || 24 }),
      });
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        toast.push({ type: "success", title: "Kill switch triggered" });
        setState((prev) => prev ? { ...prev, kill_switch_active: true, reason: triggerReason.trim() } : prev);
        setShowTrigger(false);
        setTriggerReason("");
      } else {
        toast.push({ type: "error", title: "Trigger failed", body: body?.detail });
      }
    } catch (e) {
      toast.push({ type: "error", title: "Trigger error", body: String(e) });
    } finally {
      setTriggerLoading(false);
    }
  }

  const lim = state?.limits;
  const fil = state?.filters;
  const demoted = toDemotedList(state?.demoted_symbols);

  const hourNow = utcNow();
  const hourStart = fil?.trade_hour_start_utc ?? 8;
  const hourEnd = fil?.trade_hour_end_utc ?? 22;
  const hourOpen = fil?.trade_hour_filter ? (hourNow >= hourStart && hourNow < hourEnd) : true;

  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Risk</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">Kill switch, exposure gauges, and active filters.</p>
        </div>
        {!loading && state && (
          <div className="flex gap-2">
            {state.kill_switch_active ? (
              <button
                onClick={() => setConfirmReset(true)}
                disabled={resetLoading}
                className="px-3 py-1.5 rounded bg-[#10B981] text-white text-xs font-semibold hover:bg-[#059669] disabled:opacity-50 transition-colors"
              >
                {resetLoading ? "Resetting…" : "Reset Kill Switch"}
              </button>
            ) : (
              <button
                onClick={() => setShowTrigger((v) => !v)}
                className="px-3 py-1.5 rounded border border-[#EF4444]/40 text-[#EF4444] text-xs font-semibold hover:bg-[#EF4444]/10 transition-colors"
              >
                Manual Trigger
              </button>
            )}
          </div>
        )}
      </div>

      {error && <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>}

      {/* Kill switch hero */}
      {loading || !state ? (
        <SkeletonKpi />
      ) : (
        <div className={`rounded-xl border p-5 flex items-center justify-between flex-wrap gap-4 ${
          state.kill_switch_active ? "bg-[#EF4444]/10 border-[#EF4444]/40" : "bg-[#10B981]/10 border-[#10B981]/40"
        }`}>
          <div className="flex items-center gap-4">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold ${
              state.kill_switch_active ? "bg-[#EF4444]/20 text-[#EF4444]" : "bg-[#10B981]/20 text-[#10B981]"
            }`}>
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
              {state.kill_switch_active && state.since && (
                <div className="text-[10px] text-[#9CA3AF] mt-0.5">since {fmtDateTime(state.since)}</div>
              )}
            </div>
          </div>
          {demoted.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {demoted.map((sym) => (
                <span key={sym} className="px-2 py-0.5 rounded bg-[#F59E0B]/15 text-[#F59E0B] text-[10px] font-mono font-semibold border border-[#F59E0B]/30">
                  {sym} demoted
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Manual trigger form */}
      {showTrigger && !state?.kill_switch_active && (
        <div className="bg-[#10151D] border border-[#EF4444]/30 rounded-lg p-4 space-y-3">
          <div className="text-xs font-semibold text-[#EF4444] uppercase tracking-wider">Manual Kill Switch Trigger</div>
          <div className="flex gap-3 flex-wrap">
            <input
              type="text"
              placeholder="Reason (required)"
              value={triggerReason}
              onChange={(e) => setTriggerReason(e.target.value)}
              className="flex-1 min-w-48 bg-[#0a0e15] border border-[#374151] rounded px-3 py-1.5 text-sm text-[#F3F4F6] placeholder-[#6B7280] focus:outline-none focus:border-[#EF4444]/60"
            />
            <div className="flex items-center gap-2">
              <span className="text-xs text-[#9CA3AF]">Duration</span>
              <select
                value={triggerDuration}
                onChange={(e) => setTriggerDuration(e.target.value)}
                className="bg-[#0a0e15] border border-[#374151] rounded px-2 py-1.5 text-sm text-[#F3F4F6] focus:outline-none focus:border-[#EF4444]/60"
              >
                {["1", "4", "8", "12", "24", "48", "168"].map((h) => (
                  <option key={h} value={h}>{h}h</option>
                ))}
              </select>
            </div>
            <button
              onClick={handleTrigger}
              disabled={triggerLoading || !triggerReason.trim()}
              className="px-4 py-1.5 rounded bg-[#EF4444] text-white text-xs font-semibold hover:bg-[#DC2626] disabled:opacity-50 transition-colors"
            >
              {triggerLoading ? "Triggering…" : "Trigger"}
            </button>
            <button
              onClick={() => setShowTrigger(false)}
              className="px-3 py-1.5 rounded border border-[#374151] text-[#9CA3AF] text-xs hover:text-[#F3F4F6]"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Exposure gauges */}
      {loading || !state ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {Array.from({ length: 3 }).map((_, i) => <SkeletonKpi key={i} />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Gauge
            label="Daily PnL"
            current={state.daily_pnl_pct}
            limit={lim?.max_daily_loss_pct ?? 5}
            unit="%"
            invert
          />
          <Gauge
            label="7d Drawdown"
            current={state.drawdown_7d_pct}
            limit={lim?.max_drawdown_7d_pct ?? 10}
            unit="%"
          />
          <Gauge
            label="Consecutive Losses"
            current={state.consecutive_losses}
            limit={lim?.max_consecutive_losses ?? 3}
            unit=""
          />
        </div>
      )}

      {/* Position limits row */}
      {lim && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Max Positions", value: lim.max_positions, unit: "" },
            { label: "Risk / Trade", value: lim.risk_per_trade_pct.toFixed(1), unit: "%" },
            { label: "ATR SL Mult", value: fil?.atr_sl_mult.toFixed(1) ?? "—", unit: "×" },
            { label: "ATR TP Mult", value: fil?.atr_tp_mult.toFixed(1) ?? "—", unit: "×" },
          ].map((item) => (
            <div key={item.label} className="bg-[#10151D] border border-[#374151] rounded-lg px-4 py-3">
              <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF]">{item.label}</div>
              <div className={`font-mono ${TABULAR} text-lg font-bold text-[#F3F4F6] mt-1`}>
                {item.value}{item.unit}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Active filters */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Active Filters</div>
        {fil ? (
          <div className="divide-y divide-[#374151]/0">
            <FilterBadge
              label="Hour Filter"
              active={fil.trade_hour_filter}
              detail={fil.trade_hour_filter
                ? `${hourStart}:00–${hourEnd}:00 UTC · now ${hourNow}:xx · ${hourOpen ? "OPEN" : "CLOSED"}`
                : "disabled"}
            />
            <FilterBadge
              label="SMA200 (1H)"
              active={fil.sma200_filter}
              detail="blocks LONG when price below 1H SMA200"
            />
            <FilterBadge
              label="Regime Filter"
              active={fil.regime_filter_enabled}
              detail={fil.regime_filter_enabled ? `threshold ${(fil.regime_trending_threshold * 100).toFixed(0)}% — RANGING blocks entry` : "disabled"}
            />
            <FilterBadge
              label="Spread Gate"
              active
              detail={`max ${fil.max_spread_pct.toFixed(2)}% spread — wider = skip`}
            />
            <FilterBadge
              label="Base SL"
              active
              detail={`${fil.base_sl_pct.toFixed(2)}% fixed stop-loss`}
            />
          </div>
        ) : (
          <div className="text-[#9CA3AF] text-xs">Loading filter config…</div>
        )}
      </div>

      {/* History */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Kill Switch History</div>
        {history.length === 0 ? (
          <EmptyState icon="≣" title="No kill switch events" body="Triggers and resets will appear here." />
        ) : (
          <ul className="space-y-2">
            {history.map((ev, i) => {
              const badgeCls =
                ev.action === "triggered" ? "bg-[#EF4444]/15 text-[#EF4444]" :
                ev.action === "reset" ? "bg-[#10B981]/15 text-[#10B981]" :
                ev.action === "demoted" ? "bg-[#F59E0B]/15 text-[#F59E0B]" :
                "bg-[#374151] text-[#9CA3AF]";
              return (
                <li key={i} className="flex items-start gap-3 text-sm border-b border-[#374151]/40 pb-2 last:border-0">
                  <span className={`mt-0.5 px-2 py-0.5 rounded text-[10px] font-semibold shrink-0 ${badgeCls}`}>
                    {(ev.action || "event").toUpperCase()}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className={`font-mono ${TABULAR} text-[#9CA3AF] text-xs`}>{fmtDateTime(ev.ts)}</div>
                    {ev.reason && <div className="text-[#F3F4F6] text-xs mt-0.5">{ev.reason}</div>}
                    {ev.symbol && <div className="text-[#6B7280] text-[10px] mt-0.5">symbol: <span className="font-mono">{ev.symbol}</span></div>}
                    {ev.duration_hours !== undefined && (
                      <div className="text-[#6B7280] text-[10px] mt-0.5">duration: {ev.duration_hours}h</div>
                    )}
                    {ev.triggered_by && <div className="text-[#6B7280] text-[10px] mt-0.5">via {ev.triggered_by}</div>}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Confirm reset dialog */}
      {confirmReset && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setConfirmReset(false)}>
          <div className="bg-[#10151D] border border-[#EF4444]/40 rounded-xl shadow-2xl p-5 w-full max-w-sm mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="text-[#F3F4F6] font-semibold mb-2">Reset Kill Switch?</div>
            <div className="text-xs text-[#9CA3AF] mb-4">
              This re-enables new entries. Review the trigger reason before resetting.
            </div>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setConfirmReset(false)} className="px-3 py-1.5 rounded border border-[#374151] text-[#9CA3AF] text-xs hover:text-[#F3F4F6]">
                Cancel
              </button>
              <button onClick={handleReset} className="px-3 py-1.5 rounded bg-[#10B981] text-white text-xs font-medium hover:bg-[#059669]">
                Yes, reset
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
