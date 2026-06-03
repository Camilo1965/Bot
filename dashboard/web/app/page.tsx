"use client";

import { useEffect, useState } from "react";
import { KpiCard } from "@/components/design/KpiCard";
import { EquityChart } from "@/components/charts/EquityChart";
import { OpenPositionsList } from "@/components/OpenPositionsList";
import { RecentSignalsTable } from "@/components/RecentSignalsTable";
import { SymbolHealthStrip } from "@/components/SymbolHealthStrip";
import { useWebSocket } from "@/hooks/useWebSocket";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface BotState {
  balance: number;
  equity: number;
  daily_pnl_pct: number;
  open_positions_count: number;
  kill_switch_active: boolean;
  last_updated: string;
}

export default function OverviewPage() {
  const [state, setState] = useState<BotState | null>(null);
  const [equityHistory, setEquityHistory] = useState<{ ts: string; equity: number }[]>([]);

  const { lastMessage } = useWebSocket(`${API.replace("http", "ws")}/ws/stream`);

  useEffect(() => {
    fetch(`${API}/api/state`)
      .then((r) => r.json())
      .then(setState)
      .catch(console.error);

    fetch(`${API}/api/equity?days=30`)
      .then((r) => r.json())
      .then(setEquityHistory)
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!lastMessage) return;
    const msg = JSON.parse(lastMessage);
    if (msg.event === "equity.tick") {
      setState((prev) =>
        prev ? { ...prev, equity: msg.equity, daily_pnl_pct: msg.daily_pnl_pct } : prev
      );
      setEquityHistory((prev) => [
        ...prev.slice(-500),
        { ts: msg.ts, equity: msg.equity },
      ]);
    }
  }, [lastMessage]);

  const pnlColor = (v: number) => (v >= 0 ? "text-accent-green" : "text-accent-red");

  return (
    <div className="space-y-6">
      {state?.kill_switch_active && (
        <div className="bg-accent-critical/20 border border-accent-critical rounded-card px-4 py-3 text-sm font-mono">
          KILL SWITCH ACTIVE — all new entries halted
        </div>
      )}

      {/* KPI Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <KpiCard label="Balance" value={`$${(state?.balance ?? 0).toFixed(2)}`} />
        <KpiCard label="Equity" value={`$${(state?.equity ?? 0).toFixed(2)}`} />
        <KpiCard
          label="Daily PnL"
          value={`${(state?.daily_pnl_pct ?? 0) >= 0 ? "+" : ""}${(state?.daily_pnl_pct ?? 0).toFixed(2)}%`}
          valueClass={pnlColor(state?.daily_pnl_pct ?? 0)}
        />
        <KpiCard label="Open Pos." value={String(state?.open_positions_count ?? 0)} />
        <KpiCard
          label="Kill Switch"
          value={state?.kill_switch_active ? "ACTIVE" : "OFF"}
          valueClass={state?.kill_switch_active ? "text-accent-critical" : "text-accent-green"}
        />
      </div>

      {/* Equity curve */}
      <div className="bg-bg-surface border border-border-default rounded-card p-4">
        <h2 className="text-xs uppercase tracking-widest text-text-secondary mb-3">Equity Curve (30d)</h2>
        <EquityChart data={equityHistory} />
      </div>

      {/* Bottom grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-bg-surface border border-border-default rounded-card p-4">
          <h2 className="text-xs uppercase tracking-widest text-text-secondary mb-3">Open Positions</h2>
          <OpenPositionsList apiBase={API} />
        </div>
        <div className="bg-bg-surface border border-border-default rounded-card p-4">
          <h2 className="text-xs uppercase tracking-widest text-text-secondary mb-3">Recent Signals</h2>
          <RecentSignalsTable apiBase={API} />
        </div>
      </div>

      <SymbolHealthStrip apiBase={API} />
    </div>
  );
}
