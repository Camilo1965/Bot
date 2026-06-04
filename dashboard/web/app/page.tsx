"use client";

import { useEffect, useState } from "react";
import { KpiCard } from "@/components/ui/KpiCard";
import { EquityChart } from "@/components/charts/EquityChart";
import { OpenPositionsList } from "@/components/OpenPositionsList";
import { RecentSignalsTable } from "@/components/RecentSignalsTable";
import { SymbolHealthStrip } from "@/components/SymbolHealthStrip";
import { EmptyState } from "@/components/ui/EmptyState";
import { useWebSocket } from "@/hooks/useWebSocket";
import { fmtMoney, fmtPct, fmtTimeAgo, pnlColor, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface BotState {
  balance: number;
  equity: number;
  daily_pnl_pct: number;
  open_positions_count: number;
  kill_switch_active: boolean;
  kill_switch_reason?: string | null;
  last_updated: string;
}

interface EquityPoint { ts: string; equity: number }

export default function OverviewPage() {
  const [state, setState] = useState<BotState | null>(null);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [loaded, setLoaded] = useState(false);

  const { lastMessage } = useWebSocket(`${API.replace("http", "ws")}/ws/stream`);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/state`).then((r) => r.json()),
      fetch(`${API}/api/equity?days=30`).then((r) => r.json()),
    ])
      .then(([st, eq]) => {
        setState(st);
        setEquityHistory(Array.isArray(eq) ? eq : []);
      })
      .finally(() => setLoaded(true));
  }, []);

  useEffect(() => {
    if (!lastMessage) return;
    try {
      const msg = JSON.parse(lastMessage);
      if (msg.event === "equity.tick") {
        setState((prev) =>
          prev ? { ...prev, equity: msg.equity, daily_pnl_pct: msg.daily_pnl_pct } : prev
        );
        setEquityHistory((prev) => [...prev.slice(-500), { ts: msg.ts, equity: msg.equity }]);
      }
    } catch {}
  }, [lastMessage]);

  const equitySpark = equityHistory.slice(-30).map((p) => p.equity);
  const startEquity = equityHistory[0]?.equity ?? state?.balance ?? 0;
  const currentEquity = state?.equity ?? 0;
  const periodDelta = startEquity > 0 ? ((currentEquity - startEquity) / startEquity) * 100 : null;

  return (
    <div className="space-y-5">
      {/* Page header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Overview</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">
            Last update: <span className={TABULAR}>{fmtTimeAgo(state?.last_updated)}</span>
          </p>
        </div>
        {state?.kill_switch_active && (
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-[#EF4444]/15 border border-[#EF4444]/40 text-[#EF4444] text-xs font-semibold tracking-wide">
            <span className="w-1.5 h-1.5 rounded-full bg-[#EF4444] animate-pulse" />
            KILL SWITCH ACTIVE
            {state.kill_switch_reason && <span className="font-normal opacity-80">· {state.kill_switch_reason}</span>}
          </span>
        )}
      </div>

      {/* KPI grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <KpiCard
          label="Balance"
          value={fmtMoney(state?.balance ?? null)}
          loading={!loaded}
          hint="Cash + open margin"
        />
        <KpiCard
          label="Equity"
          value={fmtMoney(state?.equity ?? null)}
          loading={!loaded}
          delta={periodDelta}
          deltaLabel="30d"
          sparkline={equitySpark}
          hint="Balance + unrealized PnL"
        />
        <KpiCard
          label="Daily PnL"
          value={fmtPct(state?.daily_pnl_pct ?? null)}
          valueClass={pnlColor(state?.daily_pnl_pct ?? 0)}
          loading={!loaded}
        />
        <KpiCard
          label="Open Positions"
          value={String(state?.open_positions_count ?? 0)}
          loading={!loaded}
          hint="Concurrent live positions"
        />
        <KpiCard
          label="Kill Switch"
          value={state?.kill_switch_active ? "ACTIVE" : "OFF"}
          valueClass={state?.kill_switch_active ? "text-[#EF4444]" : "text-[#10B981]"}
          loading={!loaded}
        />
      </div>

      {/* Equity curve */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Equity Curve (30d)</h2>
          {periodDelta !== null && (
            <span className={`text-xs font-mono ${TABULAR} ${pnlColor(periodDelta)}`}>
              {fmtPct(periodDelta)}
            </span>
          )}
        </div>
        {equityHistory.length === 0 ? (
          <EmptyState icon="∿" title="No equity history yet" body="Bot needs to make trades for the chart to populate." />
        ) : (
          <EquityChart data={equityHistory} />
        )}
      </div>

      {/* Bottom grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Open Positions</h2>
          <OpenPositionsList apiBase={API} />
        </div>
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Recent Signals</h2>
          <RecentSignalsTable apiBase={API} />
        </div>
      </div>

      <SymbolHealthStrip apiBase={API} />
    </div>
  );
}
