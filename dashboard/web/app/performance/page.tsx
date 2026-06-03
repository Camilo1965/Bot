"use client";

import { useEffect, useState } from "react";
import { KpiCard } from "@/components/design/KpiCard";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PerformanceSummary {
  total_trades: number;
  win_rate: number;
  pnl_pct: number;
  profit_factor: number;
  sharpe: number;
  max_drawdown_pct: number;
}

interface SymbolPerf {
  symbol: string;
  trades: number;
  win_rate: number;
  pnl_pct: number;
  profit_factor: number;
  avg_holding_h: number;
}

interface PerformanceData {
  summary: PerformanceSummary;
  by_symbol: SymbolPerf[];
}

function HeatmapPlaceholder({ title }: { title: string }) {
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
      <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">{title}</div>
      <div className="h-32 flex items-center justify-center border border-dashed border-[#374151] rounded text-[#9CA3AF] text-sm">
        Chart placeholder — implement with a heatmap library
      </div>
    </div>
  );
}

export default function PerformancePage() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/api/performance?range=30d`)
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  const s = data?.summary;

  const pnlColor = (v: number) => (v >= 0 ? "text-[#10B981]" : "text-[#EF4444]");
  const fmt = (v: number, digits = 2) => v.toFixed(digits);
  const pct = (v: number) => `${v >= 0 ? "+" : ""}${fmt(v)}%`;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Performance</h1>

      {loading && (
        <div className="text-[#9CA3AF] text-sm py-8 text-center">Loading performance data…</div>
      )}
      {error && (
        <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>
      )}

      {!loading && !error && s && (
        <>
          {/* KPI grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <KpiCard label="Total Trades" value={String(s.total_trades)} />
            <KpiCard
              label="Win Rate"
              value={`${fmt(s.win_rate * 100)}%`}
              valueClass={s.win_rate >= 0.5 ? "text-[#10B981]" : "text-[#EF4444]"}
            />
            <KpiCard
              label="PnL (30d)"
              value={pct(s.pnl_pct)}
              valueClass={pnlColor(s.pnl_pct)}
            />
            <KpiCard
              label="Profit Factor"
              value={fmt(s.profit_factor)}
              valueClass={s.profit_factor >= 1 ? "text-[#10B981]" : "text-[#EF4444]"}
            />
            <KpiCard label="Sharpe" value={fmt(s.sharpe)} />
            <KpiCard
              label="Max DD"
              value={`-${fmt(s.max_drawdown_pct)}%`}
              valueClass="text-[#EF4444]"
            />
          </div>

          {/* Per-symbol table */}
          <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
            <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">
              Per-Symbol Breakdown (30d)
            </div>
            {data!.by_symbol.length === 0 ? (
              <div className="text-[#9CA3AF] text-sm text-center py-6">No data.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-widest text-[#9CA3AF] border-b border-[#374151]">
                      {["Symbol", "Trades", "Win Rate", "PnL%", "Profit Factor", "Avg Hold (h)"].map(
                        (h) => (
                          <th key={h} className="pb-2 pr-4 font-medium">
                            {h}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {data!.by_symbol.map((row) => (
                      <tr
                        key={row.symbol}
                        className="border-b border-[#374151]/40 hover:bg-white/5 transition-colors"
                      >
                        <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{row.symbol}</td>
                        <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{row.trades}</td>
                        <td className={`py-2 pr-4 font-mono ${row.win_rate >= 0.5 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                          {fmt(row.win_rate * 100)}%
                        </td>
                        <td className={`py-2 pr-4 font-mono ${pnlColor(row.pnl_pct)}`}>
                          {pct(row.pnl_pct)}
                        </td>
                        <td className={`py-2 pr-4 font-mono ${row.profit_factor >= 1 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                          {fmt(row.profit_factor)}
                        </td>
                        <td className="py-2 pr-4 font-mono text-[#9CA3AF]">
                          {fmt(row.avg_holding_h, 1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      {/* Heatmap placeholders (always shown) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <HeatmapPlaceholder title="Hour of Day Win Rate" />
        <HeatmapPlaceholder title="Day of Week Win Rate" />
      </div>
    </div>
  );
}
