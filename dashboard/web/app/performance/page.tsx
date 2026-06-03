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

interface HourBucket { hour: number; win_rate: number; count: number; }
interface DowBucket  { dow: number; day_name: string; win_rate: number; count: number; }

function winRateColor(wr: number, count: number): string {
  if (count === 0) return "#1E2530";
  if (wr >= 0.7)  return "#10B981";
  if (wr >= 0.55) return "#34d399";
  if (wr >= 0.45) return "#F59E0B";
  if (wr >= 0.35) return "#f97316";
  return "#EF4444";
}

function HourHeatmap({ data }: { data: HourBucket[] }) {
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
      <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-4">Hour of Day — Win Rate</div>
      <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(24, 1fr)" }}>
        {data.map((b) => (
          <div key={b.hour} className="flex flex-col items-center gap-1">
            <div
              className="w-full rounded-sm transition-colors"
              style={{ height: 32, backgroundColor: winRateColor(b.win_rate, b.count) }}
              title={`${b.hour}:00 — WR ${(b.win_rate * 100).toFixed(0)}% (${b.count} trades)`}
            />
            <span className="text-[9px] text-[#9CA3AF] font-mono tabular-nums">
              {String(b.hour).padStart(2, "0")}
            </span>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-3 mt-3">
        {[["#10B981","≥70%"],["#F59E0B","45–55%"],["#EF4444","<35%"],["#1E2530","No data"]].map(([c,l]) => (
          <div key={l} className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: c }} />
            <span className="text-[10px] text-[#9CA3AF]">{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DowHeatmap({ data }: { data: DowBucket[] }) {
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
      <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-4">Day of Week — Win Rate</div>
      <div className="grid grid-cols-7 gap-2">
        {data.map((b) => (
          <div key={b.dow} className="flex flex-col items-center gap-2">
            <div
              className="w-full rounded-md flex flex-col items-center justify-center transition-colors"
              style={{ height: 64, backgroundColor: winRateColor(b.win_rate, b.count) }}
              title={`${b.day_name} — WR ${(b.win_rate * 100).toFixed(0)}% (${b.count} trades)`}
            >
              <span className="text-xs font-mono font-bold text-white/90 tabular-nums">
                {b.count > 0 ? `${(b.win_rate * 100).toFixed(0)}%` : "—"}
              </span>
              <span className="text-[9px] text-white/60">{b.count}t</span>
            </div>
            <span className="text-[10px] text-[#9CA3AF]">{b.day_name.slice(0, 3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function PerformancePage() {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [hourData, setHourData] = useState<HourBucket[]>([]);
  const [dowData, setDowData]   = useState<DowBucket[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/performance?range=30d`).then((r) => r.json()),
      fetch(`${API}/api/performance/by-hour`).then((r) => r.json()),
      fetch(`${API}/api/performance/by-dow`).then((r) => r.json()),
    ])
      .then(([perf, hours, dows]) => {
        setData(perf);
        setHourData(hours);
        setDowData(dows);
        setLoading(false);
      })
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

      {/* Heatmaps (always shown, show empty cells when no trades yet) */}
      {hourData.length > 0 && <HourHeatmap data={hourData} />}
      {dowData.length > 0 && <DowHeatmap data={dowData} />}
    </div>
  );
}
