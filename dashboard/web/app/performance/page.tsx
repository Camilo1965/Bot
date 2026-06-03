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
interface PnlHistogram { bin_edges: number[]; counts: number[]; stats: { n: number; mean: number; median: number; min: number; max: number; wins: number; losses: number; win_rate: number; } }
interface SharpePoint { ts: string; sharpe: number; n: number; }

function PnlHistogramChart({ data }: { data: PnlHistogram | null }) {
  if (!data || data.counts.length === 0) {
    return <div className="text-[#9CA3AF] text-sm text-center py-6">No PnL data.</div>;
  }
  const max = Math.max(...data.counts, 1);
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-2 text-[10px] text-[#9CA3AF]">
        <div>n={data.stats.n}</div>
        <div>μ={data.stats.mean}</div>
        <div>WR={(data.stats.win_rate * 100).toFixed(1)}%</div>
        <div>range [{data.stats.min}, {data.stats.max}]</div>
      </div>
      <div className="flex items-end gap-px h-32 border-b border-[#374151]">
        {data.counts.map((c, i) => {
          const mid = (data.bin_edges[i] + data.bin_edges[i + 1]) / 2;
          const color = mid >= 0 ? "bg-[#10B981]" : "bg-[#EF4444]";
          const h = (c / max) * 100;
          return (
            <div key={i} className={`flex-1 ${color} opacity-80 hover:opacity-100 transition-opacity`} style={{ height: `${h}%` }}
              title={`${data.bin_edges[i].toFixed(2)} → ${data.bin_edges[i + 1].toFixed(2)}: ${c} trades`} />
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-[#9CA3AF] font-mono">
        <span>{data.bin_edges[0]?.toFixed(1)}</span>
        <span>0</span>
        <span>{data.bin_edges[data.bin_edges.length - 1]?.toFixed(1)}</span>
      </div>
    </div>
  );
}

function RollingSharpeChart({ data }: { data: SharpePoint[] }) {
  if (data.length === 0) {
    return <div className="text-[#9CA3AF] text-sm text-center py-6">Need at least 20 trades to compute rolling Sharpe.</div>;
  }
  const max = Math.max(...data.map((d) => d.sharpe), 0);
  const min = Math.min(...data.map((d) => d.sharpe), 0);
  const range = Math.max(max - min, 0.01);
  return (
    <div className="space-y-2">
      <div className="flex items-end gap-px h-32 border-b border-l border-[#374151] relative">
        {/* Zero baseline */}
        <div className="absolute left-0 right-0 border-t border-dashed border-[#9CA3AF]/40"
             style={{ top: `${((max - 0) / range) * 100}%` }} />
        {data.map((p, i) => {
          const norm = (p.sharpe - min) / range;
          const h = norm * 100;
          const color = p.sharpe >= 0.5 ? "bg-[#10B981]" : p.sharpe >= 0 ? "bg-[#3B82F6]" : "bg-[#EF4444]";
          return (
            <div key={i} className={`flex-1 ${color} opacity-80`} style={{ height: `${h}%` }}
              title={`${new Date(p.ts).toLocaleDateString()}: Sharpe ${p.sharpe}`} />
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-[#9CA3AF] font-mono">
        <span>{data[0] ? new Date(data[0].ts).toLocaleDateString() : ""}</span>
        <span>{data[data.length - 1] ? new Date(data[data.length - 1].ts).toLocaleDateString() : ""}</span>
      </div>
      <div className="text-[10px] text-[#9CA3AF]">window=20 trades · annualized</div>
    </div>
  );
}

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
  const [pnlHist, setPnlHist]   = useState<PnlHistogram | null>(null);
  const [sharpe, setSharpe]     = useState<SharpePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/performance?range=30d`).then((r) => r.json()),
      fetch(`${API}/api/performance/by-hour`).then((r) => r.json()),
      fetch(`${API}/api/performance/by-dow`).then((r) => r.json()),
      fetch(`${API}/api/performance/pnl-histogram?bins=20&days=90`).then((r) => r.json()),
      fetch(`${API}/api/performance/rolling-sharpe?window=20&days=90`).then((r) => r.json()),
    ])
      .then(([perf, hours, dows, hist, sh]) => {
        setData(perf);
        setHourData(hours);
        setDowData(dows);
        setPnlHist(hist);
        setSharpe(Array.isArray(sh) ? sh : []);
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

      {/* Distribution + Sharpe */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">PnL Distribution (90d)</div>
          <PnlHistogramChart data={pnlHist} />
        </div>
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">Rolling Sharpe (window=20)</div>
          <RollingSharpeChart data={sharpe} />
        </div>
      </div>

      {/* Heatmaps (always shown, show empty cells when no trades yet) */}
      {hourData.length > 0 && <HourHeatmap data={hourData} />}
      {dowData.length > 0 && <DowHeatmap data={dowData} />}
    </div>
  );
}
