"use client";

import { useEffect, useState } from "react";
import { KpiCard } from "@/components/ui/KpiCard";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { useRefreshKey } from "@/hooks/useRefreshKey";
import { SortableTable, Column } from "@/components/ui/SortableTable";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonKpi, SkeletonTable } from "@/components/ui/Skeleton";
import { fmtMoney, fmtPct, fmtNum, fmtDate, pnlColor, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PerformanceSummary {
  total_trades: number; win_rate: number; pnl_pct: number;
  profit_factor: number; sharpe: number; max_drawdown_pct: number;
}
interface SymbolPerf {
  symbol: string; trades: number; win_rate: number;
  pnl_pct: number; profit_factor: number; avg_holding_h: number;
}
interface PerformanceData { summary: PerformanceSummary; by_symbol: SymbolPerf[] }
interface HourBucket { hour: number; win_rate: number; count: number }
interface DowBucket  { dow: number; day_name: string; win_rate: number; count: number }
interface PnlHist { bin_edges: number[]; counts: number[]; stats: { n: number; mean: number; median: number; min: number; max: number; wins: number; losses: number; win_rate: number } }
interface SharpePt { ts: string; sharpe: number; n: number }

function wrColor(wr: number, count: number): string {
  if (count === 0) return "#1E2530";
  if (wr >= 0.7) return "#10B981";
  if (wr >= 0.55) return "#34d399";
  if (wr >= 0.45) return "#F59E0B";
  if (wr >= 0.35) return "#f97316";
  return "#EF4444";
}

function HourHeatmap({ data }: { data: HourBucket[] }) {
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
      <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Hour of Day · Win Rate</div>
      <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(24, 1fr)" }}>
        {data.map((b) => (
          <div key={b.hour} className="flex flex-col items-center gap-1 group cursor-pointer">
            <div
              className="w-full rounded-sm transition-all group-hover:ring-1 ring-[#3B82F6]"
              style={{ height: 32, backgroundColor: wrColor(b.win_rate, b.count) }}
              title={`${b.hour}:00 UTC · WR ${(b.win_rate * 100).toFixed(0)}% · ${b.count} trades`}
            />
            <span className={`text-[9px] text-[#9CA3AF] font-mono ${TABULAR}`}>
              {String(b.hour).padStart(2, "0")}
            </span>
          </div>
        ))}
      </div>
      <Legend />
    </div>
  );
}

function DowHeatmap({ data }: { data: DowBucket[] }) {
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
      <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Day of Week · Win Rate</div>
      <div className="grid grid-cols-7 gap-2">
        {data.map((b) => (
          <div key={b.dow} className="flex flex-col items-center gap-2 group cursor-pointer">
            <div
              className="w-full rounded-md transition-all group-hover:ring-1 ring-[#3B82F6] flex items-center justify-center font-mono text-xs font-semibold text-white"
              style={{ height: 56, backgroundColor: wrColor(b.win_rate, b.count) }}
              title={`${b.day_name} · WR ${(b.win_rate * 100).toFixed(0)}% · ${b.count} trades`}
            >
              {b.count > 0 ? `${(b.win_rate * 100).toFixed(0)}%` : "—"}
            </div>
            <span className="text-[10px] text-[#9CA3AF]">{b.day_name.slice(0, 3)}</span>
            <span className={`text-[9px] font-mono ${TABULAR} text-[#6B7280]`}>n={b.count}</span>
          </div>
        ))}
      </div>
      <Legend />
    </div>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-3 mt-3 flex-wrap">
      {[["#10B981","≥70%"],["#34d399","55-70%"],["#F59E0B","45-55%"],["#EF4444","<35%"],["#1E2530","no data"]].map(([c,l]) => (
        <div key={l} className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: c }} />
          <span className="text-[10px] text-[#9CA3AF]">{l}</span>
        </div>
      ))}
    </div>
  );
}

function PnlHistogramChart({ data }: { data: PnlHist | null }) {
  if (!data || data.counts.length === 0) return <EmptyState icon="∿" title="No PnL data yet" />;
  const max = Math.max(...data.counts, 1);
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-2 text-[10px] text-[#9CA3AF]">
        <div>n={data.stats.n}</div>
        <div className={pnlColor(data.stats.mean)}>μ={fmtMoney(data.stats.mean)}</div>
        <div>WR={(data.stats.win_rate * 100).toFixed(1)}%</div>
        <div className="truncate">[{fmtNum(data.stats.min)}, {fmtNum(data.stats.max)}]</div>
      </div>
      <div className="flex items-end gap-px h-32 border-b border-[#374151] relative">
        {/* Zero line */}
        {(() => {
          const range = data.bin_edges[data.bin_edges.length - 1] - data.bin_edges[0];
          if (range <= 0) return null;
          const zeroPct = ((0 - data.bin_edges[0]) / range) * 100;
          if (zeroPct < 0 || zeroPct > 100) return null;
          return (
            <div
              className="absolute top-0 bottom-0 border-l border-dashed border-[#9CA3AF]/40 z-10"
              style={{ left: `${zeroPct}%` }}
            />
          );
        })()}
        {data.counts.map((c, i) => {
          const mid = (data.bin_edges[i] + data.bin_edges[i + 1]) / 2;
          const color = mid >= 0 ? "bg-[#10B981]" : "bg-[#EF4444]";
          const h = (c / max) * 100;
          return (
            <div key={i} className={`flex-1 ${color} opacity-80 hover:opacity-100 transition-opacity rounded-t-sm`}
              style={{ height: `${h}%` }}
              title={`[${data.bin_edges[i].toFixed(2)}, ${data.bin_edges[i + 1].toFixed(2)}]: ${c} trades`} />
          );
        })}
      </div>
      <div className={`flex justify-between text-[10px] font-mono ${TABULAR} text-[#9CA3AF]`}>
        <span>{fmtNum(data.bin_edges[0])}</span>
        <span>0</span>
        <span>{fmtNum(data.bin_edges[data.bin_edges.length - 1])}</span>
      </div>
    </div>
  );
}

function RollingSharpeChart({ data }: { data: SharpePt[] }) {
  if (data.length === 0) return <EmptyState icon="∿" title="Need 20+ trades for rolling Sharpe" />;
  const max = Math.max(...data.map((d) => d.sharpe), 0.1);
  const min = Math.min(...data.map((d) => d.sharpe), -0.1);
  const range = max - min || 0.01;
  const zeroPct = ((max - 0) / range) * 100;
  const lastSharpe = data[data.length - 1]?.sharpe ?? 0;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-[10px] text-[#9CA3AF]">
        <span>window=20 · annualized</span>
        <span className={`font-mono ${TABULAR} ${pnlColor(lastSharpe - 0.5)}`}>
          current: {lastSharpe.toFixed(2)}
        </span>
      </div>
      <div className="flex items-end gap-px h-32 border-b border-l border-[#374151] relative">
        <div className="absolute left-0 right-0 border-t border-dashed border-[#9CA3AF]/40 z-10"
             style={{ top: `${zeroPct}%` }} />
        {data.map((p, i) => {
          const norm = (p.sharpe - min) / range;
          const h = norm * 100;
          const color = p.sharpe >= 0.5 ? "bg-[#10B981]" : p.sharpe >= 0 ? "bg-[#3B82F6]" : "bg-[#EF4444]";
          return (
            <div key={i} className={`flex-1 ${color} opacity-80 hover:opacity-100`} style={{ height: `${h}%` }}
              title={`${fmtDate(p.ts)} · Sharpe ${p.sharpe.toFixed(3)}`} />
          );
        })}
      </div>
      <div className={`flex justify-between text-[10px] font-mono ${TABULAR} text-[#9CA3AF]`}>
        <span>{data[0] ? fmtDate(data[0].ts) : ""}</span>
        <span>{data[data.length - 1] ? fmtDate(data[data.length - 1].ts) : ""}</span>
      </div>
    </div>
  );
}

export default function PerformancePage() {
  const [data, setData]         = useState<PerformanceData | null>(null);
  const [hourData, setHourData] = useState<HourBucket[]>([]);
  const [dowData, setDowData]   = useState<DowBucket[]>([]);
  const [pnlHist, setPnlHist]   = useState<PnlHist | null>(null);
  const [sharpe, setSharpe]     = useState<SharpePt[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useLocalStorage<"7d" | "30d" | "90d">("clawdbot:perf:range", "30d");
  const [refreshKey] = useRefreshKey();

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/performance?range=${range}`).then((r) => r.json()),
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
  }, [range, refreshKey]);

  const s = data?.summary;

  const symbolCols: Column<SymbolPerf>[] = [
    { key: "symbol", label: "Symbol", sortable: true,
      render: (r) => <span className="font-mono text-[#F3F4F6]">{r.symbol}</span> },
    { key: "trades", label: "Trades", align: "right", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR}`}>{r.trades}</span> },
    { key: "win_rate", label: "Win %", align: "right", sortable: true, sortBy: (r) => r.win_rate,
      render: (r) => <span className={`font-mono ${TABULAR} ${r.win_rate >= 0.5 ? "text-[#10B981]" : "text-[#EF4444]"}`}>{(r.win_rate * 100).toFixed(1)}%</span> },
    { key: "pnl_pct", label: "PnL %", align: "right", sortable: true, sortBy: (r) => r.pnl_pct,
      render: (r) => <span className={`font-mono ${TABULAR} ${pnlColor(r.pnl_pct)}`}>{fmtPct(r.pnl_pct)}</span> },
    { key: "profit_factor", label: "PF", align: "right", sortable: true, sortBy: (r) => r.profit_factor,
      render: (r) => <span className={`font-mono ${TABULAR} ${r.profit_factor >= 1 ? "text-[#10B981]" : "text-[#EF4444]"}`}>{r.profit_factor.toFixed(2)}</span> },
    { key: "avg_holding_h", label: "Avg Hold", align: "right", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#9CA3AF]`}>{r.avg_holding_h.toFixed(1)}h</span> },
  ];

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Performance</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">Aggregate metrics, distribution, and timing edge.</p>
        </div>
        <div className="flex gap-1 bg-[#10151D] border border-[#374151] rounded-md p-0.5">
          {(["7d", "30d", "90d"] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                range === r ? "bg-[#3B82F6]/20 text-[#3B82F6]" : "text-[#9CA3AF] hover:text-[#F3F4F6]"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>}

      {loading || !s ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonKpi key={i} />)}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <KpiCard label="Trades" value={String(s.total_trades)} />
          <KpiCard label="Win Rate" value={`${(s.win_rate * 100).toFixed(1)}%`}
            valueClass={s.win_rate >= 0.5 ? "text-[#10B981]" : "text-[#EF4444]"} />
          <KpiCard label={`PnL (${range})`} value={fmtPct(s.pnl_pct)} valueClass={pnlColor(s.pnl_pct)} />
          <KpiCard label="Profit Factor" value={s.profit_factor.toFixed(2)}
            valueClass={s.profit_factor >= 1 ? "text-[#10B981]" : "text-[#EF4444]"}
            hint="Gross win / |Gross loss|" />
          <KpiCard label="Sharpe" value={s.sharpe.toFixed(2)} hint="Annualized" />
          <KpiCard label="Max DD" value={`-${s.max_drawdown_pct.toFixed(2)}%`}
            valueClass="text-[#EF4444]" hint="Peak-to-trough drawdown" />
        </div>
      )}

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">
          Per-Symbol Breakdown ({range})
        </div>
        {loading ? <SkeletonTable rows={5} cols={6} /> : (
          (data?.by_symbol?.length ?? 0) === 0 ? (
            <EmptyState icon="▥" title="No symbol data yet" />
          ) : (
            <SortableTable
              rows={data!.by_symbol}
              columns={symbolCols}
              rowKey={(r) => r.symbol}
              initialSort={{ col: "pnl_pct", dir: "desc" }}
              compact
            />
          )
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">
            PnL Distribution (90d)
          </div>
          <PnlHistogramChart data={pnlHist} />
        </div>
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">
            Rolling Sharpe
          </div>
          <RollingSharpeChart data={sharpe} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {hourData.length > 0 && <HourHeatmap data={hourData} />}
        {dowData.length > 0 && <DowHeatmap data={dowData} />}
      </div>
    </div>
  );
}
