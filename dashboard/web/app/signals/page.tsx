"use client";

import { useEffect, useMemo, useState } from "react";
import { TradingViewChart } from "@/components/TradingViewChart";
import { TradeLifecycleExplainer } from "@/components/TradeLifecycleExplainer";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { useWs } from "@/hooks/WsProvider";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { fmtTime, TABULAR } from "@/lib/format";

import { API_BASE as API } from "@/lib/api";

const SYMBOLS_FALLBACK = [
  "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "NEAR/USDT",
];

const INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"] as const;
type IntervalKey = (typeof INTERVALS)[number];

interface SignalRow {
  ts: string;
  cal_prob: number;
  threshold: number;
  ml_decision: "long" | "short" | "no_trade" | string;
  regime?: string;
}

interface SymbolSignals {
  symbol: string;
  rows: SignalRow[];
  loading: boolean;
  error: string | null;
}

function DecisionBadge({ decision }: { decision: string }) {
  const map: Record<string, string> = {
    long: "bg-[#10B981]/15 text-[#10B981]",
    short: "bg-[#EF4444]/15 text-[#EF4444]",
    no_trade: "bg-[#374151] text-[#9CA3AF]",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${map[decision] ?? map.no_trade}`}>
      {decision.toUpperCase()}
    </span>
  );
}

function ProbBar({ prob, threshold }: { prob: number; threshold: number }) {
  const passes = prob >= threshold;
  const color = passes ? "#10B981" : "#9CA3AF";
  return (
    <div className="relative h-1.5 bg-[#0a0e15] rounded-full overflow-visible">
      <div
        className="absolute h-1.5 rounded-full"
        style={{ width: `${Math.min(prob * 100, 100)}%`, backgroundColor: color }}
      />
      <div
        className="absolute top-[-3px] w-px h-3 bg-[#F59E0B]"
        style={{ left: `${Math.min(threshold * 100, 100)}%` }}
        title={`Threshold: ${threshold.toFixed(2)}`}
      />
    </div>
  );
}

function SymbolColumn({ data }: { data: SymbolSignals }) {
  const latest = data.rows[0];

  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-3 space-y-2 hover:border-[#4B5563] transition-colors">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[#F3F4F6] text-xs font-semibold">{data.symbol}</span>
        {latest && <DecisionBadge decision={latest.ml_decision} />}
      </div>

      {data.loading ? (
        <Skeleton className="h-10" />
      ) : data.error ? (
        <div className="text-[10px] text-[#EF4444]">err</div>
      ) : !latest ? (
        <div className="text-[10px] text-[#9CA3AF]">No data</div>
      ) : (
        <>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-[#9CA3AF]">cal_prob</span>
              <span className={`font-mono ${TABULAR} text-[#F3F4F6]`}>{latest.cal_prob.toFixed(3)}</span>
            </div>
            <ProbBar prob={latest.cal_prob} threshold={latest.threshold} />
            <div className={`flex items-center justify-between text-[9px] font-mono ${TABULAR} text-[#6B7280]`}>
              <span>0</span>
              <span>th {latest.threshold.toFixed(2)}</span>
              <span>1</span>
            </div>
          </div>
        </>
      )}

      {!data.loading && data.rows.length > 1 && (
        <div className="border-t border-[#374151] pt-2 space-y-1">
          <div className="text-[9px] text-[#6B7280] uppercase tracking-wider">Recent</div>
          {data.rows.slice(1, 5).map((r, i) => (
            <div key={i} className="flex justify-between items-center text-[10px]">
              <span className={`text-[#9CA3AF] font-mono ${TABULAR}`}>{fmtTime(r.ts)}</span>
              <DecisionBadge decision={r.ml_decision} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

type ChartSize = "S" | "M" | "L" | "XL";

const SIZE_TO_CSS: Record<ChartSize, string> = {
  S: "560px",
  M: "calc(100vh - 220px)",
  L: "calc(100vh - 80px)",
  XL: "calc(100vh - 80px)",
};

export default function SignalsPage() {
  const [columns, setColumns] = useState<SymbolSignals[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useLocalStorage<string>("clawdbot:signals:symbol", "BTC/USDT");
  const [chartInterval, setChartInterval] = useLocalStorage<IntervalKey>("clawdbot:signals:interval", "15m");
  const [chartSize, setChartSize] = useLocalStorage<ChartSize>("clawdbot:signals:size", "M");
  const ws = useWs();

  const latestForSelected = useMemo(() => {
    return columns.find((c) => c.symbol === selectedSymbol)?.rows[0];
  }, [columns, selectedSymbol]);

  // Live push from WS — merge into existing columns
  useEffect(() => {
    const updates = ws.signals;
    if (Object.keys(updates).length === 0) return;
    setColumns((prev) =>
      prev.map((col) => {
        const u = updates[col.symbol];
        if (!u) return col;
        const newRow: SignalRow = {
          ts: u.ts,
          cal_prob: u.cal_prob,
          threshold: u.threshold,
          ml_decision: (u.decision as any) ?? "no_trade",
        };
        // Avoid duplicates with existing first row
        if (col.rows[0]?.ts === newRow.ts) return col;
        return { ...col, rows: [newRow, ...col.rows].slice(0, 10) };
      })
    );
  }, [ws.signals]);

  useEffect(() => {
    fetch(`${API}/api/symbols/config`)
      .then((r) => r.json())
      .then((cfg) => {
        const symbols: string[] = Array.isArray(cfg) && cfg.length > 0
          ? cfg.filter((c: any) => c.enabled !== false).map((c: any) => c.symbol)
          : SYMBOLS_FALLBACK;
        const initial = symbols.map((s) => ({ symbol: s, rows: [] as SignalRow[], loading: true, error: null as string | null }));
        setColumns(initial);
        if (symbols.length > 0 && !symbols.includes(selectedSymbol)) {
          setSelectedSymbol(symbols[0]);
        }
        symbols.forEach((sym, idx) => {
          const slug = sym.replace("/", "_");
          fetch(`${API}/api/signals/by-symbol/${slug}?limit=10`)
            .then((r) => r.json())
            .then((data) => {
              setColumns((prev) =>
                prev.map((c, i) =>
                  i === idx ? { ...c, rows: Array.isArray(data) ? data : [], loading: false } : c
                )
              );
            })
            .catch((e) => {
              setColumns((prev) =>
                prev.map((c, i) =>
                  i === idx ? { ...c, loading: false, error: String(e) } : c
                )
              );
            });
        });
      })
      .catch(() => {
        const initial = SYMBOLS_FALLBACK.map((s) => ({ symbol: s, rows: [] as SignalRow[], loading: false, error: "config_fetch_failed" }));
        setColumns(initial);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Signals</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">
            Live ML probabilities per symbol with TradingView chart.
          </p>
          <details className="mt-1 text-xs">
            <summary className="cursor-pointer text-[#3B82F6] hover:text-[#60A5FA] inline-block">How does the bot decide?</summary>
            <div className="mt-3">
              <TradeLifecycleExplainer />
            </div>
          </details>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          {latestForSelected && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#10151D] border border-[#374151]">
              <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wider">Prob</span>
              <span className={`font-mono ${TABULAR} text-sm font-semibold ${
                latestForSelected.cal_prob >= latestForSelected.threshold ? "text-[#10B981]" : "text-[#F3F4F6]"
              }`}>
                {(latestForSelected.cal_prob * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-[#6B7280]">/</span>
              <span className={`font-mono ${TABULAR} text-xs text-[#F59E0B]`}>
                th {(latestForSelected.threshold * 100).toFixed(0)}%
              </span>
              <DecisionBadge decision={latestForSelected.ml_decision} />
            </div>
          )}
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            aria-label="Chart symbol"
            className="bg-[#0a0e15] border border-[#374151] rounded px-3 py-1.5 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
          >
            {columns.map((c) => (
              <option key={c.symbol} value={c.symbol}>{c.symbol}</option>
            ))}
          </select>
          <div className="flex gap-0.5 bg-[#10151D] border border-[#374151] rounded-md p-0.5">
            {INTERVALS.map((tf) => (
              <button
                key={tf}
                onClick={() => setChartInterval(tf)}
                className={`px-2 py-1 text-[11px] rounded transition-colors font-mono ${TABULAR} ${
                  chartInterval === tf
                    ? "bg-[#3B82F6]/20 text-[#3B82F6]"
                    : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
          <div className="flex gap-0.5 bg-[#10151D] border border-[#374151] rounded-md p-0.5" title="Chart size">
            {(["S", "M", "L", "XL"] as ChartSize[]).map((s) => (
              <button
                key={s}
                onClick={() => setChartSize(s)}
                className={`w-8 py-1 text-[11px] rounded transition-colors font-mono ${
                  chartSize === s
                    ? "bg-[#3B82F6]/20 text-[#3B82F6]"
                    : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                }`}
                title={`${s} = ${SIZE_TO_CSS[s]}`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {selectedSymbol && (
        <div className="bg-[#10151D] border border-[#374151] rounded-lg overflow-hidden">
          <TradingViewChart
            key={`${selectedSymbol}-${chartInterval}-${chartSize}`}
            symbol={selectedSymbol}
            interval={chartInterval}
            height={SIZE_TO_CSS[chartSize]}
          />
        </div>
      )}

      {chartSize !== "XL" && (
        <div>
          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
            <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Per-symbol decisions</h2>
            <div className="flex items-center gap-3 text-[10px] text-[#9CA3AF] font-mono flex-wrap">
              <span className="inline-flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#10B981]/40" /> BUY <span className="text-[#6B7280]">prob ≥ threshold</span>
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#F59E0B]/40" /> SKIP_SPREAD <span className="text-[#6B7280]">spread &gt; 0.8%</span>
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#F59E0B]/40" /> SKIP_VOL <span className="text-[#6B7280]">vol &lt; 0.5×sma</span>
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2 h-2 rounded-sm bg-[#F59E0B]/40" /> SKIP_HOUR <span className="text-[#6B7280]">UTC 22–08</span>
              </span>
              <span className="text-[#6B7280]">· {columns.length} symbols</span>
            </div>
          </div>
          {columns.length === 0 ? (
            <EmptyState icon="↯" title="No symbol configs available" />
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-2">
              {columns.map((col) => (
                <button
                  key={col.symbol}
                  onClick={() => setSelectedSymbol(col.symbol)}
                  className={`text-left transition-colors ${
                    col.symbol === selectedSymbol ? "ring-1 ring-[#3B82F6]/60 rounded-lg" : ""
                  }`}
                >
                  <SymbolColumn data={col} />
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
