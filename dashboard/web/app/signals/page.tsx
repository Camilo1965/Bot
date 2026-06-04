"use client";

import { useEffect, useState } from "react";
import { TradingViewChart } from "@/components/TradingViewChart";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { fmtTime, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SYMBOLS_FALLBACK = [
  "BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT",
  "NEAR/USDT", "LINK/USDT", "JTO/USDT", "INJ/USDT",
];

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

export default function SignalsPage() {
  const [columns, setColumns] = useState<SymbolSignals[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTC/USDT");

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
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Signals</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">Live ML probabilities per symbol with TradingView chart.</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-[#9CA3AF]">Chart:</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="bg-[#0a0e15] border border-[#374151] rounded px-2 py-1 text-xs text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
          >
            {columns.map((c) => (
              <option key={c.symbol} value={c.symbol}>{c.symbol}</option>
            ))}
          </select>
        </div>
      </div>

      {selectedSymbol && (
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-2 overflow-hidden">
          <TradingViewChart symbol={selectedSymbol} interval="15m" height={460} />
        </div>
      )}

      {columns.length === 0 ? (
        <EmptyState icon="↯" title="No symbol configs available" />
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
          {columns.map((col) => (
            <SymbolColumn key={col.symbol} data={col} />
          ))}
        </div>
      )}
    </div>
  );
}
