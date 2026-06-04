"use client";

import { useEffect, useState } from "react";
import { fmtMoney, fmtPct, pnlColor, TABULAR } from "@/lib/format";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { useWs } from "@/hooks/WsProvider";

interface Position {
  trade_id: string;
  symbol: string;
  side: string;
  entry_price: number;
  current_price: number;
  pnl_usd: number;
  pnl_pct: number;
  sl_price: number;
  tp_price: number;
  size_usd: number;
}

function progress(entry: number, current: number, sl: number, tp: number, side: string): number {
  const isLong = side?.toLowerCase() === "long";
  const lo = isLong ? sl : tp;
  const hi = isLong ? tp : sl;
  const range = hi - lo;
  if (!range) return 50;
  const pct = ((current - lo) / range) * 100;
  return Math.max(0, Math.min(100, pct));
}

export function OpenPositionsList({ apiBase }: { apiBase: string }) {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loaded, setLoaded] = useState(false);
  const ws = useWs();

  // Initial load via REST
  useEffect(() => {
    fetch(`${apiBase}/api/positions/open`)
      .then((r) => r.json())
      .then((d) => { setPositions(Array.isArray(d) ? d : []); setLoaded(true); })
      .catch(() => setLoaded(true));
  }, [apiBase]);

  // Live snapshot from WS — overrides
  useEffect(() => {
    if (ws.lastPositionsSnapshot !== null) {
      setPositions(ws.lastPositionsSnapshot as Position[]);
      setLoaded(true);
    }
  }, [ws.lastPositionsSnapshot]);

  if (!loaded) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 2 }).map((_, i) => <Skeleton key={i} className="h-16" />)}
      </div>
    );
  }
  if (!positions.length) {
    return <EmptyState icon="○" title="No open positions" />;
  }

  return (
    <div className="space-y-2">
      {positions.map((p) => {
        const isLong = p.side?.toLowerCase() === "long";
        const sideColor = isLong ? "#10B981" : "#EF4444";
        const pos = progress(p.entry_price, p.current_price, p.sl_price, p.tp_price, p.side);
        const entryPos = progress(p.entry_price, p.entry_price, p.sl_price, p.tp_price, p.side);
        return (
          <div key={p.trade_id} className="bg-[#0a0e15] border border-[#374151] rounded-lg p-3 hover:border-[#4B5563] transition-colors">
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[#F3F4F6] text-sm">{p.symbol}</span>
                <span
                  className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                  style={{ backgroundColor: `${sideColor}22`, color: sideColor }}
                >
                  {p.side.toUpperCase()}
                </span>
              </div>
              <div className="text-right">
                <div className={`font-mono ${TABULAR} text-sm ${pnlColor(p.pnl_usd)}`}>
                  {fmtMoney(p.pnl_usd)}
                </div>
                <div className={`font-mono ${TABULAR} text-[10px] ${pnlColor(p.pnl_pct)} opacity-80`}>
                  {fmtPct(p.pnl_pct)}
                </div>
              </div>
            </div>
            <div className="relative h-1 bg-[#374151] rounded-full overflow-visible mb-1.5">
              <div
                className="absolute h-1 rounded-full transition-all duration-300"
                style={{
                  left: `${Math.min(entryPos, pos)}%`,
                  width: `${Math.abs(pos - entryPos)}%`,
                  backgroundColor: pnlColor(p.pnl_usd) === "text-[#10B981]" ? "#10B981" : "#EF4444",
                  opacity: 0.6,
                }}
              />
              <div
                className="absolute top-[-3px] w-px h-3 bg-[#9CA3AF]"
                style={{ left: `${entryPos}%` }}
                title={`Entry ${p.entry_price.toFixed(4)}`}
              />
              <div
                className="absolute top-[-4px] w-1.5 h-2 rounded-sm transition-all duration-300"
                style={{ left: `calc(${pos}% - 3px)`, backgroundColor: sideColor }}
                title={`Current ${p.current_price.toFixed(4)}`}
              />
            </div>
            <div className={`flex justify-between text-[9px] font-mono ${TABULAR} text-[#6B7280]`}>
              <span className="text-[#EF4444]/70">SL {p.sl_price.toFixed(4)}</span>
              <span>{p.current_price.toFixed(4)}</span>
              <span className="text-[#10B981]/70">TP {p.tp_price.toFixed(4)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
