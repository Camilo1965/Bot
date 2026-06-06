"use client";

import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { LivePositionCard, type LivePosition } from "@/components/LivePositionCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { useSessionEquity, usePositionPriceBuffers } from "@/hooks/useSessionEquity";
import { useWs } from "@/hooks/WsProvider";
import { fmtMoney, pnlColor, TABULAR } from "@/lib/format";

interface Props {
  startingBalance?: number;
}

export function LiveTradesPanel({ startingBalance }: Props) {
  const ws = useWs();
  const sessionPoints = useSessionEquity();
  const priceBuffers = usePositionPriceBuffers();

  const positions: LivePosition[] = useMemo(() => {
    const snap = ws.lastPositionsSnapshot;
    if (!Array.isArray(snap)) return [];
    return snap as LivePosition[];
  }, [ws.lastPositionsSnapshot]);

  if (positions.length > 0) {
    const totalPnl = positions.reduce((acc, p) => acc + (p.pnl_usd ?? 0), 0);
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">
            Live trades ({positions.length})
          </h2>
          <span className={`font-mono ${TABULAR} text-sm ${pnlColor(totalPnl)}`}>
            unrealized {fmtMoney(totalPnl)}
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {positions.map((p) => (
            <LivePositionCard
              key={p.trade_id || p.symbol}
              position={p}
              priceHistory={priceBuffers[p.trade_id || p.symbol] || []}
            />
          ))}
        </div>
      </div>
    );
  }

  if (sessionPoints.length < 2) {
    return (
      <EmptyState
        icon="∿"
        title="Waiting for live data"
        body="The chart fills once the bot publishes equity ticks (every ~1s). When a trade opens, individual mini-charts appear here with entry/SL/TP markers."
      />
    );
  }

  const formatted = sessionPoints.map((p, i) => ({
    index: i,
    equity: p.equity,
    balance: p.balance,
    label: new Date(p.ts).toLocaleTimeString("en-GB", { hour12: false }),
  }));
  const base = startingBalance ?? sessionPoints[0]?.balance ?? 10000;
  const yMin = Math.min(...sessionPoints.map((p) => p.equity), base);
  const yMax = Math.max(...sessionPoints.map((p) => p.equity), base);
  const pad = Math.max(1, (yMax - yMin) * 0.15);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">
          Session equity (live · {sessionPoints.length}pt)
        </h2>
        <span className="text-[10px] text-[#6B7280] font-mono">starts at {fmtMoney(base)}</span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={formatted} margin={{ top: 8, right: 12, bottom: 0, left: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="label" tick={{ fill: "#9CA3AF", fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
          <YAxis
            domain={[yMin - pad, yMax + pad]}
            tick={{ fill: "#9CA3AF", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
          />
          <Tooltip
            contentStyle={{ background: "#10151D", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#9CA3AF" }}
            itemStyle={{ color: "#10B981" }}
            formatter={(v: number) => [`$${v.toFixed(2)}`, "Equity"]}
          />
          <ReferenceLine y={base} stroke="#9CA3AF" strokeOpacity={0.4} strokeDasharray="4 4" />
          <Line type="monotone" dataKey="equity" stroke="#3B82F6" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
