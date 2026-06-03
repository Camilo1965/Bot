"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface DataPoint {
  ts: string;
  equity: number;
}

export function EquityChart({ data }: { data: DataPoint[] }) {
  if (!data.length) {
    return <div className="h-48 flex items-center justify-center text-text-secondary text-sm">No equity data yet</div>;
  }

  const formatted = data.map((d, i) => ({
    index: i,
    equity: d.equity,
    label: new Date(d.ts).toLocaleDateString(),
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={formatted} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="label" tick={{ fill: "#9CA3AF", fontSize: 11 }} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: "#9CA3AF", fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v.toFixed(0)}`} />
        <Tooltip
          contentStyle={{ background: "#10151D", border: "1px solid #374151", borderRadius: 8 }}
          labelStyle={{ color: "#9CA3AF" }}
          itemStyle={{ color: "#10B981" }}
          formatter={(v: number) => [`$${v.toFixed(2)}`, "Equity"]}
        />
        <Line type="monotone" dataKey="equity" stroke="#3B82F6" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
