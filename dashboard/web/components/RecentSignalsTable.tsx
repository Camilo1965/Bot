"use client";

import { useEffect, useState } from "react";

interface Signal {
  timestamp: string;
  symbol: string;
  cal_prob: number;
  threshold: number;
  ml_decision: string;
  executed: string;
}

export function RecentSignalsTable({ apiBase }: { apiBase: string }) {
  const [signals, setSignals] = useState<Signal[]>([]);

  useEffect(() => {
    const load = () =>
      fetch(`${apiBase}/api/signals/recent?limit=20`)
        .then((r) => r.json())
        .then(setSignals)
        .catch(() => {});
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [apiBase]);

  if (!signals.length) {
    return <p className="text-text-secondary text-sm">No signals yet</p>;
  }

  return (
    <div className="overflow-auto max-h-64">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-text-secondary border-b border-border-default">
            <th className="text-left pb-2">Symbol</th>
            <th className="text-right pb-2">Prob</th>
            <th className="text-right pb-2">Thr</th>
            <th className="text-center pb-2">Decision</th>
            <th className="text-center pb-2">Exec</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <tr key={i} className="border-b border-border-default/50 hover:bg-bg-elevated">
              <td className="py-1.5 text-text-primary">{s.symbol}</td>
              <td className="py-1.5 text-right">{(parseFloat(String(s.cal_prob)) * 100).toFixed(1)}%</td>
              <td className="py-1.5 text-right text-text-secondary">{(parseFloat(String(s.threshold)) * 100).toFixed(0)}%</td>
              <td className={`py-1.5 text-center ${s.ml_decision === "BUY" ? "text-pnl-positive" : "text-text-secondary"}`}>
                {s.ml_decision}
              </td>
              <td className={`py-1.5 text-center ${s.executed === "True" ? "text-pnl-positive" : "text-text-secondary"}`}>
                {s.executed === "True" ? "✓" : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
