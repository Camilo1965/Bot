"use client";

import { useEffect, useState } from "react";

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

export function OpenPositionsList({ apiBase }: { apiBase: string }) {
  const [positions, setPositions] = useState<Position[]>([]);

  useEffect(() => {
    const load = () =>
      fetch(`${apiBase}/api/positions/open`)
        .then((r) => r.json())
        .then(setPositions)
        .catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [apiBase]);

  if (!positions.length) {
    return <p className="text-text-secondary text-sm">No open positions</p>;
  }

  return (
    <div className="space-y-2">
      {positions.map((p) => (
        <div key={p.trade_id} className="bg-bg-elevated rounded-lg p-3 text-xs font-mono">
          <div className="flex justify-between items-center">
            <span className="font-bold text-text-primary">{p.symbol}</span>
            <span className={p.pnl_usd >= 0 ? "text-pnl-positive" : "text-pnl-negative"}>
              {p.pnl_usd >= 0 ? "+" : ""}{p.pnl_usd.toFixed(2)} USD
            </span>
          </div>
          <div className="flex gap-3 text-text-secondary mt-1">
            <span>Entry: {p.entry_price.toFixed(4)}</span>
            <span>SL: {p.sl_price.toFixed(4)}</span>
            <span>TP: {p.tp_price.toFixed(4)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
