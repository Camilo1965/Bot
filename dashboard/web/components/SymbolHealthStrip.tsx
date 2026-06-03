"use client";

import { useEffect, useState } from "react";

interface ModelInfo {
  symbol: string;
  has_long_model: boolean;
  has_short_model: boolean;
  meta?: { metrics?: { auc?: number } };
}

export function SymbolHealthStrip({ apiBase }: { apiBase: string }) {
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    fetch(`${apiBase}/api/models`)
      .then((r) => r.json())
      .then(setModels)
      .catch(() => {});
  }, [apiBase]);

  if (!models.length) return null;

  return (
    <div className="bg-bg-surface border border-border-default rounded-card p-4">
      <h2 className="text-xs uppercase tracking-widest text-text-secondary mb-3">Symbol Health</h2>
      <div className="flex flex-wrap gap-2">
        {models.map((m) => {
          const auc = m.meta?.metrics?.auc;
          const aucOk = auc === undefined || auc >= 0.57;
          return (
            <div
              key={m.symbol}
              className={`px-3 py-1.5 rounded-lg border text-xs font-mono ${
                m.has_long_model && aucOk
                  ? "border-pnl-positive/40 text-pnl-positive"
                  : "border-pnl-negative/40 text-pnl-negative"
              }`}
            >
              {m.symbol.replace("/USDT", "")}
              {auc && <span className="ml-1 text-text-secondary">AUC {auc.toFixed(2)}</span>}
              {m.has_short_model && <span className="ml-1 text-accent-purple text-[10px]">SHORT</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
