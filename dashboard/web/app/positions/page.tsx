"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Tab = "open" | "closed" | "all";

interface Position {
  id: string;
  symbol: string;
  side: "long" | "short";
  entry_price: number;
  current_price: number;
  sl_price: number;
  tp_price: number;
  size: number;
  pnl_usd: number;
  pnl_pct: number;
  opened_at?: string;
  closed_at?: string;
  close_reason?: string;
}

const TABS: { key: Tab; label: string }[] = [
  { key: "open", label: "Open" },
  { key: "closed", label: "Closed" },
  { key: "all", label: "All" },
];

function PnlCell({ value }: { value: number }) {
  const color = value >= 0 ? "text-[#10B981]" : "text-[#EF4444]";
  return (
    <span className={`font-mono ${color}`}>
      {value >= 0 ? "+" : ""}
      {value.toFixed(2)}
    </span>
  );
}

function PositionTable({ rows }: { rows: Position[] }) {
  if (rows.length === 0) {
    return (
      <div className="text-center text-[#9CA3AF] py-10 text-sm">No positions found.</div>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-widest text-[#9CA3AF] border-b border-[#374151]">
            {["Symbol", "Side", "Entry", "Current", "SL", "TP", "Size", "PnL"].map((h) => (
              <th key={h} className="pb-2 pr-4 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr
              key={p.id}
              className="border-b border-[#374151]/40 hover:bg-white/5 transition-colors"
            >
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{p.symbol}</td>
              <td className="py-2 pr-4">
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                    p.side === "long"
                      ? "bg-[#10B981]/20 text-[#10B981]"
                      : "bg-[#EF4444]/20 text-[#EF4444]"
                  }`}
                >
                  {p.side.toUpperCase()}
                </span>
              </td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{p.entry_price.toFixed(4)}</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{p.current_price.toFixed(4)}</td>
              <td className="py-2 pr-4 font-mono text-[#9CA3AF]">{p.sl_price.toFixed(4)}</td>
              <td className="py-2 pr-4 font-mono text-[#9CA3AF]">{p.tp_price.toFixed(4)}</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{p.size.toFixed(4)}</td>
              <td className="py-2 pr-4">
                <PnlCell value={p.pnl_usd} />
                <span className="text-[#9CA3AF] text-[11px] ml-1">
                  (<PnlCell value={p.pnl_pct} />%)
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function PositionsPage() {
  const [tab, setTab] = useState<Tab>("open");
  const [openRows, setOpenRows] = useState<Position[]>([]);
  const [closedRows, setClosedRows] = useState<Position[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints: Record<Tab, string> = {
      open: `${API}/api/positions/open`,
      closed: `${API}/api/positions/history`,
      all: `${API}/api/positions/open`,
    };

    const fetches =
      tab === "all"
        ? Promise.all([
            fetch(`${API}/api/positions/open`).then((r) => r.json()),
            fetch(`${API}/api/positions/history`).then((r) => r.json()),
          ]).then(([o, c]) => {
            setOpenRows(Array.isArray(o) ? o : []);
            setClosedRows(Array.isArray(c) ? c : []);
          })
        : fetch(endpoints[tab])
            .then((r) => r.json())
            .then((data) => {
              if (tab === "open") setOpenRows(Array.isArray(data) ? data : []);
              else setClosedRows(Array.isArray(data) ? data : []);
            });

    fetches.catch((e) => setError(String(e))).finally(() => setLoading(false));
  }, [tab]);

  const rows =
    tab === "open" ? openRows : tab === "closed" ? closedRows : [...openRows, ...closedRows];

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Positions</h1>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-[#374151]">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
              tab === t.key
                ? "border-[#3B82F6] text-[#F3F4F6]"
                : "border-transparent text-[#9CA3AF] hover:text-[#F3F4F6]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        {loading && (
          <div className="text-[#9CA3AF] text-sm text-center py-8">Loading positions…</div>
        )}
        {error && (
          <div className="text-[#EF4444] text-sm text-center py-4">
            Failed to load: {error}
          </div>
        )}
        {!loading && !error && <PositionTable rows={rows} />}
      </div>
    </div>
  );
}
