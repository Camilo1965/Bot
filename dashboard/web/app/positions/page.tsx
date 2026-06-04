"use client";

import { useCallback, useEffect, useState } from "react";
import { SortableTable, Column } from "@/components/ui/SortableTable";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { useRefreshKey } from "@/hooks/useRefreshKey";
import { fmtMoney, fmtPct, fmtTime, pnlColor, TABULAR } from "@/lib/format";

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

function SideBadge({ side }: { side: "long" | "short" }) {
  const c = side === "long" ? "bg-[#10B981]/15 text-[#10B981]" : "bg-[#EF4444]/15 text-[#EF4444]";
  return <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${c}`}>{side.toUpperCase()}</span>;
}

function CloseButton({ symbol, onClosed }: { symbol: string; onClosed: () => void }) {
  const [state, setState] = useState<"idle" | "sent" | "error">("idle");
  const toast = useToast();
  async function handleClose() {
    setState("sent");
    const sym = symbol.replace("/", "_");
    try {
      const res = await fetch(`${API}/api/positions/${sym}/close`, { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        toast.push({ type: "success", title: `Close queued: ${symbol}`, body: "Signal sent to bot." });
        onClosed();
      } else {
        setState("error");
        toast.push({ type: "error", title: `Close failed: ${symbol}` });
      }
    } catch {
      setState("error");
      toast.push({ type: "error", title: `Close failed: ${symbol}` });
    }
  }
  return (
    <button
      onClick={handleClose}
      disabled={state === "sent"}
      className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
        state === "sent"
          ? "bg-[#F59E0B]/15 text-[#F59E0B]"
          : state === "error"
          ? "bg-[#EF4444]/15 text-[#EF4444]"
          : "bg-[#374151] text-[#9CA3AF] hover:bg-[#EF4444]/15 hover:text-[#EF4444]"
      }`}
    >
      {state === "sent" ? "QUEUED" : state === "error" ? "ERR" : "CLOSE"}
    </button>
  );
}

function PnlCell({ usd, pct }: { usd: number; pct: number }) {
  return (
    <div className="flex flex-col items-end leading-tight">
      <span className={`font-mono ${TABULAR} ${pnlColor(usd)}`}>{fmtMoney(usd)}</span>
      <span className={`text-[10px] font-mono ${TABULAR} ${pnlColor(pct)} opacity-80`}>{fmtPct(pct)}</span>
    </div>
  );
}

export default function PositionsPage() {
  const [tab, setTab] = useLocalStorage<Tab>("clawdbot:positions:tab", "open");
  const [openRows, setOpenRows] = useState<Position[]>([]);
  const [closedRows, setClosedRows] = useState<Position[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, refresh] = useRefreshKey();

  useEffect(() => {
    setLoading(true);
    setError(null);
    const fetchOpen = () => fetch(`${API}/api/positions/open`).then((r) => r.json());
    const fetchClosed = () => fetch(`${API}/api/positions/history`).then((r) => r.json());
    const job =
      tab === "all"
        ? Promise.all([fetchOpen(), fetchClosed()]).then(([o, c]) => {
            setOpenRows(Array.isArray(o) ? o : []);
            setClosedRows(Array.isArray(c) ? c : []);
          })
        : tab === "open"
        ? fetchOpen().then((o) => setOpenRows(Array.isArray(o) ? o : []))
        : fetchClosed().then((c) => setClosedRows(Array.isArray(c) ? c : []));
    job.catch((e) => setError(String(e))).finally(() => setLoading(false));
  }, [tab, refreshKey]);

  const rows = tab === "open" ? openRows : tab === "closed" ? closedRows : [...openRows, ...closedRows];
  const isOpenTab = tab === "open";

  const columns: Column<Position>[] = [
    {
      key: "symbol", label: "Symbol", sortable: true,
      render: (r) => <span className="font-mono text-[#F3F4F6]">{r.symbol}</span>,
    },
    {
      key: "side", label: "Side", sortable: true,
      render: (r) => <SideBadge side={r.side} />,
    },
    {
      key: "entry_price", label: "Entry", align: "right", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#F3F4F6]`}>{r.entry_price.toFixed(4)}</span>,
    },
    {
      key: "current_price", label: "Current", align: "right", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#F3F4F6]`}>{r.current_price.toFixed(4)}</span>,
    },
    {
      key: "sl_price", label: "SL", align: "right",
      render: (r) => <span className={`font-mono ${TABULAR} text-[#EF4444]/70`}>{r.sl_price.toFixed(4)}</span>,
    },
    {
      key: "tp_price", label: "TP", align: "right",
      render: (r) => <span className={`font-mono ${TABULAR} text-[#10B981]/70`}>{r.tp_price.toFixed(4)}</span>,
    },
    {
      key: "size", label: "Size", align: "right", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#9CA3AF]`}>{r.size.toFixed(4)}</span>,
    },
    {
      key: "pnl_usd", label: "PnL", align: "right", sortable: true,
      sortBy: (r) => r.pnl_usd,
      render: (r) => <PnlCell usd={r.pnl_usd} pct={r.pnl_pct} />,
    },
    ...(tab === "closed" || tab === "all"
      ? [{
          key: "closed_at" as keyof Position, label: "Closed", sortable: true,
          render: (r: Position) => <span className={`font-mono ${TABULAR} text-[10px] text-[#9CA3AF]`}>{fmtTime(r.closed_at)}</span>,
        }]
      : []),
    ...(tab === "closed" || tab === "all"
      ? [{
          key: "close_reason" as keyof Position, label: "Reason",
          render: (r: Position) => <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wide">{r.close_reason ?? "—"}</span>,
        }]
      : []),
    ...(isOpenTab
      ? [{
          key: "_actions" as keyof Position, label: "", align: "right" as const,
          render: (r: Position) => <CloseButton symbol={r.symbol} onClosed={refresh} />,
        }]
      : []),
  ];

  const totalPnl = openRows.reduce((acc, p) => acc + p.pnl_usd, 0);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Positions</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">
            {openRows.length} open · unrealized PnL{" "}
            <span className={`font-mono ${TABULAR} ${pnlColor(totalPnl)}`}>{fmtMoney(totalPnl)}</span>
          </p>
        </div>
        <button
          onClick={refresh}
          className="px-3 py-1.5 rounded border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] hover:border-[#4B5563] transition-colors"
        >
          ↻ Refresh
        </button>
      </div>

      {/* Tabs */}
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
            {t.key === "open" && openRows.length > 0 && (
              <span className="ml-1.5 px-1.5 py-0.5 text-[9px] font-mono bg-[#374151] text-[#9CA3AF] rounded">
                {openRows.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        {loading && (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        )}
        {error && (
          <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>
        )}
        {!loading && !error && rows.length === 0 && (
          <EmptyState
            icon={tab === "open" ? "○" : "≣"}
            title={tab === "open" ? "No open positions" : "No closed positions yet"}
            body={tab === "open" ? "Bot will open positions when signals exceed threshold." : "Trade history will appear here once positions close."}
          />
        )}
        {!loading && !error && rows.length > 0 && (
          <SortableTable
            rows={rows}
            columns={columns}
            rowKey={(r) => r.id}
            initialSort={isOpenTab ? { col: "pnl_usd", dir: "desc" } : { col: "closed_at", dir: "desc" }}
            compact
          />
        )}
      </div>
    </div>
  );
}
