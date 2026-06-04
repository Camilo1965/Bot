"use client";

import { useEffect, useMemo, useState } from "react";
import { SortableTable, Column } from "@/components/ui/SortableTable";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton, SkeletonTable } from "@/components/ui/Skeleton";
import { fmtMoney, fmtPct, fmtDateTime, pnlColor, TABULAR } from "@/lib/format";
import { useToast } from "@/components/ui/Toast";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Tab = "trades" | "vibe" | "notes";

interface JournalTrade {
  id: string;
  timestamp_close: string;
  symbol: string;
  side: "long" | "short";
  pnl_usd: number;
  pnl_pct: number;
  close_reason: string;
  ml_confidence: number;
}

const TABS: { key: Tab; label: string }[] = [
  { key: "trades", label: "Trades" },
  { key: "vibe", label: "VIBE Insights" },
  { key: "notes", label: "Notes" },
];

function TradesTab() {
  const [trades, setTrades] = useState<JournalTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [symbolFilter, setSymbolFilter] = useState<string>("all");
  const [sideFilter, setSideFilter] = useState<"all" | "long" | "short">("all");
  const [winFilter, setWinFilter] = useState<"all" | "win" | "loss">("all");
  const toast = useToast();

  useEffect(() => {
    fetch(`${API}/api/journal/trades?limit=500`)
      .then((r) => r.json())
      .then((data) => {
        setTrades(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  const symbols = useMemo(() => {
    const s = new Set(trades.map((t) => t.symbol));
    return ["all", ...Array.from(s).sort()];
  }, [trades]);

  const filtered = useMemo(() => {
    return trades.filter((t) => {
      if (symbolFilter !== "all" && t.symbol !== symbolFilter) return false;
      if (sideFilter !== "all" && t.side !== sideFilter) return false;
      if (winFilter === "win" && !(t.pnl_usd > 0)) return false;
      if (winFilter === "loss" && !(t.pnl_usd < 0)) return false;
      return true;
    });
  }, [trades, symbolFilter, sideFilter, winFilter]);

  const stats = useMemo(() => {
    const n = filtered.length;
    const wins = filtered.filter((t) => t.pnl_usd > 0).length;
    const losses = filtered.filter((t) => t.pnl_usd < 0).length;
    const totalPnl = filtered.reduce((acc, t) => acc + t.pnl_usd, 0);
    const wr = n > 0 ? (wins / n) * 100 : 0;
    return { n, wins, losses, totalPnl, wr };
  }, [filtered]);

  async function exportCsv() {
    try {
      const res = await fetch(`${API}/api/journal/trades/csv?limit=5000`);
      const text = await res.text();
      const blob = new Blob([text], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `clawdbot-journal-${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.push({ type: "success", title: "Journal exported." });
    } catch {
      toast.push({ type: "error", title: "Export failed." });
    }
  }

  if (loading) return <SkeletonTable rows={6} cols={6} />;
  if (error) return <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>;

  const columns: Column<JournalTrade>[] = [
    {
      key: "timestamp_close", label: "Closed At", sortable: true,
      render: (t) => <span className={`font-mono ${TABULAR} text-[#9CA3AF] text-xs whitespace-nowrap`}>{fmtDateTime(t.timestamp_close)}</span>,
    },
    {
      key: "symbol", label: "Symbol", sortable: true,
      render: (t) => <span className="font-mono text-[#F3F4F6]">{t.symbol}</span>,
    },
    {
      key: "side", label: "Side", sortable: true,
      render: (t) => (
        <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${t.side === "long" ? "bg-[#10B981]/15 text-[#10B981]" : "bg-[#EF4444]/15 text-[#EF4444]"}`}>
          {t.side.toUpperCase()}
        </span>
      ),
    },
    {
      key: "pnl_usd", label: "PnL", align: "right", sortable: true, sortBy: (t) => t.pnl_usd,
      render: (t) => (
        <div className="flex flex-col items-end leading-tight">
          <span className={`font-mono ${TABULAR} ${pnlColor(t.pnl_usd)}`}>{fmtMoney(t.pnl_usd)}</span>
          <span className={`font-mono ${TABULAR} text-[10px] ${pnlColor(t.pnl_pct)} opacity-80`}>{fmtPct(t.pnl_pct)}</span>
        </div>
      ),
    },
    {
      key: "close_reason", label: "Reason",
      render: (t) => <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wide">{t.close_reason || "—"}</span>,
    },
    {
      key: "ml_confidence", label: "ML Conf", align: "right", sortable: true,
      render: (t) => <span className={`font-mono ${TABULAR} text-[#F3F4F6]`}>{(t.ml_confidence * 100).toFixed(1)}%</span>,
    },
  ];

  return (
    <div className="space-y-4">
      {/* Filter strip + stats */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <select
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            className="bg-[#0a0e15] border border-[#374151] rounded px-2 py-1 text-xs text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
          >
            {symbols.map((s) => <option key={s} value={s}>{s === "all" ? "All symbols" : s}</option>)}
          </select>
          <select
            value={sideFilter}
            onChange={(e) => setSideFilter(e.target.value as any)}
            className="bg-[#0a0e15] border border-[#374151] rounded px-2 py-1 text-xs text-[#F3F4F6] focus:outline-none focus:border-[#3B82F6]"
          >
            <option value="all">Both sides</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
          <select
            value={winFilter}
            onChange={(e) => setWinFilter(e.target.value as any)}
            className="bg-[#0a0e15] border border-[#374151] rounded px-2 py-1 text-xs text-[#F3F4F6] focus:outline-none focus:border-[#3B82F6]"
          >
            <option value="all">Wins & losses</option>
            <option value="win">Wins only</option>
            <option value="loss">Losses only</option>
          </select>
        </div>
        <button
          onClick={exportCsv}
          className="px-3 py-1 rounded border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] hover:border-[#4B5563] transition-colors"
        >
          ↓ Export CSV
        </button>
      </div>

      {/* Stat row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="Trades" value={String(stats.n)} />
        <Stat label="Win Rate" value={`${stats.wr.toFixed(1)}%`} valueClass={stats.wr >= 50 ? "text-[#10B981]" : "text-[#EF4444]"} />
        <Stat label="Wins · Losses" value={`${stats.wins} · ${stats.losses}`} />
        <Stat label="Net PnL" value={fmtMoney(stats.totalPnl)} valueClass={pnlColor(stats.totalPnl)} />
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon="≣" title="No trades match filters" body="Adjust the filters above or wait for new trades to close." />
      ) : (
        <SortableTable
          rows={filtered}
          columns={columns}
          rowKey={(t, i) => t.id || String(i)}
          initialSort={{ col: "timestamp_close", dir: "desc" }}
          compact
        />
      )}
    </div>
  );
}

function Stat({ label, value, valueClass = "text-[#F3F4F6]" }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="bg-[#0a0e15] border border-[#374151] rounded p-2.5">
      <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-1">{label}</div>
      <div className={`font-mono ${TABULAR} text-sm ${valueClass}`}>{value}</div>
    </div>
  );
}

function VibeTab() {
  return (
    <EmptyState
      icon="∿"
      title="VIBE Insights not generated"
      body="Run python vibe/journal_analyzer.py to generate behavioral analysis. Results will appear here automatically."
    />
  );
}

function NotesTab() {
  const [notes, setNotes] = useState("");

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("clawdbot:journal:notes") : null;
    if (saved) setNotes(saved);
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const t = setTimeout(() => localStorage.setItem("clawdbot:journal:notes", notes), 300);
      return () => clearTimeout(t);
    }
  }, [notes]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#9CA3AF]">Local scratch pad — autosaved to browser.</span>
        <button
          onClick={() => setNotes("")}
          className="text-[10px] text-[#9CA3AF] hover:text-[#F3F4F6]"
        >
          Clear
        </button>
      </div>
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Write your trading notes here…&#10;Saved automatically."
        rows={14}
        className="w-full bg-[#0a0e15] border border-[#374151] rounded-lg px-4 py-3 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6] resize-y"
      />
      <div className="text-[10px] text-[#9CA3AF] font-mono">
        {notes.length} chars · {notes.split(/\s+/).filter(Boolean).length} words
      </div>
    </div>
  );
}

export default function JournalPage() {
  const [tab, setTab] = useState<Tab>("trades");

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Journal</h1>
        <p className="text-xs text-[#9CA3AF] mt-0.5">Closed trades, VIBE behavioral insights, and notes.</p>
      </div>

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

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        {tab === "trades" && <TradesTab />}
        {tab === "vibe" && <VibeTab />}
        {tab === "notes" && <NotesTab />}
      </div>
    </div>
  );
}
