"use client";

import { useEffect, useState } from "react";

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

function PnlCell({ usd, pct }: { usd: number; pct: number }) {
  const color = usd >= 0 ? "text-[#10B981]" : "text-[#EF4444]";
  return (
    <span className={`font-mono ${color}`}>
      {usd >= 0 ? "+" : ""}
      {usd.toFixed(2)}
      <span className="text-[10px] ml-1 opacity-70">
        ({pct >= 0 ? "+" : ""}
        {pct.toFixed(2)}%)
      </span>
    </span>
  );
}

function TradesTab() {
  const [trades, setTrades] = useState<JournalTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/api/journal/trades`)
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

  if (loading) return <div className="text-[#9CA3AF] text-sm text-center py-8">Loading trades…</div>;
  if (error) return <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>;
  if (trades.length === 0) return <div className="text-[#9CA3AF] text-sm text-center py-8">No trades recorded yet.</div>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-widest text-[#9CA3AF] border-b border-[#374151]">
            {["Closed At", "Symbol", "Side", "PnL", "Close Reason", "ML Conf"].map((h) => (
              <th key={h} className="pb-2 pr-4 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr
              key={t.id}
              className="border-b border-[#374151]/40 hover:bg-white/5 transition-colors"
            >
              <td className="py-2 pr-4 font-mono text-[#9CA3AF] text-xs whitespace-nowrap">
                {new Date(t.timestamp_close).toLocaleString()}
              </td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{t.symbol}</td>
              <td className="py-2 pr-4">
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                    t.side === "long"
                      ? "bg-[#10B981]/20 text-[#10B981]"
                      : "bg-[#EF4444]/20 text-[#EF4444]"
                  }`}
                >
                  {t.side.toUpperCase()}
                </span>
              </td>
              <td className="py-2 pr-4">
                <PnlCell usd={t.pnl_usd} pct={t.pnl_pct} />
              </td>
              <td className="py-2 pr-4 text-[#9CA3AF] text-xs">{t.close_reason}</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">
                {(t.ml_confidence * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VibeTab() {
  return (
    <div className="space-y-4">
      <div className="bg-[#374151]/30 border border-[#374151] rounded-lg p-5 text-sm text-[#9CA3AF]">
        <div className="font-semibold text-[#F3F4F6] mb-2">VIBE Insights not generated yet</div>
        <p>Run the VIBE journal analyzer to generate behavioral insights from your trade history:</p>
        <pre className="mt-3 bg-[#0a0e15] border border-[#374151] rounded px-3 py-2 text-xs font-mono text-[#10B981] overflow-x-auto">
          python vibe/journal_analyzer.py
        </pre>
        <p className="mt-3 text-xs text-[#9CA3AF]">
          Results will be stored in the database and displayed here automatically on next page load.
        </p>
      </div>
    </div>
  );
}

function NotesTab() {
  const [notes, setNotes] = useState("");

  return (
    <div className="space-y-3">
      <div className="text-xs text-[#9CA3AF]">
        Local scratch pad — not persisted to the server yet.
      </div>
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Write your trading notes here…"
        rows={14}
        className="w-full bg-[#10151D] border border-[#374151] rounded-lg px-4 py-3 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6] resize-y"
      />
      <div className="text-[10px] text-[#9CA3AF]">
        {notes.length} characters — persistence coming in a future release.
      </div>
    </div>
  );
}

export default function JournalPage() {
  const [tab, setTab] = useState<Tab>("trades");

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Journal</h1>

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

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        {tab === "trades" && <TradesTab />}
        {tab === "vibe" && <VibeTab />}
        {tab === "notes" && <NotesTab />}
      </div>
    </div>
  );
}
