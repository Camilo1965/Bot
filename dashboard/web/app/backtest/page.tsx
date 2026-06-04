"use client";

import { useEffect, useRef, useState } from "react";
import { SortableTable, Column } from "@/components/ui/SortableTable";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { fmtAbbrev, fmtDateTime, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Mode = "disk" | "inline";
type RunState = "idle" | "running" | "done" | "error";

interface ReportFile {
  filename: string;
  created_at: string;
  size_bytes?: number;
  path?: string;
}

export default function BacktestPage() {
  const [symbols, setSymbols] = useState("BTC/USDT");
  const [days, setDays] = useState(30);
  const [mode, setMode] = useState<Mode>("disk");
  const [reports, setReports] = useState<ReportFile[]>([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [runState, setRunState] = useState<RunState>("idle");
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const toast = useToast();

  function loadReports() {
    setReportsLoading(true);
    fetch(`${API}/api/backtest/list`)
      .then((r) => r.json())
      .then((data) => { setReports(Array.isArray(data) ? data : []); setReportsLoading(false); })
      .catch(() => setReportsLoading(false));
  }

  useEffect(() => { loadReports(); }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  async function runBacktest() {
    setLogs([]);
    setRunState("running");
    try {
      const res = await fetch(`${API}/api/backtest/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols, days, mode }),
      });
      if (!res.ok || !res.body) { setRunState("error"); toast.push({ type: "error", title: "Backtest failed to start" }); return; }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.replace(/^data:\s*/, "").trim();
          if (!line) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.type === "log") setLogs((l) => [...l, ev.line]);
            if (ev.type === "done") {
              setRunState(ev.exit_code === 0 ? "done" : "error");
              toast.push({
                type: ev.exit_code === 0 ? "success" : "warn",
                title: `Backtest ${ev.exit_code === 0 ? "completed" : "exited with code " + ev.exit_code}`,
              });
              loadReports();
            }
            if (ev.type === "error") {
              setLogs((l) => [...l, `ERROR: ${ev.msg}`]);
              setRunState("error");
              toast.push({ type: "error", title: "Backtest error", body: ev.msg });
            }
          } catch {}
        }
      }
    } catch (e) {
      setLogs((l) => [...l, String(e)]);
      setRunState("error");
    }
  }

  const btnLabel = runState === "running" ? "Running…" : runState === "done" ? "✓ Done" : runState === "error" ? "✕ Error" : "▷ Run Backtest";
  const btnCls =
    runState === "running" ? "bg-[#F59E0B]/15 text-[#F59E0B] cursor-wait"
    : runState === "done" ? "bg-[#10B981]/15 text-[#10B981]"
    : runState === "error" ? "bg-[#EF4444]/15 text-[#EF4444]"
    : "bg-[#3B82F6]/15 text-[#3B82F6] hover:bg-[#3B82F6]/25 cursor-pointer";

  const reportCols: Column<ReportFile>[] = [
    {
      key: "filename", label: "File", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#F3F4F6] text-xs truncate block`}>{r.filename}</span>,
    },
    {
      key: "size_bytes", label: "Size", align: "right", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#9CA3AF] text-xs`}>{r.size_bytes ? fmtAbbrev(r.size_bytes / 1024, 1) + "K" : "—"}</span>,
    },
    {
      key: "created_at", label: "Created", sortable: true,
      render: (r) => <span className={`font-mono ${TABULAR} text-[#9CA3AF] text-xs`}>{fmtDateTime(r.created_at)}</span>,
    },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Backtest</h1>
        <p className="text-xs text-[#9CA3AF] mt-0.5">Run portfolio simulation with current models + threshold sweep.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Form panel */}
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-5 space-y-4">
          <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Configure Run</div>

          <div className="space-y-1">
            <label className="text-xs text-[#9CA3AF]" htmlFor="bt-symbols">Symbols (comma-separated)</label>
            <input id="bt-symbols" type="text" value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              className="w-full bg-[#0a0e15] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
              placeholder="BTC/USDT, ETH/USDT" />
          </div>

          <div className="space-y-1">
            <label className="text-xs text-[#9CA3AF]" htmlFor="bt-days">Days</label>
            <div className="flex gap-1">
              {[7, 30, 60, 180].map((d) => (
                <button key={d} onClick={() => setDays(d)}
                  className={`flex-1 py-1.5 rounded text-xs transition-colors ${days === d ? "bg-[#3B82F6]/20 text-[#3B82F6] border border-[#3B82F6]/30" : "bg-[#0a0e15] border border-[#374151] text-[#9CA3AF] hover:text-[#F3F4F6]"}`}>
                  {d}d
                </button>
              ))}
            </div>
            <input id="bt-days" type="number" min={1} max={365} value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className={`w-full bg-[#0a0e15] border border-[#374151] rounded px-3 py-1.5 text-sm text-[#F3F4F6] font-mono ${TABULAR} focus:outline-none focus:border-[#3B82F6]`} />
          </div>

          <div className="space-y-1">
            <label className="text-xs text-[#9CA3AF]" htmlFor="bt-mode">Mode</label>
            <select id="bt-mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}
              className="w-full bg-[#0a0e15] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] focus:outline-none focus:border-[#3B82F6]">
              <option value="disk">disk · load pre-saved models</option>
              <option value="inline">inline · full pipeline backtest</option>
            </select>
          </div>

          <button onClick={runBacktest} disabled={runState === "running"}
            className={`w-full px-4 py-2.5 rounded text-sm font-medium transition-colors ${btnCls}`}>
            {btnLabel}
          </button>
        </div>

        {/* Live output panel */}
        <div className="bg-[#0a0e15] border border-[#374151] rounded-lg p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Live Output</div>
            {logs.length > 0 && (
              <span className="text-[10px] text-[#6B7280] font-mono">{logs.length} lines</span>
            )}
          </div>
          {logs.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-[#6B7280] text-xs">
              {runState === "running" ? "Connecting…" : "Output will stream here when you run a backtest."}
            </div>
          ) : (
            <div className="h-64 overflow-y-auto font-mono text-[10px] text-[#10B981] space-y-0.5">
              {logs.map((line, i) => (
                <div key={i} className="leading-4 whitespace-pre-wrap break-all">{line}</div>
              ))}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Reports */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">Recent Reports</div>
        {reportsLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-8" />)}
          </div>
        ) : reports.length === 0 ? (
          <EmptyState icon="▷" title="No reports yet" body="Run a backtest above to generate one." />
        ) : (
          <SortableTable
            rows={reports}
            columns={reportCols}
            rowKey={(r) => r.filename}
            initialSort={{ col: "created_at", dir: "desc" }}
            compact
          />
        )}
      </div>
    </div>
  );
}
