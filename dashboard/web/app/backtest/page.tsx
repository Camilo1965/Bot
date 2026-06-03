"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type BacktestMode = "disk" | "inline";
type RunState = "idle" | "running" | "done" | "error";

interface ReportFile {
  filename: string;
  created_at: string;
  size_bytes?: number;
}

export default function BacktestPage() {
  const [symbols, setSymbols] = useState("BTC/USDT");
  const [days, setDays] = useState(30);
  const [mode, setMode] = useState<BacktestMode>("disk");
  const [reports, setReports] = useState<ReportFile[]>([]);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [runState, setRunState] = useState<RunState>("idle");
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API}/api/backtest/list`)
      .then((r) => r.json())
      .then((data) => { setReports(Array.isArray(data) ? data : []); setReportsLoading(false); })
      .catch(() => setReportsLoading(false));
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  function runBacktest() {
    setLogs([]);
    setRunState("running");
    const es = new EventSource(
      `${API}/api/backtest/run?_t=${Date.now()}`,
    );

    // POST via fetch + read SSE manually (EventSource only supports GET)
    es.close();

    fetch(`${API}/api/backtest/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbols, days, mode }),
    }).then(async (res) => {
      if (!res.ok || !res.body) { setRunState("error"); return; }
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
            if (ev.type === "done") setRunState(ev.exit_code === 0 ? "done" : "error");
            if (ev.type === "error") { setLogs((l) => [...l, `ERROR: ${ev.msg}`]); setRunState("error"); }
          } catch {}
        }
      }
    }).catch((e) => { setLogs((l) => [...l, String(e)]); setRunState("error"); });
  }

  const btnLabel = runState === "running" ? "Running…" : runState === "done" ? "Done" : "Run Backtest";
  const btnClass =
    runState === "running"
      ? "bg-[#F59E0B]/20 text-[#F59E0B] cursor-wait"
      : runState === "done"
      ? "bg-[#10B981]/20 text-[#10B981]"
      : runState === "error"
      ? "bg-[#EF4444]/20 text-[#EF4444]"
      : "bg-[#3B82F6]/20 text-[#3B82F6] hover:bg-[#3B82F6]/30 cursor-pointer";

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Backtest</h1>

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-5 space-y-4 max-w-xl">
        <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-2">Configure Run</div>

        <div className="space-y-1">
          <label className="text-xs text-[#9CA3AF]" htmlFor="bt-symbols">Symbols (comma-separated)</label>
          <input id="bt-symbols" type="text" value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            className="w-full bg-[#10151D] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
            placeholder="BTC/USDT, ETH/USDT" />
        </div>

        <div className="space-y-1">
          <label className="text-xs text-[#9CA3AF]" htmlFor="bt-days">Days</label>
          <input id="bt-days" type="number" min={1} max={365} value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="w-full bg-[#10151D] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]" />
        </div>

        <div className="space-y-1">
          <label className="text-xs text-[#9CA3AF]" htmlFor="bt-mode">Mode</label>
          <select id="bt-mode" value={mode} onChange={(e) => setMode(e.target.value as BacktestMode)}
            className="w-full bg-[#10151D] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] focus:outline-none focus:border-[#3B82F6]">
            <option value="disk">disk — load pre-saved models from disk</option>
            <option value="inline">inline — full pipeline backtest</option>
          </select>
        </div>

        <button onClick={runBacktest} disabled={runState === "running"}
          className={`w-full px-4 py-2 rounded text-sm font-medium transition-colors ${btnClass}`}>
          {btnLabel}
        </button>
      </div>

      {logs.length > 0 && (
        <div className="bg-[#0a0e15] border border-[#374151] rounded-lg p-4">
          <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-2">Live Output</div>
          <div className="h-64 overflow-y-auto font-mono text-xs text-[#10B981] space-y-0.5">
            {logs.map((line, i) => (
              <div key={i} className="leading-5 whitespace-pre-wrap break-all">{line}</div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">Recent Reports</div>
        {reportsLoading && <div className="text-[#9CA3AF] text-sm text-center py-6">Loading…</div>}
        {!reportsLoading && reports.length === 0 && (
          <div className="text-[#9CA3AF] text-sm text-center py-6">No reports found.</div>
        )}
        {!reportsLoading && reports.length > 0 && (
          <ul className="space-y-2">
            {reports.map((r) => (
              <li key={r.filename} className="flex items-center justify-between text-sm border-b border-[#374151]/40 pb-2">
                <span className="font-mono text-[#F3F4F6] text-xs truncate">{r.filename}</span>
                <div className="flex items-center gap-4 shrink-0 ml-4">
                  {r.size_bytes !== undefined && (
                    <span className="text-[#9CA3AF] text-xs">{(r.size_bytes / 1024).toFixed(1)} KB</span>
                  )}
                  <span className="text-[#9CA3AF] text-xs">{new Date(r.created_at).toLocaleDateString()}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
