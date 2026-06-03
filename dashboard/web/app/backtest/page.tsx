"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type BacktestMode = "disk" | "inline";

interface ReportFile {
  filename: string;
  created_at: string;
  size_bytes?: number;
  path?: string;
}

export default function BacktestPage() {
  const [symbols, setSymbols] = useState("BTC/USDT");
  const [days, setDays] = useState(30);
  const [mode, setMode] = useState<BacktestMode>("disk");
  const [reports, setReports] = useState<ReportFile[]>([]);
  const [reportsLoading, setReportsLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/backtest/list`)
      .then((r) => r.json())
      .then((data) => {
        setReports(Array.isArray(data) ? data : []);
        setReportsLoading(false);
      })
      .catch(() => setReportsLoading(false));
  }, []);

  const scriptName =
    mode === "disk" ? "backtest_disk_loaded.py" : "backtest_full_bot.py";

  const command = `python scripts/${scriptName} --symbols "${symbols}" --report --days ${days}`;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Backtest</h1>

      {/* Form */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-5 space-y-4 max-w-xl">
        <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-2">
          Configure Run
        </div>

        <div className="space-y-1">
          <label className="text-xs text-[#9CA3AF]" htmlFor="bt-symbols">
            Symbols (comma-separated)
          </label>
          <input
            id="bt-symbols"
            type="text"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            className="w-full bg-[#10151D] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
            placeholder="BTC/USDT, ETH/USDT"
          />
        </div>

        <div className="space-y-1">
          <label className="text-xs text-[#9CA3AF]" htmlFor="bt-days">
            Days
          </label>
          <input
            id="bt-days"
            type="number"
            min={1}
            max={365}
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="w-full bg-[#10151D] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
          />
        </div>

        <div className="space-y-1">
          <label className="text-xs text-[#9CA3AF]" htmlFor="bt-mode">
            Mode
          </label>
          <select
            id="bt-mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as BacktestMode)}
            className="w-full bg-[#10151D] border border-[#374151] rounded px-3 py-2 text-sm text-[#F3F4F6] focus:outline-none focus:border-[#3B82F6]"
          >
            <option value="disk">disk — load pre-saved models from disk</option>
            <option value="inline">inline — full pipeline backtest</option>
          </select>
        </div>

        {/* Command preview */}
        <div className="space-y-1">
          <div className="text-xs text-[#9CA3AF]">Command to run</div>
          <pre className="bg-[#0a0e15] border border-[#374151] rounded px-3 py-3 text-xs font-mono text-[#10B981] overflow-x-auto whitespace-pre-wrap break-all">
            {command}
          </pre>
        </div>

        <div className="relative group">
          <button
            disabled
            className="w-full px-4 py-2 rounded bg-[#374151] text-[#9CA3AF] text-sm cursor-not-allowed"
          >
            Run Backtest (manual only)
          </button>
          <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-1 hidden group-hover:block bg-[#374151] text-[#9CA3AF] text-xs rounded px-2 py-1 whitespace-nowrap z-10">
            Copy the command above and run it in your terminal
          </div>
        </div>
      </div>

      {/* Recent reports */}
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
        <div className="text-xs uppercase tracking-widest text-[#9CA3AF] mb-3">
          Recent Report Files
        </div>

        {reportsLoading && (
          <div className="text-[#9CA3AF] text-sm text-center py-6">Checking for reports…</div>
        )}

        {!reportsLoading && reports.length === 0 && (
          <div className="text-[#9CA3AF] text-sm text-center py-6">
            No report files found. Run a backtest to generate reports.
          </div>
        )}

        {!reportsLoading && reports.length > 0 && (
          <ul className="space-y-2">
            {reports.map((r) => (
              <li
                key={r.filename}
                className="flex items-center justify-between text-sm border-b border-[#374151]/40 pb-2"
              >
                <span className="font-mono text-[#F3F4F6] text-xs truncate">{r.filename}</span>
                <div className="flex items-center gap-4 shrink-0 ml-4">
                  {r.size_bytes !== undefined && (
                    <span className="text-[#9CA3AF] text-xs">
                      {(r.size_bytes / 1024).toFixed(1)} KB
                    </span>
                  )}
                  <span className="text-[#9CA3AF] text-xs">
                    {new Date(r.created_at).toLocaleDateString()}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
