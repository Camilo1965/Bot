"use client";

import { useEffect, useState } from "react";
import { useConnectionState } from "@/hooks/useConnectionState";
import { useWs } from "@/hooks/WsProvider";
import { StatusDot } from "@/components/ui/StatusDot";
import { fmtMoney, fmtPct, fmtTimeAgo, pnlColor, TABULAR } from "@/lib/format";

import { API_BASE as API } from "@/lib/api";

interface Props {
  onOpenPalette?: () => void;
  onOpenShortcuts?: () => void;
}

function fmtUptime(startedAt: string | null): string {
  if (!startedAt) return "";
  const t = Date.parse(startedAt);
  if (isNaN(t)) return "";
  const sec = Math.floor((Date.now() - t) / 1000);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
  return `${Math.floor(sec / 86400)}d ${Math.floor((sec % 86400) / 3600)}h`;
}

export function TopBar({ onOpenPalette, onOpenShortcuts }: Props) {
  const { state, lastOk, latencyMs } = useConnectionState();
  const ws = useWs();
  const [now, setNow] = useState<Date | null>(null);
  const [meta, setMeta] = useState<{ execution_mode?: string; session_started_at?: string; session_id?: string; balance?: number; daily_pnl_pct?: number } | null>(null);

  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    async function tick() {
      try {
        const r = await fetch(`${API}/api/state`);
        const j = await r.json();
        setMeta({
          execution_mode: j?.execution_mode,
          session_started_at: j?.session_started_at,
          session_id: j?.session_id,
          balance: j?.balance,
          daily_pnl_pct: j?.daily_pnl_pct,
        });
      } catch {}
    }
    tick();
    const id = setInterval(tick, 15000);
    return () => clearInterval(id);
  }, []);

  const utcStr = now ? now.toLocaleString("en-GB", { timeZone: "UTC", hour12: false }).replace(",", "") : "—";
  const stateLabel = state === "live" ? "LIVE" : state === "stale" ? "STALE" : state === "down" ? "OFFLINE" : "...";
  const mode = (meta?.execution_mode ?? "paper").toUpperCase();
  const modeColor = mode === "MT5" ? "text-[#EF4444]" : mode === "PAPER" ? "text-[#10B981]" : "text-[#F59E0B]";
  const uptime = fmtUptime(meta?.session_started_at ?? null);

  return (
    <header className="h-12 bg-[#10151D] border-b border-[#374151] flex items-center px-4 shrink-0 gap-4">
      {/* Clock + mode + uptime */}
      <div className="flex items-center gap-3 min-w-0">
        <span className={`text-xs font-mono ${TABULAR} text-[#F3F4F6] hidden sm:inline`}>{utcStr}</span>
        <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wider hidden md:inline">UTC</span>
        <span
          className={`inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wider ${modeColor}`}
          title={meta?.session_id ? `session ${meta.session_id}` : undefined}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-current" /> {mode}
        </span>
        {uptime && (
          <span className={`hidden lg:inline text-[10px] text-[#9CA3AF] font-mono ${TABULAR}`} title="Bot session uptime">
            ↑ {uptime}
          </span>
        )}
        {meta?.balance != null && (
          <span className={`hidden xl:inline text-[10px] font-mono ${TABULAR} text-[#9CA3AF]`} title="Account balance">
            {fmtMoney(meta.balance)}
          </span>
        )}
        {meta?.daily_pnl_pct != null && (
          <span
            className={`hidden xl:inline text-[10px] font-mono font-semibold ${TABULAR} ${pnlColor(meta.daily_pnl_pct)}`}
            title="Daily PnL %"
          >
            {meta.daily_pnl_pct >= 0 ? "+" : ""}{fmtPct(meta.daily_pnl_pct)}
          </span>
        )}
      </div>

      {/* Command palette trigger */}
      <button
        onClick={onOpenPalette}
        className="hidden sm:flex items-center gap-2 ml-auto h-7 px-2.5 rounded-md border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] hover:border-[#4B5563] transition-colors min-w-[160px]"
        aria-label="Open command palette"
      >
        <span className="opacity-70">⌕</span>
        <span className="flex-1 text-left">Search…</span>
        <kbd className="text-[10px] font-mono bg-[#1E2530] border border-[#374151] rounded px-1">⌘K</kbd>
      </button>

      {/* WS state */}
      <div
        className="hidden md:flex items-center gap-1.5 text-[10px] text-[#9CA3AF] font-mono border border-[#374151] rounded px-2 py-1"
        title={`WebSocket: ${ws.state}`}
      >
        <StatusDot state={ws.state === "open" ? "live" : ws.state === "connecting" ? "warn" : "down"} pulse />
        <span className="uppercase tracking-wider">WS</span>
      </div>

      {/* Connection + latency */}
      <div
        className="hidden md:flex items-center gap-2 text-[10px] text-[#9CA3AF] font-mono"
        title={lastOk ? `Last OK: ${new Date(lastOk).toLocaleTimeString()}` : "Never connected"}
      >
        <StatusDot state={state === "connecting" ? "warn" : state} pulse />
        <span className="uppercase tracking-wider">{stateLabel}</span>
        {latencyMs !== null && (
          <span className={TABULAR}>{latencyMs}ms</span>
        )}
        {lastOk && state !== "live" && (
          <span>· {fmtTimeAgo(new Date(lastOk).toISOString())}</span>
        )}
      </div>

      {/* Help */}
      <button
        onClick={onOpenShortcuts}
        className="hidden lg:flex items-center justify-center w-7 h-7 rounded border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] hover:border-[#4B5563] transition-colors"
        title="Keyboard shortcuts (?)"
        aria-label="Keyboard shortcuts"
      >
        ?
      </button>

      {/* Open palette on mobile compact */}
      <button
        onClick={onOpenPalette}
        className="sm:hidden ml-auto w-7 h-7 rounded border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] flex items-center justify-center"
        aria-label="Command palette"
      >
        ⌕
      </button>
    </header>
  );
}
