"use client";

import { useEffect, useState } from "react";
import { useConnectionState } from "@/hooks/useConnectionState";
import { StatusDot } from "@/components/ui/StatusDot";
import { fmtTimeAgo, TABULAR } from "@/lib/format";

interface Props {
  onOpenPalette?: () => void;
  onOpenShortcuts?: () => void;
  mode?: string;
}

export function TopBar({ onOpenPalette, onOpenShortcuts, mode = "PAPER" }: Props) {
  const { state, lastOk, latencyMs } = useConnectionState();
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const utcStr = now.toLocaleString("en-GB", { timeZone: "UTC", hour12: false }).replace(",", "");
  const stateLabel = state === "live" ? "LIVE" : state === "stale" ? "STALE" : state === "down" ? "OFFLINE" : "...";
  const modeColor = mode === "LIVE" ? "text-[#EF4444]" : mode === "PAPER" ? "text-[#10B981]" : "text-[#F59E0B]";

  return (
    <header className="h-12 bg-[#10151D] border-b border-[#374151] flex items-center px-4 shrink-0 gap-4">
      {/* Clock + mode */}
      <div className="flex items-center gap-3 min-w-0">
        <span className={`text-xs font-mono ${TABULAR} text-[#F3F4F6] hidden sm:inline`}>{utcStr}</span>
        <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wider hidden md:inline">UTC</span>
        <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold tracking-wider ${modeColor}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-current" /> {mode}
        </span>
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
