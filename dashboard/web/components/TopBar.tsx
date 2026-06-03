"use client";

export function TopBar() {
  return (
    <header className="h-12 bg-bg-surface border-b border-border-default flex items-center px-6 shrink-0">
      <span className="text-xs text-text-secondary font-mono">
        {new Date().toLocaleString("en-US", { timeZone: "UTC", hour12: false })} UTC
      </span>
      <div className="ml-auto flex items-center gap-3">
        <span className="inline-flex items-center gap-1.5 text-xs text-pnl-positive">
          <span className="w-2 h-2 rounded-full bg-pnl-positive animate-pulse" />
          PAPER MODE
        </span>
      </div>
    </header>
  );
}
