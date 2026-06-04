"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

interface Command {
  id: string;
  label: string;
  hint?: string;
  group: "Navigate" | "Action";
  onRun: () => void;
}

interface Props {
  open: boolean;
  onClose: () => void;
}

export function CommandPalette({ open, onClose }: Props) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);

  const cmds = useMemo<Command[]>(() => [
    { id: "go-overview",    label: "Go to Overview",    hint: "g o", group: "Navigate", onRun: () => router.push("/") },
    { id: "go-positions",   label: "Go to Positions",   hint: "g p", group: "Navigate", onRun: () => router.push("/positions") },
    { id: "go-signals",     label: "Go to Signals",     hint: "g s", group: "Navigate", onRun: () => router.push("/signals") },
    { id: "go-performance", label: "Go to Performance", hint: "g f", group: "Navigate", onRun: () => router.push("/performance") },
    { id: "go-risk",        label: "Go to Risk",        hint: "g r", group: "Navigate", onRun: () => router.push("/risk") },
    { id: "go-models",      label: "Go to Models",      hint: "g m", group: "Navigate", onRun: () => router.push("/models") },
    { id: "go-backtest",    label: "Go to Backtest",    hint: "g b", group: "Navigate", onRun: () => router.push("/backtest") },
    { id: "go-journal",     label: "Go to Journal",     hint: "g j", group: "Navigate", onRun: () => router.push("/journal") },
    { id: "go-settings",    label: "Go to Settings",    hint: "g ,", group: "Navigate", onRun: () => router.push("/settings") },
    { id: "go-alerts",      label: "Go to Alerts",      hint: "g a", group: "Navigate", onRun: () => router.push("/alerts") },
  ], [router]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return cmds;
    return cmds.filter((c) => c.label.toLowerCase().includes(q) || c.id.includes(q));
  }, [cmds, query]);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setActiveIdx(0);
    }
  }, [open]);

  useEffect(() => {
    setActiveIdx(0);
  }, [query]);

  useEffect(() => {
    if (!open) return;
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIdx((i) => Math.min(i + 1, filtered.length - 1));
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIdx((i) => Math.max(i - 1, 0));
        return;
      }
      if (e.key === "Enter") {
        e.preventDefault();
        const cmd = filtered[activeIdx];
        if (cmd) {
          cmd.onRun();
          onClose();
        }
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, filtered, activeIdx, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 backdrop-blur-sm pt-[20vh]"
      onClick={onClose}
    >
      <div
        className="bg-[#10151D] border border-[#374151] rounded-xl shadow-2xl w-full max-w-xl mx-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-[#374151] px-3 py-2">
          <input
            type="text"
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search commands…"
            className="w-full bg-transparent text-[#F3F4F6] placeholder:text-[#6B7280] text-sm focus:outline-none font-mono"
          />
        </div>
        <div className="max-h-96 overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="px-3 py-6 text-center text-[#9CA3AF] text-sm">No matches.</div>
          ) : (
            filtered.map((c, i) => (
              <button
                key={c.id}
                onClick={() => { c.onRun(); onClose(); }}
                onMouseEnter={() => setActiveIdx(i)}
                className={`w-full text-left flex items-center justify-between px-3 py-2 transition-colors ${
                  i === activeIdx ? "bg-[#3B82F6]/20 text-[#F3F4F6]" : "text-[#F3F4F6] hover:bg-white/5"
                }`}
              >
                <span className="text-sm flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-[#9CA3AF]">{c.group}</span>
                  <span>{c.label}</span>
                </span>
                {c.hint && (
                  <span className="flex gap-1">
                    {c.hint.split(" ").map((k, j) => (
                      <kbd key={j} className="px-1.5 py-0.5 text-[10px] font-mono bg-[#1E2530] border border-[#374151] rounded text-[#9CA3AF]">{k}</kbd>
                    ))}
                  </span>
                )}
              </button>
            ))
          )}
        </div>
        <div className="border-t border-[#374151] px-3 py-1.5 flex items-center justify-between text-[10px] text-[#9CA3AF]">
          <div className="flex items-center gap-3">
            <span><kbd className="px-1 bg-[#1E2530] border border-[#374151] rounded font-mono">↑↓</kbd> navigate</span>
            <span><kbd className="px-1 bg-[#1E2530] border border-[#374151] rounded font-mono">↵</kbd> run</span>
            <span><kbd className="px-1 bg-[#1E2530] border border-[#374151] rounded font-mono">esc</kbd> close</span>
          </div>
          <span>{filtered.length} match{filtered.length === 1 ? "" : "es"}</span>
        </div>
      </div>
    </div>
  );
}
