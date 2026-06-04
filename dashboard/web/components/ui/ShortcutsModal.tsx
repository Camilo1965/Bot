"use client";

import { useEffect } from "react";
import { SHORTCUT_HELP } from "@/hooks/useKeyboardShortcuts";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function ShortcutsModal({ open, onClose }: Props) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    if (open) window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-[#10151D] border border-[#374151] rounded-xl shadow-2xl p-6 w-full max-w-md mx-4"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm uppercase tracking-widest text-[#F3F4F6] font-semibold">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="text-[#9CA3AF] hover:text-[#F3F4F6] text-sm"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="space-y-2">
          {SHORTCUT_HELP.map((s) => (
            <div key={s.keys} className="flex items-center justify-between border-b border-[#374151]/40 pb-1.5">
              <span className="text-sm text-[#F3F4F6]">{s.label}</span>
              <div className="flex gap-1">
                {s.keys.split(" ").map((k, i) => (
                  <kbd
                    key={i}
                    className="px-2 py-0.5 text-[11px] font-mono bg-[#1E2530] border border-[#374151] rounded text-[#9CA3AF]"
                  >
                    {k}
                  </kbd>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-3 border-t border-[#374151] text-[11px] text-[#9CA3AF]">
          Press <kbd className="px-1.5 py-0.5 bg-[#1E2530] border border-[#374151] rounded font-mono">Esc</kbd> to close.
        </div>
      </div>
    </div>
  );
}
