"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/**
 * Vim-style 2-key sequences: `g + letter`.
 *   g o → Overview      g p → Positions     g s → Signals
 *   g f → Performance   g r → Risk          g m → Models
 *   g b → Backtest      g j → Journal       g , → Settings
 *   g a → Alerts        ?   → shortcut help dialog (handled externally)
 */
const ROUTES: Record<string, string> = {
  o: "/",
  p: "/positions",
  s: "/signals",
  f: "/performance",
  r: "/risk",
  m: "/models",
  b: "/backtest",
  j: "/journal",
  ",": "/settings",
  a: "/alerts",
};

export function useKeyboardShortcuts(onHelp?: () => void) {
  const router = useRouter();

  useEffect(() => {
    let leader = false;
    let leaderTimer: ReturnType<typeof setTimeout> | null = null;

    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName?.toUpperCase();
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === "?") {
        e.preventDefault();
        onHelp?.();
        return;
      }
      if (e.key === "g") {
        leader = true;
        if (leaderTimer) clearTimeout(leaderTimer);
        leaderTimer = setTimeout(() => { leader = false; }, 1200);
        return;
      }
      if (leader) {
        const target = ROUTES[e.key.toLowerCase()];
        if (target) {
          e.preventDefault();
          router.push(target);
        }
        leader = false;
        if (leaderTimer) {
          clearTimeout(leaderTimer);
          leaderTimer = null;
        }
      }
    }

    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      if (leaderTimer) clearTimeout(leaderTimer);
    };
  }, [router, onHelp]);
}

export const SHORTCUT_HELP = [
  { keys: "g o", label: "Overview" },
  { keys: "g p", label: "Positions" },
  { keys: "g s", label: "Signals" },
  { keys: "g f", label: "Performance" },
  { keys: "g r", label: "Risk" },
  { keys: "g m", label: "Models" },
  { keys: "g b", label: "Backtest" },
  { keys: "g j", label: "Journal" },
  { keys: "g ,", label: "Settings" },
  { keys: "g a", label: "Alerts" },
  { keys: "?",   label: "Show this help" },
];
