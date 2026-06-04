"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { ShortcutsModal } from "@/components/ui/ShortcutsModal";
import { ToastHost } from "@/components/ui/Toast";
import { NavProgressBar } from "@/components/ui/NavProgressBar";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { usePathname } from "next/navigation";
import { WsProvider } from "@/hooks/WsProvider";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const pathname = usePathname();

  useKeyboardShortcuts(() => setShortcutsOpen(true));

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileNavOpen(false);
  }, [pathname]);

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName?.toUpperCase();
      const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

      // Cmd/Ctrl+K — palette
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        return;
      }
      if (inField || e.metaKey || e.ctrlKey || e.altKey) return;

      // / — open palette
      if (e.key === "/") {
        e.preventDefault();
        setPaletteOpen(true);
        return;
      }
      // r — refresh current page data
      if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent("clawdbot:refresh"));
        return;
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <ToastHost>
    <WsProvider>
      <NavProgressBar />

      {/* Desktop sidebar */}
      <div className="hidden md:block">
        <Sidebar />
      </div>

      {/* Mobile drawer */}
      {mobileNavOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={() => setMobileNavOpen(false)}
        >
          <div
            className="absolute left-0 top-0 bottom-0 w-52 bg-[#10151D] border-r border-[#374151] shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <Sidebar />
          </div>
        </div>
      )}

      <div className="flex flex-col flex-1 overflow-hidden min-w-0">
        <div className="flex">
          <button
            onClick={() => setMobileNavOpen(true)}
            className="md:hidden h-12 px-3 text-[#9CA3AF] hover:text-[#F3F4F6] border-r border-[#374151] bg-[#10151D]"
            aria-label="Open menu"
          >
            ☰
          </button>
          <div className="flex-1">
            <TopBar
              onOpenPalette={() => setPaletteOpen(true)}
              onOpenShortcuts={() => setShortcutsOpen(true)}
            />
          </div>
        </div>
        <main className="flex-1 overflow-auto p-4 sm:p-6 page-fade" key={pathname}>
          {children}
        </main>
      </div>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      <ShortcutsModal open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </WsProvider>
    </ToastHost>
  );
}
