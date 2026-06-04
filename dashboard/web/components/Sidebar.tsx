"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface NavItem {
  href: string;
  label: string;
  icon: string;
  shortcut: string;
  badgeUrl?: string;
}

const NAV: NavItem[] = [
  { href: "/",            label: "Overview",    icon: "◆", shortcut: "g o" },
  { href: "/positions",   label: "Positions",   icon: "▤", shortcut: "g p" },
  { href: "/signals",     label: "Signals",     icon: "↯", shortcut: "g s" },
  { href: "/performance", label: "Performance", icon: "∿", shortcut: "g f" },
  { href: "/risk",        label: "Risk",        icon: "⛨", shortcut: "g r" },
  { href: "/models",      label: "Models",      icon: "▥", shortcut: "g m" },
  { href: "/backtest",    label: "Backtest",    icon: "▷", shortcut: "g b" },
  { href: "/journal",     label: "Journal",     icon: "≣", shortcut: "g j" },
  { href: "/settings",    label: "Settings",    icon: "⚙", shortcut: "g ," },
  { href: "/alerts",      label: "Alerts",      icon: "!", shortcut: "g a", badgeUrl: `${API}/api/alerts?status=unread` },
];

export function Sidebar() {
  const path = usePathname();
  const [unread, setUnread] = useState<number>(0);
  const [collapsed, setCollapsed] = useState<boolean>(false);

  useEffect(() => {
    const saved = typeof window !== "undefined" ? localStorage.getItem("clawdbot:sidebar:collapsed") : null;
    if (saved === "1") setCollapsed(true);
  }, []);

  useEffect(() => {
    async function tick() {
      try {
        const res = await fetch(`${API}/api/alerts?status=unread`);
        const data = await res.json();
        setUnread(Array.isArray(data) ? data.length : 0);
      } catch {}
    }
    tick();
    const id = setInterval(tick, 30000);
    return () => clearInterval(id);
  }, []);

  function toggleCollapse() {
    const next = !collapsed;
    setCollapsed(next);
    if (typeof window !== "undefined") localStorage.setItem("clawdbot:sidebar:collapsed", next ? "1" : "0");
  }

  return (
    <aside
      className={`${collapsed ? "w-14" : "w-52"} bg-[#10151D] border-r border-[#374151] flex flex-col py-3 shrink-0 transition-[width] duration-150`}
    >
      <div className="px-3 mb-4 flex items-center justify-between">
        {!collapsed ? (
          <div className="flex items-center gap-1">
            <span className="font-mono text-[#3B82F6] font-bold text-sm">ClawdBot</span>
            <span className="text-[10px] text-[#6B7280]">v2</span>
          </div>
        ) : (
          <span className="font-mono text-[#3B82F6] font-bold text-sm">▲</span>
        )}
        <button
          onClick={toggleCollapse}
          className="text-[#6B7280] hover:text-[#F3F4F6] text-xs"
          aria-label="Collapse sidebar"
          title="Collapse/Expand sidebar"
        >
          {collapsed ? "›" : "‹"}
        </button>
      </div>

      <nav className="flex flex-col gap-0.5 px-2 flex-1">
        {NAV.map((item) => {
          const active = path === item.href;
          const showBadge = item.href === "/alerts" && unread > 0;
          return (
            <Link
              key={item.href}
              href={item.href}
              title={collapsed ? `${item.label} (${item.shortcut})` : undefined}
              className={`group relative flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-sm transition-colors ${
                active
                  ? "bg-[#3B82F6]/15 text-[#3B82F6]"
                  : "text-[#9CA3AF] hover:bg-white/5 hover:text-[#F3F4F6]"
              }`}
            >
              <span className={`text-base leading-none w-5 text-center ${active ? "" : "opacity-70"}`}>{item.icon}</span>
              {!collapsed && (
                <>
                  <span className="flex-1 truncate">{item.label}</span>
                  {showBadge && (
                    <span className="px-1.5 py-0.5 text-[9px] font-mono bg-[#EF4444] text-white rounded">
                      {unread > 99 ? "99+" : unread}
                    </span>
                  )}
                  <kbd className="hidden xl:inline-block opacity-0 group-hover:opacity-60 text-[9px] font-mono text-[#9CA3AF] bg-[#1E2530] border border-[#374151] rounded px-1 transition-opacity">
                    {item.shortcut}
                  </kbd>
                </>
              )}
              {collapsed && showBadge && (
                <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-[#EF4444]" />
              )}
            </Link>
          );
        })}
      </nav>

      {!collapsed && (
        <div className="px-3 py-2 border-t border-[#374151] mt-2">
          <div className="text-[9px] text-[#6B7280] uppercase tracking-widest">build</div>
          <div className="text-[10px] font-mono text-[#9CA3AF]">v2.4 · 2026-06-03</div>
        </div>
      )}
    </aside>
  );
}
