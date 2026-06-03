"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/positions", label: "Positions" },
  { href: "/signals", label: "Signals" },
  { href: "/performance", label: "Performance" },
  { href: "/risk", label: "Risk" },
  { href: "/models", label: "Models" },
  { href: "/backtest", label: "Backtest" },
  { href: "/journal", label: "Journal" },
  { href: "/settings", label: "Settings" },
  { href: "/alerts", label: "Alerts" },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <aside className="w-48 bg-bg-surface border-r border-border-default flex flex-col py-4 shrink-0">
      <div className="px-4 mb-6">
        <span className="font-mono text-accent-blue font-bold text-base">ClawdBot</span>
        <span className="text-text-secondary text-xs ml-1">v2</span>
      </div>
      <nav className="flex flex-col gap-0.5 px-2">
        {NAV.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`px-3 py-2 rounded-lg text-sm transition-colors ${
              path === href
                ? "bg-accent-blue/20 text-accent-blue font-medium"
                : "text-text-secondary hover:bg-bg-elevated hover:text-text-primary"
            }`}
          >
            {label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
