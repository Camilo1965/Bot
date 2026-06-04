/**
 * Pro-grade numeric/date formatting helpers.
 * All formatters return strings; never throw on bad input.
 */

const NUMBER_SUFFIXES = [
  { value: 1e9, suffix: "B" },
  { value: 1e6, suffix: "M" },
  { value: 1e3, suffix: "K" },
];

export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtAbbrev(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  for (const { value, suffix } of NUMBER_SUFFIXES) {
    if (abs >= value) {
      return `${sign}${(abs / value).toFixed(digits)}${suffix}`;
    }
  }
  return `${sign}${abs.toFixed(digits)}`;
}

export function fmtMoney(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `$${fmtNum(n, digits)}`;
}

export function fmtMoneyAbbrev(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `$${fmtAbbrev(n, digits)}`;
}

export function fmtPct(n: number | null | undefined, digits = 2, withSign = true): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = withSign && n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}%`;
}

export function fmtSignedMoney(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}$${fmtNum(Math.abs(n), digits)}`;
}

export function pnlColor(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "text-[#9CA3AF]";
  if (v > 0) return "text-[#10B981]";
  if (v < 0) return "text-[#EF4444]";
  return "text-[#9CA3AF]";
}

export function severityColor(s: "info" | "warn" | "critical" | string): string {
  if (s === "critical") return "bg-[#EF4444]/20 text-[#EF4444] border-[#EF4444]/40";
  if (s === "warn") return "bg-[#F59E0B]/20 text-[#F59E0B] border-[#F59E0B]/40";
  return "bg-[#3B82F6]/20 text-[#3B82F6] border-[#3B82F6]/40";
}

export function fmtTimeAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const ts = new Date(iso).getTime();
    if (Number.isNaN(ts)) return "—";
    const delta = Math.floor((Date.now() - ts) / 1000);
    if (delta < 5) return "now";
    if (delta < 60) return `${delta}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  } catch {
    return "—";
  }
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "—";
  }
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return "—";
  }
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  } catch {
    return "—";
  }
}

/** Tabular numeric class — ensures monospace digit width even in non-mono fonts. */
export const TABULAR = "tabular-nums";
