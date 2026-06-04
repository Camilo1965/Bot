import { Sparkline } from "./Sparkline";
import { fmtPct, TABULAR } from "@/lib/format";

interface Props {
  label: string;
  value: string;
  valueClass?: string;
  delta?: number | null;            // percent change for trend pill
  deltaLabel?: string;              // "24h", "7d"
  sparkline?: number[];
  hint?: string;                    // tooltip
  loading?: boolean;
}

export function KpiCard({ label, value, valueClass = "text-[#F3F4F6]", delta, deltaLabel, sparkline, hint, loading }: Props) {
  if (loading) {
    return (
      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-3 space-y-2">
        <div className="h-2 w-1/2 bg-[#1E2530] rounded animate-pulse" />
        <div className="h-6 w-3/4 bg-[#1E2530] rounded animate-pulse" />
      </div>
    );
  }
  const deltaColor =
    delta === undefined || delta === null
      ? "text-[#9CA3AF]"
      : delta > 0
      ? "text-[#10B981]"
      : delta < 0
      ? "text-[#EF4444]"
      : "text-[#9CA3AF]";
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-3 hover:border-[#4B5563] transition-colors" title={hint}>
      <div className="flex items-start justify-between gap-2 mb-1">
        <span className="text-[10px] uppercase tracking-widest text-[#9CA3AF] truncate">{label}</span>
        {sparkline && sparkline.length > 1 && (
          <Sparkline data={sparkline} width={50} height={18} />
        )}
      </div>
      <div className={`text-lg font-mono font-semibold ${TABULAR} ${valueClass}`}>{value}</div>
      {(delta !== undefined && delta !== null) && (
        <div className={`text-[11px] font-mono ${TABULAR} ${deltaColor} mt-0.5`}>
          {fmtPct(delta, 2, true)} {deltaLabel && <span className="opacity-60">· {deltaLabel}</span>}
        </div>
      )}
    </div>
  );
}
