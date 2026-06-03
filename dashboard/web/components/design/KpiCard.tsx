interface KpiCardProps {
  label: string;
  value: string;
  valueClass?: string;
  sub?: string;
}

export function KpiCard({ label, value, valueClass = "text-text-primary", sub }: KpiCardProps) {
  return (
    <div className="bg-bg-surface border border-border-default rounded-card p-4">
      <div className="text-[11px] text-text-secondary uppercase tracking-wider mb-1">{label}</div>
      <div className={`font-mono text-xl font-bold tabular ${valueClass}`}>{value}</div>
      {sub && <div className="text-[11px] text-text-secondary mt-1">{sub}</div>}
    </div>
  );
}
