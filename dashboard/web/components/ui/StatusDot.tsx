interface Props {
  state: "live" | "stale" | "down" | "warn";
  size?: number;
  label?: string;
  pulse?: boolean;
}

const COLORS: Record<Props["state"], string> = {
  live: "#10B981",
  stale: "#F59E0B",
  warn: "#F59E0B",
  down: "#EF4444",
};

export function StatusDot({ state, size = 8, label, pulse = true }: Props) {
  const color = COLORS[state];
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="relative inline-flex" style={{ width: size, height: size }}>
        {pulse && state === "live" && (
          <span
            className="absolute inset-0 rounded-full animate-ping opacity-60"
            style={{ backgroundColor: color }}
          />
        )}
        <span
          className="relative inline-block rounded-full"
          style={{ width: size, height: size, backgroundColor: color }}
        />
      </span>
      {label && <span className="text-[10px] uppercase tracking-wider text-[#9CA3AF]">{label}</span>}
    </span>
  );
}
