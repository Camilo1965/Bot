"use client";

import { fmtMoney, fmtPct, TABULAR } from "@/lib/format";
import type { LivePosition } from "@/components/LivePositionCard";

interface Props {
  position: LivePosition;
  buyProbThreshold?: number;
  ttlHours?: number;
}

interface Condition {
  label: string;
  detail: string;
  status: "win" | "loss" | "neutral" | "trail";
  hint?: string;
}

export function PositionExitConditions({ position, buyProbThreshold, ttlHours = 5 }: Props) {
  const isLong = position.side?.toLowerCase() === "long";
  const entry = position.entry_price;
  const sl = position.current_sl ?? position.sl_price;
  const tp = position.tp_price;
  const size = position.size_usd ?? 0;

  const slDistPct = entry > 0 ? Math.abs((sl - entry) / entry) * 100 : 0;
  const tpDistPct = entry > 0 ? Math.abs((tp - entry) / entry) * 100 : 0;
  const slDollarLoss = -(slDistPct / 100) * size;
  const tpDollarGain = (tpDistPct / 100) * size;
  const trailingArmed = position.trailing_active === true;

  const openedAt = position.opened_at ? Date.parse(position.opened_at) : null;
  const ttlExpiresAt = openedAt ? new Date(openedAt + ttlHours * 3600_000) : null;
  const minutesLeft = ttlExpiresAt ? Math.max(0, Math.floor((ttlExpiresAt.getTime() - Date.now()) / 60000)) : null;
  const hLeft = minutesLeft !== null ? Math.floor(minutesLeft / 60) : null;
  const mLeft = minutesLeft !== null ? minutesLeft % 60 : null;

  const mlEntry = position.ml_confidence;
  const thr = buyProbThreshold ?? 0.55;

  const conditions: Condition[] = [
    {
      label: isLong ? "Hits TP price (profit)" : "Hits TP price (short profit)",
      detail: `${tp.toFixed(4)} · ${fmtPct(isLong ? tpDistPct : -tpDistPct)} ≈ ${fmtMoney(tpDollarGain)}`,
      status: "win",
      hint: "Bot closes immediately and credits gross PnL.",
    },
    {
      label: isLong ? "Hits SL price (loss)" : "Hits SL price (short loss)",
      detail: `${sl.toFixed(4)} · ${fmtPct(isLong ? -slDistPct : slDistPct)} ≈ ${fmtMoney(slDollarLoss)}`,
      status: "loss",
      hint: "Trailing stop ratchets toward profit once activation_pct triggers.",
    },
    {
      label: "ML probability reversal",
      detail: mlEntry !== undefined
        ? `Drops below ${(thr * 100).toFixed(0)}% (opened at ${(mlEntry * 100).toFixed(1)}%)`
        : `Drops below ${(thr * 100).toFixed(0)}%`,
      status: "neutral",
      hint: "check_ml_exit closes the position when conviction dies, even before SL.",
    },
    {
      label: "TTL expiry (max hold)",
      detail: ttlExpiresAt
        ? `${ttlHours}h cap · ${hLeft}h ${mLeft}m left`
        : `${ttlHours}h cap from open time`,
      status: "neutral",
      hint: "Hard exit at TTL_HOURS to avoid stuck capital.",
    },
    {
      label: trailingArmed ? "Trailing stop ARMED" : "Trailing stop arms when",
      detail: trailingArmed
        ? "SL ratchets behind peak now"
        : `${isLong ? "price >" : "price <"} entry × (1 ${isLong ? "+" : "-"} activation_pct)`,
      status: "trail",
      hint: trailingArmed
        ? "Stop already moved to lock in profit. SL only goes one direction."
        : "Once price runs in your favor by activation_pct, SL begins to follow peak − trailing_distance_pct.",
    },
  ];

  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Exit conditions</div>
          <div className="text-[11px] text-[#6B7280] mt-0.5">
            {position.symbol} · {isLong ? "long" : "short"} · any condition below closes the position
          </div>
        </div>
        <div className="flex flex-col items-end leading-tight">
          <span className={`font-mono ${TABULAR} text-xs text-[#10B981]`}>+{fmtMoney(tpDollarGain)}</span>
          <span className={`font-mono ${TABULAR} text-[10px] text-[#EF4444]`}>{fmtMoney(slDollarLoss)}</span>
        </div>
      </div>

      <ul className="space-y-2">
        {conditions.map((c, i) => {
          const color =
            c.status === "win" ? "text-[#10B981] border-[#10B981]/30 bg-[#10B981]/5" :
            c.status === "loss" ? "text-[#EF4444] border-[#EF4444]/30 bg-[#EF4444]/5" :
            c.status === "trail" ? "text-[#3B82F6] border-[#3B82F6]/30 bg-[#3B82F6]/5" :
            "text-[#F59E0B] border-[#F59E0B]/20 bg-[#F59E0B]/5";
          return (
            <li key={i} className={`rounded border p-2.5 ${color}`}>
              <div className="flex items-start justify-between gap-3">
                <div className="text-xs font-semibold">{c.label}</div>
                <div className={`text-[11px] font-mono ${TABULAR} text-[#F3F4F6]`}>{c.detail}</div>
              </div>
              {c.hint && <div className="text-[10px] text-[#9CA3AF] mt-1">{c.hint}</div>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
