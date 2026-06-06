"use client";

import Link from "next/link";
import { TABULAR } from "@/lib/format";

interface Step {
  num: number;
  title: string;
  body: string;
  color: string;
}

const STEPS: Step[] = [
  {
    num: 1,
    title: "ML probability ≥ symbol threshold",
    body: "Every ~15s the bot scores each symbol with the XGBoost model. Decision is BUY only if calibrated probability ≥ prob_threshold from SYMBOL_CONFIG (e.g. 0.85 for BTC, 0.55 for DOGE).",
    color: "#3B82F6",
  },
  {
    num: 2,
    title: "Broker spread tight enough",
    body: "Spread/midprice must be below MAX_SPREAD_PCT (0.20% default). Above that the symbol is SKIP_SPREAD until the broker normalizes.",
    color: "#10B981",
  },
  {
    num: 3,
    title: "Volume + trading hour gates",
    body: "Last candle volume ≥ 0.5× SMA(20). UTC hour inside [08, 22) when TRADE_HOUR_FILTER is on. Misses become SKIP_VOL / SKIP_HOUR.",
    color: "#F59E0B",
  },
  {
    num: 4,
    title: "Risk budget available",
    body: "MAX_POSITIONS=2 concurrent, total risk ≤ portfolio cap, daily loss < kill switch threshold, and risk manager confirms balance can margin the trade.",
    color: "#9CA3AF",
  },
  {
    num: 5,
    title: "Trade opens with auto SL / TP / trailing",
    body: "Position size is Kelly-scaled by ML probability and SL distance. Initial SL = entry × (1 − sl_pct). TP = entry × (1 + tp_pct) or ATR-scaled. Trailing stop arms once price moves activation_pct in your favor.",
    color: "#EF4444",
  },
];

export function TradeLifecycleExplainer() {
  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-5 space-y-4">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-[#F3F4F6]">How a trade is born</h3>
        <p className="text-xs text-[#9CA3AF]">
          Five gates must pass in order. Once a trade is open the bot manages SL/TP/trailing/TTL automatically.
        </p>
      </div>

      <ol className="space-y-2">
        {STEPS.map((s) => (
          <li key={s.num} className="flex gap-3">
            <div
              className={`shrink-0 w-7 h-7 rounded-md flex items-center justify-center font-mono ${TABULAR} text-xs font-bold border`}
              style={{ color: s.color, borderColor: `${s.color}55`, background: `${s.color}10` }}
            >
              {s.num}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold text-[#F3F4F6]">{s.title}</div>
              <div className="text-[11px] text-[#9CA3AF] leading-relaxed mt-0.5">{s.body}</div>
            </div>
          </li>
        ))}
      </ol>

      <div className="flex flex-wrap gap-2 pt-1 border-t border-[#374151]">
        <Link
          href="/signals"
          className="px-3 py-1.5 rounded text-xs font-medium bg-[#3B82F6]/15 text-[#3B82F6] hover:bg-[#3B82F6]/25 transition-colors"
        >
          See live probabilities →
        </Link>
        <Link
          href="/settings"
          className="px-3 py-1.5 rounded text-xs font-medium bg-[#374151] text-[#9CA3AF] hover:text-[#F3F4F6] transition-colors"
        >
          Tune thresholds in Settings
        </Link>
      </div>
    </div>
  );
}
