"use client";

import { useEffect, useState } from "react";
import { EmptyState } from "@/components/ui/EmptyState";
import { fmtTimeAgo, TABULAR } from "@/lib/format";

interface Signal {
  timestamp: string;
  symbol: string;
  cal_prob: number;
  threshold: number;
  ml_decision: string;
  executed: string;
  reason_skip?: string;
  regime_prob?: number;
}

const DECISION_STYLE: Record<string, { bg: string; fg: string }> = {
  BUY:         { bg: "bg-[#10B981]/15", fg: "text-[#10B981]" },
  SELL:        { bg: "bg-[#EF4444]/15", fg: "text-[#EF4444]" },
  HOLD:        { bg: "bg-[#374151]",    fg: "text-[#9CA3AF]" },
  SKIP_SPREAD: { bg: "bg-[#F59E0B]/10", fg: "text-[#F59E0B]/80" },
  SKIP_VOL:    { bg: "bg-[#F59E0B]/10", fg: "text-[#F59E0B]/80" },
  SKIP_HOUR:   { bg: "bg-[#F59E0B]/10", fg: "text-[#F59E0B]/80" },
};

const REASON_LABEL: Record<string, string> = {
  no_signal:    "prob < thr",
  regime_block: "ranging market",
  spread:       "spread wide",
  vol:          "low volume",
  hour:         "off-hours",
};

function ReasonChip({ reason, regimeProb }: { reason?: string; regimeProb?: number }) {
  if (!reason || reason === "no_signal" || reason === "") return null;
  const label = REASON_LABEL[reason] ?? reason.replace(/_/g, " ");
  const extra = reason === "regime_block" && regimeProb != null
    ? ` (${(regimeProb * 100).toFixed(0)}%)`
    : "";
  return (
    <span className="ml-1 text-[9px] text-[#6B7280] font-normal">· {label}{extra}</span>
  );
}

export function RecentSignalsTable({ apiBase }: { apiBase: string }) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const load = () =>
      fetch(`${apiBase}/api/signals/recent?limit=30`)
        .then((r) => r.json())
        .then((d) => { setSignals(Array.isArray(d) ? d : []); setLoaded(true); })
        .catch(() => setLoaded(true));
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [apiBase]);

  if (loaded && !signals.length) {
    return (
      <EmptyState
        icon="↯"
        title="No signals yet"
        body="Signals stream every ~15s once spreads normalize."
      />
    );
  }

  return (
    <div className="overflow-auto max-h-72">
      <table className="w-full text-xs font-mono">
        <thead className="sticky top-0 bg-[#10151D] z-10">
          <tr className="text-[#9CA3AF] border-b border-[#374151]">
            <th className="text-left pb-2 pr-2">Time</th>
            <th className="text-left pb-2">Symbol</th>
            <th className="text-right pb-2">Prob</th>
            <th className="text-right pb-2">Thr</th>
            <th className="text-left pb-2 pl-2">Decision</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => {
            const prob = parseFloat(String(s.cal_prob));
            const thr = parseFloat(String(s.threshold));
            const decision = String(s.ml_decision || "HOLD").toUpperCase();
            const style = DECISION_STYLE[decision] ?? DECISION_STYLE.HOLD;
            const reason = String(s.reason_skip ?? "");
            const regProb = typeof s.regime_prob === "number" ? s.regime_prob
              : typeof (s as any).regime_prob === "string" ? parseFloat((s as any).regime_prob) : undefined;
            return (
              <tr key={i} className="border-b border-[#374151]/40 hover:bg-white/5 transition-colors">
                <td className={`py-1.5 pr-2 text-[#6B7280] ${TABULAR}`}>{fmtTimeAgo(s.timestamp)}</td>
                <td className="py-1.5 text-[#F3F4F6]">{s.symbol}</td>
                <td className={`py-1.5 text-right ${TABULAR} ${prob >= thr ? "text-[#10B981]" : "text-[#F3F4F6]"}`}>
                  {(prob * 100).toFixed(1)}%
                </td>
                <td className={`py-1.5 text-right text-[#F59E0B]/70 ${TABULAR}`}>{(thr * 100).toFixed(0)}%</td>
                <td className="py-1.5 pl-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${style.bg} ${style.fg}`}>
                    {decision}
                  </span>
                  <ReasonChip reason={reason} regimeProb={regProb} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
