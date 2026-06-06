"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useWs } from "@/hooks/WsProvider";
import { TABULAR } from "@/lib/format";

interface SymConfig {
  symbol: string;
  prob_threshold: number;
  enabled?: boolean;
  timeframe?: string;
}

interface SignalRow {
  cal_prob?: number;
  raw_prob?: number;
  threshold?: number;
  ml_decision?: string;
  timestamp?: string;
  regime_prob?: number | null;
  htf_pass?: boolean | null;
  reason_skip?: string;
}

interface SymReadiness {
  symbol: string;
  threshold: number;
  prob: number | null;
  decision: string;
  blocked: boolean;
  blockedBy: string | null;
  regimeProb: number | null;
  htfPass: boolean | null;
  reasonSkip: string;
  ts: string;
}

const GATE_HINT: Record<string, string> = {
  SKIP_SPREAD: "broker spread above max",
  SKIP_VOL: "volume < 0.5× sma(20)",
  SKIP_HOUR: "outside UTC 08–22",
};

const GATE_LABEL: Record<string, string> = {
  SKIP_SPREAD: "SPREAD",
  SKIP_VOL: "VOLUME",
  SKIP_HOUR: "HOUR",
};

interface Props {
  apiBase: string;
  title?: string;
  body?: string;
}

export function SymbolReadinessGrid({ apiBase, title = "Signal pipeline", body }: Props) {
  const ws = useWs();
  const [config, setConfig] = useState<SymConfig[]>([]);
  const [byRest, setByRest] = useState<Record<string, SignalRow>>({});
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/api/symbols/config`)
      .then((r) => r.json())
      .then((cfg) => {
        const list = Array.isArray(cfg) ? cfg.filter((c: SymConfig) => c.enabled !== false) : [];
        setConfig(list);
      })
      .catch(() => {});
  }, [apiBase]);

  useEffect(() => {
    if (config.length === 0) return;
    let cancelled = false;
    const load = () => {
      Promise.all(
        config.map((c) =>
          fetch(`${apiBase}/api/signals/by-symbol/${c.symbol.replace("/", "_")}?limit=1`)
            .then((r) => r.json())
            .then((arr) => ({ sym: c.symbol, row: Array.isArray(arr) ? arr[0] : null }))
            .catch(() => ({ sym: c.symbol, row: null }))
        )
      ).then((results) => {
        if (cancelled) return;
        const m: Record<string, SignalRow> = {};
        for (const { sym, row } of results) if (row) m[sym] = row as SignalRow;
        setByRest(m);
        setLoaded(true);
      });
    };
    load();
    const id = setInterval(load, 12000);
    return () => { cancelled = true; clearInterval(id); };
  }, [apiBase, config]);

  const rows: SymReadiness[] = useMemo(() => {
    return config.map((c) => {
      const liveSig = ws.signals[c.symbol];
      const restSig = byRest[c.symbol];
      const decision = String(liveSig?.decision ?? restSig?.ml_decision ?? "—").toUpperCase();
      const probValRaw = liveSig?.cal_prob ?? restSig?.cal_prob ?? restSig?.raw_prob ?? null;
      const blocked = decision.startsWith("SKIP_");
      const prob = blocked ? null : (typeof probValRaw === "number" ? probValRaw : null);
      const threshold = liveSig?.threshold ?? restSig?.threshold ?? c.prob_threshold ?? 0.55;
      const ts = String(liveSig?.ts ?? restSig?.timestamp ?? "");
      const regimeProb = (liveSig as any)?.regime_prob ?? restSig?.regime_prob ?? null;
      const htfPass = (liveSig as any)?.htf_pass ?? restSig?.htf_pass ?? null;
      const reasonSkip = String((liveSig as any)?.reason_skip ?? restSig?.reason_skip ?? "");
      return {
        symbol: c.symbol,
        threshold,
        prob,
        decision,
        blocked,
        blockedBy: blocked ? decision : null,
        regimeProb: typeof regimeProb === "number" ? regimeProb : null,
        htfPass: typeof htfPass === "boolean" ? htfPass : null,
        reasonSkip,
        ts,
      };
    });
  }, [config, ws.signals, byRest]);

  return (
    <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h3 className="text-sm font-semibold text-[#F3F4F6]">{title}</h3>
          <p className="text-[11px] text-[#9CA3AF] mt-0.5">
            {body ?? "Per-symbol blocking reason. FILTERED = prob ≥ threshold but SMA200 or Regime blocks entry."}
          </p>
        </div>
        <Link
          href="/signals"
          className="px-2.5 py-1 rounded text-[10px] font-semibold bg-[#3B82F6]/15 text-[#3B82F6] hover:bg-[#3B82F6]/25 transition-colors"
        >
          Live signals →
        </Link>
      </div>

      {!loaded && config.length === 0 ? (
        <div className="text-[#9CA3AF] text-xs py-3">Loading watchlist…</div>
      ) : rows.length === 0 ? (
        <div className="text-[#9CA3AF] text-xs py-3">No enabled symbols in SYMBOL_CONFIG.</div>
      ) : (
        <ul className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2">
          {rows.map((r) => (
            <ReadinessRow key={r.symbol} row={r} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ReadinessRow({ row }: { row: SymReadiness }) {
  const thrPct = (row.threshold ?? 0) * 100;

  // Gate-blocked (SKIP_SPREAD / SKIP_VOL / SKIP_HOUR)
  if (row.blocked && row.blockedBy) {
    return (
      <li className="bg-[#0a0e15] border border-[#F59E0B]/25 rounded p-2.5">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[#F3F4F6] text-xs font-semibold">{row.symbol}</span>
          <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold bg-[#F59E0B]/15 text-[#F59E0B]">
            {GATE_LABEL[row.blockedBy] ?? row.blockedBy.replace("SKIP_", "")}
          </span>
        </div>
        <div className="text-[10px] text-[#9CA3AF] mt-1">
          {GATE_HINT[row.blockedBy] ?? row.blockedBy.toLowerCase()}
        </div>
        <div className={`text-[9px] font-mono ${TABULAR} text-[#6B7280] mt-1`}>
          target prob ≥ {thrPct.toFixed(0)}%
        </div>
      </li>
    );
  }

  // No data yet
  if (row.prob === null) {
    return (
      <li className="bg-[#0a0e15] border border-[#374151]/50 rounded p-2.5">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[#F3F4F6] text-xs font-semibold">{row.symbol}</span>
          <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold bg-[#374151] text-[#9CA3AF]">WAITING</span>
        </div>
        <div className="text-[10px] text-[#9CA3AF] mt-1">ML hasn't scored this symbol yet.</div>
        <div className={`text-[9px] font-mono ${TABULAR} text-[#6B7280] mt-1`}>target prob ≥ {thrPct.toFixed(0)}%</div>
      </li>
    );
  }

  const probPct = row.prob * 100;
  const gap = thrPct - probPct;
  const atThreshold = gap <= 0;
  const actuallyBuying = row.decision === "BUY";

  // Prob above threshold but downstream filter (SMA200 or Regime) is blocking
  const filteredDownstream = atThreshold && !actuallyBuying;

  // Determine the filter reason
  let filterReason = "";
  if (filteredDownstream) {
    if (row.htfPass === false) {
      filterReason = "SMA200 1H";
    } else if (row.regimeProb !== null && row.regimeProb < 0.65) {
      filterReason = `RANGING (${(row.regimeProb * 100).toFixed(0)}%)`;
    } else if (row.reasonSkip === "sma200") {
      filterReason = "SMA200 1H";
    } else if (row.reasonSkip === "regime") {
      filterReason = "REGIME";
    } else {
      filterReason = "SMA200 / Regime";
    }
  }

  const color = actuallyBuying ? "#10B981" : filteredDownstream ? "#F59E0B" : gap < 5 ? "#F59E0B" : "#3B82F6";
  const fillPct = Math.min(100, (probPct / Math.max(1, thrPct)) * 100);

  return (
    <li className="bg-[#0a0e15] border rounded p-2.5" style={{ borderColor: `${color}40` }}>
      <div className="flex items-center justify-between">
        <span className="font-mono text-[#F3F4F6] text-xs font-semibold">{row.symbol}</span>
        <span
          className="px-1.5 py-0.5 rounded text-[9px] font-semibold"
          style={{ backgroundColor: `${color}25`, color }}
        >
          {actuallyBuying ? "BUY!" : filteredDownstream ? "FILTERED" : `Δ ${gap.toFixed(1)} pts`}
        </span>
      </div>

      {filteredDownstream && filterReason && (
        <div className="text-[10px] text-[#F59E0B]/80 mt-1 flex items-center gap-1">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#F59E0B]/60 shrink-0" />
          blocked by {filterReason}
        </div>
      )}

      <div className="relative h-1.5 bg-[#1E2530] rounded-full mt-2 overflow-hidden">
        <div
          className="absolute top-0 left-0 h-full rounded-full transition-all"
          style={{ width: `${fillPct}%`, backgroundColor: color }}
        />
      </div>
      <div className={`flex justify-between mt-1 text-[10px] font-mono ${TABULAR} text-[#9CA3AF]`}>
        <span>prob <span className="text-[#F3F4F6]">{probPct.toFixed(1)}%</span></span>
        <span>thr <span className="text-[#F59E0B]">{thrPct.toFixed(0)}%</span></span>
      </div>

      {row.regimeProb !== null && row.regimeProb !== undefined && (
        <div className={`text-[9px] font-mono ${TABULAR} mt-1`} style={{ color: row.regimeProb >= 0.65 ? "#10B981" : "#EF4444" }}>
          regime: {row.regimeProb >= 0.65 ? "TRENDING" : "RANGING"} ({(row.regimeProb * 100).toFixed(0)}%)
        </div>
      )}
    </li>
  );
}
