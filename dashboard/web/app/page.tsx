"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { KpiCard } from "@/components/ui/KpiCard";
import { OpenPositionsList } from "@/components/OpenPositionsList";
import { useWs, useWsEvent } from "@/hooks/WsProvider";
import { useRefreshKey } from "@/hooks/useRefreshKey";
import { useToast } from "@/components/ui/Toast";
import { fmtMoney, fmtPct, fmtTimeAgo, pnlColor, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface BotState {
  balance: number;
  equity: number;
  daily_pnl_pct: number;
  open_positions_count: number;
  kill_switch_active: boolean;
  kill_switch_reason?: string | null;
  last_updated: string;
  watchlist?: string[];
}

interface SymConfig {
  symbol: string;
  prob_threshold: number;
  enabled?: boolean;
}

interface SignalLive {
  cal_prob?: number;
  threshold?: number;
  decision?: string;
  regime_prob?: number | null;
  htf_pass?: boolean | null;
  reason_skip?: string;
  ts?: string;
}

interface SymbolRow {
  symbol: string;
  threshold: number;
  prob: number | null;
  decision: string;
  regimeProb: number | null;
  htfPass: boolean | null;
  reasonSkip: string;
  ts: string;
}

// ─── Helper: classify the blocking reason ────────────────────────────────────
function classifyStatus(r: SymbolRow): {
  label: string;
  color: string;
  detail: string;
} {
  const thr = r.threshold * 100;
  if (r.prob === null) return { label: "WAITING", color: "#9CA3AF", detail: "No ML score yet" };

  const probPct = r.prob * 100;
  const aboveThreshold = probPct >= thr;

  if (r.decision === "SKIP_SPREAD") return { label: "SPREAD", color: "#F59E0B", detail: "Broker spread too wide" };
  if (r.decision === "SKIP_VOL") return { label: "VOLUME", color: "#F59E0B", detail: "Volume below threshold" };
  if (r.decision === "SKIP_HOUR") return { label: "OFF-HOURS", color: "#F59E0B", detail: "Outside UTC 08–22" };
  if (r.decision === "BUY") return { label: "BUY!", color: "#10B981", detail: "Signal active — opening position" };

  if (aboveThreshold) {
    // prob ≥ threshold but not buying → downstream filter
    if (r.htfPass === false || r.reasonSkip === "sma200") {
      return { label: "SMA200", color: "#F59E0B", detail: "Price below 1H SMA200 — no longs in downtrend" };
    }
    if ((r.regimeProb !== null && r.regimeProb < 0.65) || r.reasonSkip === "regime") {
      const rPct = r.regimeProb !== null ? ` (${(r.regimeProb * 100).toFixed(0)}%)` : "";
      return { label: "RANGING", color: "#EF4444", detail: `Market is ranging${rPct} — entry blocked` };
    }
    return { label: "FILTERED", color: "#F59E0B", detail: "SMA200 or regime filter blocking entry" };
  }

  // Below threshold
  const gap = thr - probPct;
  if (r.regimeProb !== null && r.regimeProb < 0.65) {
    return { label: `−${gap.toFixed(1)}pts + RANGING`, color: "#EF4444", detail: `Below threshold AND market ranging (${(r.regimeProb * 100).toFixed(0)}%)` };
  }
  return { label: `−${gap.toFixed(1)} pts`, color: gap < 5 ? "#F59E0B" : "#9CA3AF", detail: `Need +${gap.toFixed(1)}% more confidence` };
}

// ─── Status dot ──────────────────────────────────────────────────────────────
function StatusDot({ color }: { color: string }) {
  return <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />;
}

// ─── Decision Matrix Row ─────────────────────────────────────────────────────
function DecisionRow({ row }: { row: SymbolRow }) {
  const thr = row.threshold * 100;
  const prob = row.prob !== null ? row.prob * 100 : null;
  const status = classifyStatus(row);
  const fillPct = prob !== null ? Math.min(100, (prob / Math.max(1, thr)) * 100) : 0;
  const barColor = row.decision === "BUY" ? "#10B981" : prob !== null && prob >= thr ? "#F59E0B" : "#3B82F6";

  return (
    <tr className="border-b border-[#374151]/40 hover:bg-white/[0.02] transition-colors group">
      {/* Symbol */}
      <td className="py-2.5 pl-3 pr-2 whitespace-nowrap">
        <div className="flex items-center gap-2">
          <StatusDot color={status.color} />
          <Link href="/signals" className="font-mono text-xs font-semibold text-[#F3F4F6] hover:text-[#3B82F6] transition-colors">
            {row.symbol.replace("/USDT", "")}
          </Link>
        </div>
      </td>

      {/* Prob bar */}
      <td className="py-2.5 px-2 w-36">
        {prob !== null ? (
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] font-mono">
              <span style={{ color: barColor }}>{prob.toFixed(1)}%</span>
              <span className="text-[#6B7280]">/{thr.toFixed(0)}%</span>
            </div>
            <div className="h-1 bg-[#1E2530] rounded-full overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${fillPct}%`, backgroundColor: barColor }} />
            </div>
          </div>
        ) : (
          <span className="text-[10px] text-[#6B7280]">—</span>
        )}
      </td>

      {/* SMA200 */}
      <td className="py-2.5 px-2 text-center">
        {row.htfPass === true ? (
          <span className="text-[10px] text-[#10B981]">✓</span>
        ) : row.htfPass === false ? (
          <span className="text-[10px] font-semibold text-[#F59E0B]">✗</span>
        ) : (
          <span className="text-[10px] text-[#374151]">—</span>
        )}
      </td>

      {/* Regime */}
      <td className="py-2.5 px-2 whitespace-nowrap">
        {row.regimeProb !== null ? (
          <span
            className={`text-[10px] font-mono ${TABULAR}`}
            style={{ color: row.regimeProb >= 0.65 ? "#10B981" : "#EF4444" }}
          >
            {row.regimeProb >= 0.65 ? "TREND" : "RANGE"} {(row.regimeProb * 100).toFixed(0)}%
          </span>
        ) : (
          <span className="text-[10px] text-[#374151]">—</span>
        )}
      </td>

      {/* Status */}
      <td className="py-2.5 pl-2 pr-3">
        <div title={status.detail} className="cursor-help">
          <span
            className="px-2 py-0.5 rounded text-[10px] font-semibold"
            style={{ backgroundColor: `${status.color}20`, color: status.color }}
          >
            {status.label}
          </span>
          <div className="text-[9px] text-[#6B7280] mt-0.5 max-w-[180px] truncate">{status.detail}</div>
        </div>
      </td>
    </tr>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────
export default function OverviewPage() {
  const [state, setState] = useState<BotState | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [config, setConfig] = useState<SymConfig[]>([]);
  const [symbolRows, setSymbolRows] = useState<SymbolRow[]>([]);
  const [refreshKey] = useRefreshKey();
  const toast = useToast();
  const ws = useWs();

  // Fetch bot state
  useEffect(() => {
    setLoaded(false);
    fetch(`${API}/api/state`)
      .then((r) => r.json())
      .then((st) => setState(st))
      .finally(() => setLoaded(true));
  }, [refreshKey]);

  // Fetch symbol config once
  useEffect(() => {
    fetch(`${API}/api/symbols/config`)
      .then((r) => r.json())
      .then((cfg) => {
        if (Array.isArray(cfg)) setConfig(cfg.filter((c: SymConfig) => c.enabled !== false));
      })
      .catch(() => {});
  }, []);

  // Fetch latest signal per symbol (REST fallback)
  const [restSignals, setRestSignals] = useState<Record<string, SignalLive>>({});
  useEffect(() => {
    if (config.length === 0) return;
    const load = () => {
      Promise.all(
        config.map((c) =>
          fetch(`${API}/api/signals/by-symbol/${c.symbol.replace("/", "_")}?limit=1`)
            .then((r) => r.json())
            .then((arr) => ({ sym: c.symbol, row: Array.isArray(arr) ? arr[0] : null }))
            .catch(() => ({ sym: c.symbol, row: null }))
        )
      ).then((results) => {
        const m: Record<string, SignalLive> = {};
        for (const { sym, row } of results) {
          if (row) m[sym] = {
            cal_prob: row.cal_prob ?? row.raw_prob,
            threshold: row.threshold,
            decision: row.ml_decision,
            regime_prob: row.regime_prob ?? null,
            htf_pass: row.htf_pass ?? null,
            reason_skip: row.reason_skip ?? "",
            ts: row.timestamp ?? "",
          };
        }
        setRestSignals(m);
      });
    };
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [config]);

  // Merge WS live signals with REST signals into display rows
  const symbolRowsMerged = useMemo((): SymbolRow[] => {
    return config.map((c) => {
      const live = ws.signals[c.symbol] as any;
      const rest = restSignals[c.symbol];
      const prob = live?.cal_prob ?? rest?.cal_prob ?? null;
      const decision = String(live?.decision ?? rest?.decision ?? "—").toUpperCase();
      return {
        symbol: c.symbol,
        threshold: live?.threshold ?? rest?.threshold ?? c.prob_threshold,
        prob: typeof prob === "number" ? prob : null,
        decision,
        regimeProb: live?.regime_prob ?? rest?.regime_prob ?? null,
        htfPass: live?.htf_pass ?? rest?.htf_pass ?? null,
        reasonSkip: String(live?.reason_skip ?? rest?.reason_skip ?? ""),
        ts: String(live?.ts ?? rest?.ts ?? ""),
      };
    });
  }, [config, ws.signals, restSignals]);

  // Update symbolRows whenever merged changes
  useEffect(() => {
    setSymbolRows(symbolRowsMerged);
  }, [symbolRowsMerged]);

  // WS live updates — equity ticks (kill_switch_reason only from REST state)
  useWsEvent("equity.tick", (msg) => {
    setState((prev) => prev ? {
      ...prev,
      equity: msg.equity,
      balance: msg.balance,
      daily_pnl_pct: msg.daily_pnl_pct,
      open_positions_count: msg.open_positions_count,
      kill_switch_active: msg.kill_switch_active ?? prev.kill_switch_active,
      last_updated: msg.ts,
    } : prev);
  });

  useWsEvent("position.opened", (msg) => {
    toast.push({ type: "info", title: `Position opened: ${msg.symbol}`, duration: 4000 });
  });
  useWsEvent("position.closed", (msg) => {
    const won = (msg.pnl_usd ?? 0) > 0;
    toast.push({
      type: won ? "success" : "warn",
      title: `Position closed${won ? " · win" : " · loss"}`,
      body: `PnL ${fmtMoney(msg.pnl_usd ?? 0)} · ${msg.reason ?? ""}`,
      duration: 5000,
    });
  });
  useWsEvent("kill_switch.changed", (msg) => {
    toast.push({
      type: msg.active ? "error" : "success",
      title: msg.active ? "Kill switch TRIGGERED" : "Kill switch reset",
      body: msg.reason,
      duration: 8000,
    });
  });

  // Hour filter status (client-side, uses UTC hour)
  const nowUtc = new Date();
  const utcH = nowUtc.getUTCHours();
  const hourOpen = utcH >= 8 && utcH < 22;

  // Prefer WsProvider's lastEquityTick (persists across SPA navigation) over
  // local state (resets to null on remount, causing flash of zero).
  const tick = ws.lastEquityTick;
  const liveBalance = tick?.balance ?? state?.balance ?? null;
  const liveEquity = tick?.equity ?? state?.equity ?? null;
  const liveDailyPnl = tick?.daily_pnl_pct ?? state?.daily_pnl_pct ?? null;
  const livePositions = tick?.open_positions_count ?? state?.open_positions_count ?? 0;
  const liveKillSwitch = tick?.kill_switch_active ?? state?.kill_switch_active ?? false;
  const liveLastUpdated = tick?.ts ?? state?.last_updated;

  const hasOpenPositions = livePositions > 0;

  // Summary stats for decision matrix
  const buyCount = symbolRows.filter((r) => r.decision === "BUY").length;
  const filteredCount = symbolRows.filter((r) => {
    if (r.prob === null) return false;
    const atThr = r.prob * 100 >= r.threshold * 100;
    return atThr && r.decision !== "BUY" && !r.decision.startsWith("SKIP_");
  }).length;
  const readyCount = symbolRows.filter((r) => {
    if (r.prob === null) return false;
    return r.prob * 100 >= r.threshold * 100;
  }).length;

  return (
    <div className="space-y-4">
      {/* ── Status bar ─────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Live indicator */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#10151D] border border-[#374151]">
          <span className={`w-1.5 h-1.5 rounded-full ${liveKillSwitch ? "bg-[#EF4444]" : "bg-[#10B981] animate-pulse"}`} />
          <span className="text-[10px] font-semibold text-[#9CA3AF] uppercase tracking-wider">
            {liveKillSwitch ? "HALTED" : "LIVE"}
          </span>
        </div>

        {/* Kill switch */}
        {liveKillSwitch && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#EF4444]/10 border border-[#EF4444]/30">
            <span className="text-[10px] font-semibold text-[#EF4444] uppercase">KILL SWITCH</span>
            {state?.kill_switch_reason && (
              <span className="text-[10px] text-[#EF4444]/70">· {state.kill_switch_reason}</span>
            )}
          </div>
        )}

        {/* Hour filter */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border text-[10px] font-semibold uppercase tracking-wider ${
          hourOpen
            ? "bg-[#10B981]/10 border-[#10B981]/30 text-[#10B981]"
            : "bg-[#374151]/30 border-[#374151] text-[#9CA3AF]"
        }`}>
          {hourOpen ? "TRADING HOURS" : "OFF-HOURS"} · UTC {String(utcH).padStart(2, "0")}:xx
        </div>

        {/* Last update */}
        <div className="ml-auto text-[10px] text-[#6B7280] font-mono">
          updated <span className={TABULAR}>{fmtTimeAgo(liveLastUpdated)}</span>
        </div>
      </div>

      {/* ── KPI strip ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KpiCard
          label="Balance"
          value={fmtMoney(liveBalance)}
          numericValue={liveBalance}
          numericFormat={(n) => fmtMoney(n)}
          loading={!loaded && liveBalance === null}
          hint="Cash + open margin"
        />
        <KpiCard
          label="Equity"
          value={fmtMoney(liveEquity)}
          numericValue={liveEquity}
          numericFormat={(n) => fmtMoney(n)}
          loading={!loaded && liveEquity === null}
          hint="Balance + unrealized PnL"
        />
        <KpiCard
          label="Daily PnL"
          value={fmtPct(liveDailyPnl)}
          numericValue={liveDailyPnl}
          numericFormat={(n) => fmtPct(n)}
          valueClass={pnlColor(liveDailyPnl ?? 0)}
          loading={!loaded && liveDailyPnl === null}
        />
        <KpiCard
          label="Positions"
          value={`${livePositions} / 2`}
          numericValue={livePositions}
          numericFormat={(n) => `${Math.round(n)} / 2`}
          loading={!loaded && tick === null}
          hint="Open / max concurrent"
        />
      </div>

      {/* ── Decision Matrix + Positions ────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Decision Matrix — 2/3 width */}
        <div className="lg:col-span-2 bg-[#10151D] border border-[#374151] rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-[#374151]">
            <div>
              <h2 className="text-sm font-semibold text-[#F3F4F6]">Signal Pipeline</h2>
              <p className="text-[10px] text-[#9CA3AF] mt-0.5">
                {readyCount > 0 ? (
                  buyCount > 0
                    ? `${buyCount} BUY signal active`
                    : `${filteredCount} symbol${filteredCount !== 1 ? "s" : ""} above threshold — blocked by SMA200/Regime`
                ) : "No symbols above threshold"}
              </p>
            </div>
            <Link href="/signals" className="text-[10px] text-[#3B82F6] hover:text-[#60A5FA] transition-colors">
              Signals →
            </Link>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-[#374151]/60">
                  <th className="py-2 pl-3 pr-2 text-[10px] uppercase tracking-wider text-[#6B7280] font-semibold">Symbol</th>
                  <th className="py-2 px-2 text-[10px] uppercase tracking-wider text-[#6B7280] font-semibold">ML Prob / Thr</th>
                  <th className="py-2 px-2 text-[10px] uppercase tracking-wider text-[#6B7280] font-semibold text-center">SMA200</th>
                  <th className="py-2 px-2 text-[10px] uppercase tracking-wider text-[#6B7280] font-semibold">Regime</th>
                  <th className="py-2 pl-2 pr-3 text-[10px] uppercase tracking-wider text-[#6B7280] font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {symbolRows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-[#9CA3AF] text-xs">
                      Loading symbol data…
                    </td>
                  </tr>
                ) : (
                  symbolRows.map((r) => <DecisionRow key={r.symbol} row={r} />)
                )}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div className="px-4 py-2 border-t border-[#374151]/40 flex flex-wrap gap-3 text-[9px] text-[#6B7280]">
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-sm bg-[#10B981]/40" /> BUY — signal active</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-sm bg-[#F59E0B]/40" /> SMA200/FILTERED — prob above thr, downstream blocked</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-sm bg-[#EF4444]/40" /> RANGING — regime filter active</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-sm bg-[#3B82F6]/40" /> −Xpts — below threshold</span>
          </div>
        </div>

        {/* Positions / No-positions — 1/3 width */}
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4">
          <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] mb-3 font-semibold">
            {hasOpenPositions ? "Open Positions" : "Positions"}
          </h2>
          {hasOpenPositions ? (
            <OpenPositionsList apiBase={API} />
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center space-y-2">
              <span className="text-2xl opacity-20">⊙</span>
              <div className="text-[#9CA3AF] text-xs">No open positions</div>
              {readyCount > 0 && filteredCount === readyCount && (
                <div className="text-[10px] text-[#F59E0B]/80 mt-1 max-w-[160px]">
                  {filteredCount} symbol{filteredCount !== 1 ? "s" : ""} near signal — SMA200 blocking
                </div>
              )}
              <Link href="/positions" className="mt-2 text-[10px] text-[#3B82F6] hover:text-[#60A5FA]">
                View history →
              </Link>
            </div>
          )}
        </div>
      </div>

      {/* ── What's blocking + Recent Signals ───────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Readiness detail */}
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Why isn't the bot trading?</h2>
            <Link href="/risk" className="text-[10px] text-[#3B82F6] hover:text-[#60A5FA]">Risk →</Link>
          </div>
          <div className="space-y-2">
            {symbolRows.map((r) => {
              const status = classifyStatus(r);
              const thr = r.threshold * 100;
              const prob = r.prob !== null ? r.prob * 100 : null;
              return (
                <div key={r.symbol} className="flex items-start gap-3 py-1.5 border-b border-[#374151]/30 last:border-0">
                  <StatusDot color={status.color} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs text-[#F3F4F6] font-semibold">{r.symbol.replace("/USDT", "")}</span>
                      <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded" style={{ color: status.color, backgroundColor: `${status.color}15` }}>
                        {status.label}
                      </span>
                    </div>
                    <div className="text-[10px] text-[#9CA3AF] mt-0.5">{status.detail}</div>
                    {prob !== null && (
                      <div className={`text-[9px] font-mono ${TABULAR} text-[#6B7280] mt-0.5`}>
                        prob {prob.toFixed(1)}% / thr {thr.toFixed(0)}%
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
            {symbolRows.length === 0 && (
              <div className="text-[#9CA3AF] text-xs py-4 text-center">Loading…</div>
            )}
          </div>
        </div>

        {/* Quick links / info panel */}
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3">
          <h2 className="text-[10px] uppercase tracking-widest text-[#9CA3AF] font-semibold">Quick Access</h2>
          <div className="grid grid-cols-2 gap-2">
            {[
              { href: "/signals", label: "Signals", desc: "Live ML probabilities + charts" },
              { href: "/positions", label: "Positions", desc: "Open & trade history" },
              { href: "/performance", label: "Performance", desc: "PnL, Sharpe, heatmaps" },
              { href: "/risk", label: "Risk", desc: "Kill switch, drawdown, limits" },
              { href: "/models", label: "Models", desc: "AUC, retrain, feature counts" },
              { href: "/alerts", label: "Alerts", desc: "Kill events, drift warnings" },
            ].map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="p-3 rounded-md bg-[#0a0e15] border border-[#374151] hover:border-[#4B5563] transition-colors group"
              >
                <div className="text-xs font-semibold text-[#F3F4F6] group-hover:text-[#3B82F6] transition-colors">{link.label}</div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">{link.desc}</div>
              </Link>
            ))}
          </div>

          {/* Bot config summary */}
          <div className="border-t border-[#374151] pt-3 space-y-1">
            <div className="text-[9px] uppercase tracking-wider text-[#6B7280] mb-1">Active config</div>
            {config.map((c) => (
              <div key={c.symbol} className={`flex items-center justify-between text-[10px] font-mono ${TABULAR}`}>
                <span className="text-[#9CA3AF]">{c.symbol}</span>
                <span className="text-[#F59E0B]">thr {(c.prob_threshold * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
