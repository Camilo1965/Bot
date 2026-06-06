"use client";

import { useEffect, useMemo, useState } from "react";
import { TradingViewChart } from "@/components/TradingViewChart";
import { TradeLifecycleExplainer } from "@/components/TradeLifecycleExplainer";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { useWs } from "@/hooks/WsProvider";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { fmtTime, TABULAR } from "@/lib/format";
import { API_BASE as API } from "@/lib/api";

const SYMBOLS_FALLBACK = [
  "BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "NEAR/USDT",
];

const INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"] as const;
type IntervalKey = (typeof INTERVALS)[number];
type ChartSize = "S" | "M" | "L" | "XL";
type FilterKey = "all" | "buying" | "close" | "blocked" | "waiting";

const SIZE_TO_CSS: Record<ChartSize, string> = {
  S: "560px",
  M: "calc(100vh - 220px)",
  L: "calc(100vh - 80px)",
  XL: "calc(100vh - 80px)",
};

interface SignalRow {
  ts: string;
  cal_prob: number;
  threshold: number;
  ml_decision: string;
  regime_prob?: number | null;
  htf_pass?: boolean | null;
  reason_skip?: string;
}

interface SymbolSignals {
  symbol: string;
  rows: SignalRow[];
  loading: boolean;
  error: string | null;
}

interface SymState {
  label: string;
  color: string;
  detail: string;
  filterKey: FilterKey;
}

function classifySymbol(row: SignalRow | undefined): SymState {
  if (!row) return { label: "WAITING", color: "#9CA3AF", detail: "No ML score yet", filterKey: "waiting" };

  const decision = row.ml_decision.toUpperCase();
  const probPct = row.cal_prob * 100;
  const thrPct = row.threshold * 100;
  const gap = thrPct - probPct;

  if (decision === "SKIP_SPREAD") return { label: "SPREAD", color: "#F59E0B", detail: "Broker spread too wide", filterKey: "blocked" };
  if (decision === "SKIP_VOL") return { label: "LOW VOL", color: "#F59E0B", detail: "Volume < 0.5× SMA(20)", filterKey: "blocked" };
  if (decision === "SKIP_HOUR") return { label: "OFF-HOURS", color: "#F59E0B", detail: "Outside UTC 08–22", filterKey: "blocked" };
  if (decision === "BUY" || decision === "LONG") return { label: "BUY", color: "#10B981", detail: "Signal active — entry pending", filterKey: "buying" };

  if (probPct >= thrPct) {
    if (row.htf_pass === false || row.reason_skip === "sma200") {
      return { label: "SMA200", color: "#F59E0B", detail: "Price below 1H SMA200 — no longs", filterKey: "blocked" };
    }
    if ((row.regime_prob !== null && row.regime_prob !== undefined && row.regime_prob < 0.65) || row.reason_skip === "regime") {
      const rPct = row.regime_prob != null ? ` (${(row.regime_prob * 100).toFixed(0)}%)` : "";
      return { label: "RANGING", color: "#EF4444", detail: `Market is ranging${rPct} — entry blocked`, filterKey: "blocked" };
    }
    return { label: "FILTERED", color: "#F59E0B", detail: "Downstream filter blocking entry", filterKey: "blocked" };
  }

  if (gap < 5) return { label: `Δ ${gap.toFixed(1)}pt`, color: "#F59E0B", detail: `${gap.toFixed(1)} pts below threshold`, filterKey: "close" };
  return { label: `Δ ${gap.toFixed(0)}pt`, color: "#3B82F6", detail: `${gap.toFixed(1)} pts below threshold`, filterKey: "all" };
}

// ─── Stats bar ───────────────────────────────────────────────────────────────

function StatsBar({ columns, activeFilter, onFilter }: {
  columns: SymbolSignals[];
  activeFilter: FilterKey;
  onFilter: (k: FilterKey) => void;
}) {
  const stats = useMemo(() => {
    let buying = 0, close = 0, blocked = 0, waiting = 0;
    for (const col of columns) {
      const s = classifySymbol(col.rows[0]);
      if (s.filterKey === "buying") buying++;
      else if (s.filterKey === "close") close++;
      else if (s.filterKey === "blocked") blocked++;
      else if (s.filterKey === "waiting") waiting++;
    }
    return { buying, close, blocked, waiting };
  }, [columns]);

  const filters: { key: FilterKey; label: string; count: number; color: string }[] = [
    { key: "all", label: "All", count: columns.length, color: "#9CA3AF" },
    { key: "buying", label: "Buying", count: stats.buying, color: "#10B981" },
    { key: "close", label: "Close", count: stats.close, color: "#F59E0B" },
    { key: "blocked", label: "Blocked", count: stats.blocked, color: "#EF4444" },
    { key: "waiting", label: "Waiting", count: stats.waiting, color: "#6B7280" },
  ];

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {filters.map((f) => (
        <button
          key={f.key}
          onClick={() => onFilter(f.key)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
            activeFilter === f.key
              ? "text-[#0a0e15] font-semibold"
              : "bg-[#10151D] border border-[#374151] text-[#9CA3AF] hover:border-[#4B5563] hover:text-[#F3F4F6]"
          }`}
          style={activeFilter === f.key ? { backgroundColor: f.color, borderColor: f.color } : {}}
        >
          <span>{f.label}</span>
          <span
            className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
              activeFilter === f.key ? "bg-black/20 text-white" : ""
            }`}
            style={activeFilter !== f.key ? { color: f.color } : {}}
          >
            {f.count}
          </span>
        </button>
      ))}
    </div>
  );
}

// ─── Prob bar ────────────────────────────────────────────────────────────────

function ProbBar({ prob, threshold, color }: { prob: number; threshold: number; color: string }) {
  const fillPct = Math.min(100, prob * 100);
  const thrPct = Math.min(100, threshold * 100);
  return (
    <div className="relative h-2.5 bg-[#0a0e15] rounded-full overflow-visible">
      <div
        className="absolute top-0 left-0 h-full rounded-full transition-all duration-300"
        style={{ width: `${fillPct}%`, backgroundColor: color }}
      />
      <div
        className="absolute top-[-4px] w-[2px] h-[18px] rounded-full bg-[#F59E0B]"
        style={{ left: `${thrPct}%` }}
        title={`Threshold: ${(threshold * 100).toFixed(0)}%`}
      />
    </div>
  );
}

// ─── Symbol card ─────────────────────────────────────────────────────────────

function SymbolCard({ data, selected, onClick }: {
  data: SymbolSignals;
  selected: boolean;
  onClick: () => void;
}) {
  const latest = data.rows[0];
  const state = classifySymbol(latest);

  const regimeLabel = latest?.regime_prob != null
    ? latest.regime_prob >= 0.65 ? "TRENDING" : "RANGING"
    : null;
  const regimeColor = latest?.regime_prob != null
    ? latest.regime_prob >= 0.65 ? "#10B981" : "#EF4444"
    : "#6B7280";
  const htfIcon = latest?.htf_pass === true ? "✓" : latest?.htf_pass === false ? "✗" : "—";
  const htfColor = latest?.htf_pass === true ? "#10B981" : latest?.htf_pass === false ? "#EF4444" : "#6B7280";

  return (
    <button
      onClick={onClick}
      className="text-left w-full transition-all duration-150"
    >
      <div
        className={`rounded-xl p-4 space-y-3 transition-all duration-150 ${
          selected ? "ring-2 ring-[#3B82F6]/70" : "hover:border-opacity-60"
        }`}
        style={{
          backgroundColor: "#10151D",
          border: `1px solid ${state.color}40`,
          boxShadow: selected ? `0 0 0 0 transparent` : undefined,
        }}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-2">
          <span className="font-mono text-[#F3F4F6] text-sm font-semibold leading-tight">
            {data.symbol}
          </span>
          <div className="flex flex-col items-end gap-1 shrink-0">
            <span
              className="px-2 py-0.5 rounded-md text-[10px] font-bold tracking-wide"
              style={{ backgroundColor: `${state.color}20`, color: state.color }}
            >
              {state.label}
            </span>
            {latest && (
              <span className={`font-mono ${TABULAR} text-[9px] text-[#6B7280]`}>
                {fmtTime(latest.ts)}
              </span>
            )}
          </div>
        </div>

        {/* Prob + bar */}
        {data.loading ? (
          <Skeleton className="h-10" />
        ) : !latest ? (
          <div className="py-2">
            <p className="text-[11px] text-[#9CA3AF]">No ML score yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex items-end justify-between">
              <div>
                <div
                  className={`font-mono ${TABULAR} text-2xl font-bold leading-none`}
                  style={{ color: state.color }}
                >
                  {(latest.cal_prob * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">probability</div>
              </div>
              <div className="text-right">
                <div className={`font-mono ${TABULAR} text-sm font-semibold text-[#F59E0B]`}>
                  {(latest.threshold * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-[#6B7280] mt-0.5">threshold</div>
              </div>
            </div>
            <ProbBar prob={latest.cal_prob} threshold={latest.threshold} color={state.color} />
            {state.detail && (
              <p className="text-[10px] leading-tight" style={{ color: `${state.color}CC` }}>
                {state.detail}
              </p>
            )}
          </div>
        )}

        {/* Regime + HTF row */}
        {latest && (regimeLabel || latest.htf_pass !== undefined) && (
          <div className="flex items-center gap-3 pt-0.5">
            {regimeLabel && (
              <span className="flex items-center gap-1 text-[10px] font-mono">
                <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: regimeColor }} />
                <span style={{ color: regimeColor }}>{regimeLabel}</span>
              </span>
            )}
            {latest.htf_pass !== undefined && latest.htf_pass !== null && (
              <span className="text-[10px] font-mono" style={{ color: htfColor }}>
                HTF {htfIcon}
              </span>
            )}
          </div>
        )}

        {/* Recent history */}
        {!data.loading && data.rows.length > 1 && (
          <div className="border-t border-[#1E2530] pt-2.5 space-y-1.5">
            {data.rows.slice(1, 4).map((r, i) => {
              const dec = r.ml_decision.toUpperCase();
              const isActive = dec === "BUY" || dec === "LONG";
              const isBlocked = dec.startsWith("SKIP_") || dec === "FILTERED";
              const decColor = isActive ? "#10B981" : isBlocked ? "#F59E0B" : "#6B7280";
              const decLabel = dec === "NO_TRADE" ? "no trade" : dec === "LONG" ? "BUY" : dec.replace("SKIP_", "").toLowerCase();
              return (
                <div key={i} className="flex items-center justify-between gap-2">
                  <span className={`font-mono ${TABULAR} text-[10px] text-[#6B7280]`}>
                    {fmtTime(r.ts)}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className={`font-mono ${TABULAR} text-[10px]`} style={{ color: decColor }}>
                      {decLabel}
                    </span>
                    <span className={`font-mono ${TABULAR} text-[10px] text-[#9CA3AF]`}>
                      {(r.cal_prob * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </button>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function SignalsPage() {
  const [columns, setColumns] = useState<SymbolSignals[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useLocalStorage<string>("clawdbot:signals:symbol", "BTC/USDT");
  const [chartInterval, setChartInterval] = useLocalStorage<IntervalKey>("clawdbot:signals:interval", "15m");
  const [chartSize, setChartSize] = useLocalStorage<ChartSize>("clawdbot:signals:size", "M");
  const [activeFilter, setActiveFilter] = useLocalStorage<FilterKey>("clawdbot:signals:filter", "all");
  const ws = useWs();

  const latestForSelected = useMemo(() => {
    return columns.find((c) => c.symbol === selectedSymbol)?.rows[0];
  }, [columns, selectedSymbol]);

  const selectedState = useMemo(() => classifySymbol(latestForSelected), [latestForSelected]);

  const visibleColumns = useMemo(() => {
    if (activeFilter === "all") return columns;
    return columns.filter((col) => {
      const s = classifySymbol(col.rows[0]);
      if (activeFilter === "buying") return s.filterKey === "buying";
      if (activeFilter === "close") return s.filterKey === "close";
      if (activeFilter === "blocked") return s.filterKey === "blocked";
      if (activeFilter === "waiting") return s.filterKey === "waiting" || col.rows.length === 0;
      return true;
    });
  }, [columns, activeFilter]);

  // Live WS updates
  useEffect(() => {
    const updates = ws.signals;
    if (Object.keys(updates).length === 0) return;
    setColumns((prev) =>
      prev.map((col) => {
        const u = updates[col.symbol];
        if (!u) return col;
        const newRow: SignalRow = {
          ts: u.ts,
          cal_prob: u.cal_prob,
          threshold: u.threshold,
          ml_decision: (u.decision as string) ?? "no_trade",
        };
        if (col.rows[0]?.ts === newRow.ts) return col;
        return { ...col, rows: [newRow, ...col.rows].slice(0, 10) };
      })
    );
  }, [ws.signals]);

  // Initial load
  useEffect(() => {
    fetch(`${API}/api/symbols/config`)
      .then((r) => r.json())
      .then((cfg) => {
        const symbols: string[] = Array.isArray(cfg) && cfg.length > 0
          ? cfg.filter((c: any) => c.enabled !== false).map((c: any) => c.symbol)
          : SYMBOLS_FALLBACK;
        const initial = symbols.map((s) => ({ symbol: s, rows: [] as SignalRow[], loading: true, error: null as string | null }));
        setColumns(initial);
        if (symbols.length > 0 && !symbols.includes(selectedSymbol)) {
          setSelectedSymbol(symbols[0]);
        }
        symbols.forEach((sym, idx) => {
          const slug = sym.replace("/", "_");
          fetch(`${API}/api/signals/by-symbol/${slug}?limit=10`)
            .then((r) => r.json())
            .then((data) => {
              setColumns((prev) =>
                prev.map((c, i) =>
                  i === idx ? { ...c, rows: Array.isArray(data) ? data : [], loading: false } : c
                )
              );
            })
            .catch((e) => {
              setColumns((prev) =>
                prev.map((c, i) =>
                  i === idx ? { ...c, loading: false, error: String(e) } : c
                )
              );
            });
        });
      })
      .catch(() => {
        const initial = SYMBOLS_FALLBACK.map((s) => ({ symbol: s, rows: [] as SignalRow[], loading: false, error: "config_fetch_failed" }));
        setColumns(initial);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-5">
      {/* ── Page header ── */}
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Signals</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">
            Live ML probabilities — click a symbol to open its chart.
          </p>
          <details className="mt-1 text-xs">
            <summary className="cursor-pointer text-[#3B82F6] hover:text-[#60A5FA] inline-block select-none">
              How does the bot decide?
            </summary>
            <div className="mt-3">
              <TradeLifecycleExplainer />
            </div>
          </details>
        </div>

        {/* Chart controls */}
        <div className="flex items-center gap-2 flex-wrap">
          {latestForSelected && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#10151D] border border-[#374151]">
              <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wider">Prob</span>
              <span
                className={`font-mono ${TABULAR} text-sm font-bold`}
                style={{ color: selectedState.color }}
              >
                {(latestForSelected.cal_prob * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-[#6B7280]">/</span>
              <span className={`font-mono ${TABULAR} text-xs text-[#F59E0B]`}>
                {(latestForSelected.threshold * 100).toFixed(0)}% thr
              </span>
              <span
                className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                style={{ backgroundColor: `${selectedState.color}20`, color: selectedState.color }}
              >
                {selectedState.label}
              </span>
            </div>
          )}
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            aria-label="Chart symbol"
            className="bg-[#0a0e15] border border-[#374151] rounded-lg px-3 py-1.5 text-sm text-[#F3F4F6] font-mono focus:outline-none focus:border-[#3B82F6]"
          >
            {columns.map((c) => (
              <option key={c.symbol} value={c.symbol}>{c.symbol}</option>
            ))}
          </select>
          <div className="flex gap-0.5 bg-[#10151D] border border-[#374151] rounded-lg p-0.5">
            {INTERVALS.map((tf) => (
              <button
                key={tf}
                onClick={() => setChartInterval(tf)}
                className={`px-2 py-1 text-[11px] rounded transition-colors font-mono ${TABULAR} ${
                  chartInterval === tf
                    ? "bg-[#3B82F6]/20 text-[#3B82F6]"
                    : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
          <div className="flex gap-0.5 bg-[#10151D] border border-[#374151] rounded-lg p-0.5">
            {(["S", "M", "L", "XL"] as ChartSize[]).map((s) => (
              <button
                key={s}
                onClick={() => setChartSize(s)}
                className={`w-8 py-1 text-[11px] rounded transition-colors font-mono ${
                  chartSize === s
                    ? "bg-[#3B82F6]/20 text-[#3B82F6]"
                    : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── TradingView chart ── */}
      {selectedSymbol && (
        <div className="bg-[#10151D] border border-[#374151] rounded-xl overflow-hidden">
          <TradingViewChart
            key={`${selectedSymbol}-${chartInterval}-${chartSize}`}
            symbol={selectedSymbol}
            interval={chartInterval}
            height={SIZE_TO_CSS[chartSize]}
          />
        </div>
      )}

      {/* ── Symbol grid ── */}
      {chartSize !== "XL" && (
        <div className="space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <h2 className="text-[11px] uppercase tracking-widest text-[#9CA3AF] font-semibold">
              Symbol decisions
            </h2>
            <StatsBar
              columns={columns}
              activeFilter={activeFilter}
              onFilter={setActiveFilter}
            />
          </div>

          {columns.length === 0 ? (
            <EmptyState icon="↯" title="No symbol configs available" />
          ) : visibleColumns.length === 0 ? (
            <div className="py-8 text-center text-sm text-[#9CA3AF]">
              No symbols match the "{activeFilter}" filter right now.
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {visibleColumns.map((col) => (
                <SymbolCard
                  key={col.symbol}
                  data={col}
                  selected={col.symbol === selectedSymbol}
                  onClick={() => setSelectedSymbol(col.symbol)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
