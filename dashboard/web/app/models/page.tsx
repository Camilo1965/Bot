"use client";

import { useEffect, useState } from "react";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { fmtDate, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ModelMeta {
  trained_at: string;
  metrics: { auc?: number; auc_long?: number; auc_short?: number; [key: string]: number | undefined };
  feature_count?: number;
  features?: string[];
}

interface SymbolModels {
  symbol: string;
  has_long_model: boolean;
  has_short_model: boolean;
  long_meta?: ModelMeta;
  short_meta?: ModelMeta;
}

function Badge({ active, label }: { active: boolean; label: string }) {
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${active ? "bg-[#10B981]/15 text-[#10B981]" : "bg-[#374151] text-[#9CA3AF]"}`}>
      {label}
    </span>
  );
}

function AucBar({ auc }: { auc: number }) {
  const pct = Math.max(0, Math.min(100, (auc - 0.5) * 200)); // 0.5→0%, 1.0→100%
  const color = auc >= 0.65 ? "#10B981" : auc >= 0.55 ? "#3B82F6" : auc >= 0.50 ? "#F59E0B" : "#EF4444";
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <span className="text-[10px] text-[#9CA3AF] uppercase tracking-wider">AUC</span>
        <span className={`font-mono ${TABULAR} text-xs font-semibold`} style={{ color }}>{auc.toFixed(4)}</span>
      </div>
      <div className="h-1.5 bg-[#0a0e15] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function MetaBlock({ label, meta }: { label: string; meta?: ModelMeta }) {
  if (!meta) return null;
  const auc = meta.metrics?.auc ?? meta.metrics?.auc_long ?? meta.metrics?.auc_short;
  const featureCount = meta.feature_count ?? meta.features?.length;
  return (
    <div className="space-y-2">
      <div className="text-[10px] text-[#9CA3AF] font-semibold uppercase tracking-wider">{label}</div>
      {auc !== undefined && <AucBar auc={auc} />}
      <div className="flex justify-between text-[10px]">
        <span className="text-[#9CA3AF]">Trained</span>
        <span className={`font-mono ${TABULAR} text-[#F3F4F6]`}>{fmtDate(meta.trained_at)}</span>
      </div>
      {featureCount !== undefined && (
        <div className="flex justify-between text-[10px]">
          <span className="text-[#9CA3AF]">Features</span>
          <span className={`font-mono ${TABULAR} text-[#F3F4F6]`}>{featureCount}</span>
        </div>
      )}
    </div>
  );
}

function RetrainButton({ symbol, onDone }: { symbol: string; onDone: () => void }) {
  const [state, setState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [lines, setLines] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const sym = symbol.replace("/", "_");
  const toast = useToast();

  async function run() {
    setState("running");
    setLines([]);
    setShowLogs(true);
    try {
      const res = await fetch(`${API}/api/symbols/${sym}/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days: 180, vibe: true, side: "both" }),
      });
      if (!res.ok || !res.body) {
        setState("error");
        toast.push({ type: "error", title: `Retrain ${symbol} failed` });
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let exitCode = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.replace(/^data:\s*/, "").trim();
          if (!line) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.type === "log") setLines((l) => [...l.slice(-200), ev.line]);
            if (ev.type === "done") exitCode = ev.exit_code ?? 0;
          } catch {}
        }
      }
      const ok = exitCode === 0;
      setState(ok ? "done" : "error");
      toast.push({
        type: ok ? "success" : "warn",
        title: `Retrain ${symbol} ${ok ? "completed" : "finished with warnings"}`,
      });
      if (ok) onDone();
    } catch (e) {
      setState("error");
      toast.push({ type: "error", title: `Retrain ${symbol} error`, body: String(e) });
    }
  }

  const label = state === "running" ? "Retraining…" : state === "done" ? "✓ Done" : state === "error" ? "✕ Error" : "Retrain";
  const cls =
    state === "running" ? "bg-[#F59E0B]/15 text-[#F59E0B] cursor-wait" :
    state === "done"    ? "bg-[#10B981]/15 text-[#10B981]" :
    state === "error"   ? "bg-[#EF4444]/15 text-[#EF4444]" :
    "bg-[#374151] text-[#9CA3AF] hover:bg-[#4B5563] cursor-pointer";

  return (
    <div className="space-y-2">
      <button onClick={run} disabled={state === "running"}
        className={`w-full px-3 py-1.5 rounded text-xs font-medium transition-colors ${cls}`}>
        {label}
      </button>
      {showLogs && lines.length > 0 && (
        <details open className="text-[10px]">
          <summary className="cursor-pointer text-[#9CA3AF] hover:text-[#F3F4F6]">
            View training logs ({lines.length})
          </summary>
          <div className="mt-1 h-32 overflow-y-auto bg-[#0a0e15] border border-[#374151] rounded p-2 font-mono text-[#10B981] space-y-0.5">
            {lines.map((l, i) => <div key={i} className="leading-4 whitespace-pre-wrap break-all">{l}</div>)}
          </div>
        </details>
      )}
    </div>
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState<SymbolModels[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    fetch(`${API}/api/models`)
      .then((r) => r.json())
      .then((data) => { setModels(Array.isArray(data) ? data : []); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, [tick]);

  const summary = {
    total: models.length,
    withLong: models.filter((m) => m.has_long_model).length,
    withShort: models.filter((m) => m.has_short_model).length,
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Models</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">
            {summary.total} symbols · {summary.withLong} long · {summary.withShort} short
          </p>
        </div>
        <button
          onClick={() => setTick((t) => t + 1)}
          className="px-3 py-1.5 rounded border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] hover:border-[#4B5563] transition-colors"
        >
          ↻ Refresh
        </button>
      </div>

      {error && <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>}

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-56" />)}
        </div>
      ) : models.length === 0 ? (
        <EmptyState icon="▥" title="No models found"
          body="Train models with python scripts/retrain_model.py --all --side both" />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {models.map((m) => (
            <div key={m.symbol} className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3 hover:border-[#4B5563] transition-colors">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[#F3F4F6] font-semibold">{m.symbol}</span>
                <div className="flex gap-1">
                  <Badge active={m.has_long_model} label="LONG" />
                  <Badge active={m.has_short_model} label="SHORT" />
                </div>
              </div>
              <div className="border-t border-[#374151] pt-3 space-y-3">
                {m.has_long_model && <MetaBlock label="Long model" meta={m.long_meta} />}
                {m.has_short_model && <MetaBlock label="Short model" meta={m.short_meta} />}
                {!m.has_long_model && !m.has_short_model && (
                  <div className="text-[#9CA3AF] text-xs">No trained models on disk.</div>
                )}
              </div>
              <div className="border-t border-[#374151] pt-3">
                <RetrainButton symbol={m.symbol} onDone={() => setTick((t) => t + 1)} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
