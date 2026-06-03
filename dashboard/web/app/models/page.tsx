"use client";

import { useEffect, useState } from "react";

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

interface Toast { id: number; msg: string; ok: boolean }

function Badge({ active, label }: { active: boolean; label: string }) {
  return (
    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${active ? "bg-[#10B981]/20 text-[#10B981]" : "bg-[#374151] text-[#9CA3AF]"}`}>
      {label}
    </span>
  );
}

function MetaBlock({ label, meta }: { label: string; meta?: ModelMeta }) {
  if (!meta) return null;
  const auc = meta.metrics?.auc ?? meta.metrics?.auc_long ?? meta.metrics?.auc_short;
  const featureCount = meta.feature_count ?? meta.features?.length;
  return (
    <div className="text-[11px] space-y-0.5">
      <div className="text-[#9CA3AF] font-semibold uppercase tracking-wider text-[10px]">{label}</div>
      {auc !== undefined && (
        <div className="flex gap-2">
          <span className="text-[#9CA3AF]">AUC</span>
          <span className={`font-mono ${auc >= 0.6 ? "text-[#10B981]" : auc >= 0.5 ? "text-[#F3F4F6]" : "text-[#EF4444]"}`}>
            {auc.toFixed(4)}
          </span>
        </div>
      )}
      {meta.trained_at && (
        <div className="flex gap-2">
          <span className="text-[#9CA3AF]">Trained</span>
          <span className="font-mono text-[#F3F4F6]">{new Date(meta.trained_at).toLocaleDateString()}</span>
        </div>
      )}
      {featureCount !== undefined && (
        <div className="flex gap-2">
          <span className="text-[#9CA3AF]">Features</span>
          <span className="font-mono text-[#F3F4F6]">{featureCount}</span>
        </div>
      )}
    </div>
  );
}

function RetrainButton({ symbol, onToast }: { symbol: string; onToast: (msg: string, ok: boolean) => void }) {
  const [state, setState] = useState<"idle" | "running" | "done" | "error">("idle");
  const sym = symbol.replace("/", "_");

  async function handleRetrain() {
    setState("running");
    try {
      const res = await fetch(`${API}/api/symbols/${sym}/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days: 180, vibe: true, side: "both" }),
      });
      if (!res.ok || !res.body) { setState("error"); onToast(`Retrain ${symbol} failed`, false); return; }
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
            if (ev.type === "done") exitCode = ev.exit_code ?? 0;
          } catch {}
        }
      }
      setState(exitCode === 0 ? "done" : "error");
      onToast(`Retrain ${symbol} ${exitCode === 0 ? "completed" : "finished with warnings"}`, exitCode === 0);
    } catch (e) {
      setState("error");
      onToast(`Retrain ${symbol} error: ${e}`, false);
    }
  }

  const label = state === "running" ? "Retraining…" : state === "done" ? "Done" : state === "error" ? "Error" : "Retrain";
  const cls =
    state === "running" ? "bg-[#F59E0B]/20 text-[#F59E0B] cursor-wait" :
    state === "done" ? "bg-[#10B981]/20 text-[#10B981]" :
    state === "error" ? "bg-[#EF4444]/20 text-[#EF4444]" :
    "bg-[#374151] text-[#9CA3AF] hover:bg-[#4B5563] cursor-pointer";

  return (
    <button onClick={handleRetrain} disabled={state === "running"}
      className={`w-full px-3 py-1.5 rounded text-xs font-medium transition-colors ${cls}`}>
      {label}
    </button>
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState<SymbolModels[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    fetch(`${API}/api/models`)
      .then((r) => r.json())
      .then((data) => { setModels(Array.isArray(data) ? data : []); setLoading(false); })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  function addToast(msg: string, ok: boolean) {
    const id = Date.now();
    setToasts((t) => [...t, { id, msg, ok }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }

  return (
    <div className="space-y-6">
      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map((t) => (
          <div key={t.id} className={`px-4 py-2 rounded text-sm shadow-lg font-medium ${t.ok ? "bg-[#10B981] text-white" : "bg-[#EF4444] text-white"}`}>
            {t.msg}
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Models</h1>
      </div>

      {loading && <div className="text-[#9CA3AF] text-sm py-8 text-center">Loading models…</div>}
      {error && <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>}
      {!loading && !error && models.length === 0 && (
        <div className="text-[#9CA3AF] text-sm py-8 text-center">No models found.</div>
      )}

      {!loading && !error && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {models.map((m) => (
            <div key={m.symbol} className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-[#F3F4F6] font-mono">{m.symbol}</span>
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
                <RetrainButton symbol={m.symbol} onToast={addToast} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
