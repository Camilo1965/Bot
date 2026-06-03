"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ModelMeta {
  trained_at: string;
  metrics: {
    auc?: number;
    auc_long?: number;
    auc_short?: number;
    [key: string]: number | undefined;
  };
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
    <span
      className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
        active
          ? "bg-[#10B981]/20 text-[#10B981]"
          : "bg-[#374151] text-[#9CA3AF]"
      }`}
    >
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
      <div className="text-[#9CA3AF] font-semibold uppercase tracking-wider text-[10px]">
        {label}
      </div>
      {auc !== undefined && (
        <div className="flex gap-2">
          <span className="text-[#9CA3AF]">AUC</span>
          <span
            className={`font-mono ${
              auc >= 0.6 ? "text-[#10B981]" : auc >= 0.5 ? "text-[#F3F4F6]" : "text-[#EF4444]"
            }`}
          >
            {auc.toFixed(4)}
          </span>
        </div>
      )}
      {meta.trained_at && (
        <div className="flex gap-2">
          <span className="text-[#9CA3AF]">Trained</span>
          <span className="font-mono text-[#F3F4F6]">
            {new Date(meta.trained_at).toLocaleDateString()}
          </span>
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

export default function ModelsPage() {
  const [models, setModels] = useState<SymbolModels[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/api/models`)
      .then((r) => r.json())
      .then((data) => {
        setModels(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Models</h1>
        <div className="relative group">
          <button
            disabled
            className="px-3 py-1.5 rounded bg-[#374151] text-[#9CA3AF] text-sm cursor-not-allowed"
          >
            Retrain All
          </button>
          <div className="absolute right-0 top-full mt-1 hidden group-hover:block bg-[#374151] text-[#9CA3AF] text-xs rounded px-2 py-1 whitespace-nowrap z-10">
            Run retrain_model.py manually
          </div>
        </div>
      </div>

      {loading && (
        <div className="text-[#9CA3AF] text-sm py-8 text-center">Loading models…</div>
      )}
      {error && (
        <div className="text-[#EF4444] text-sm py-4 text-center">Failed to load: {error}</div>
      )}

      {!loading && !error && models.length === 0 && (
        <div className="text-[#9CA3AF] text-sm py-8 text-center">No models found.</div>
      )}

      {!loading && !error && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {models.map((m) => (
            <div
              key={m.symbol}
              className="bg-[#10151D] border border-[#374151] rounded-lg p-4 space-y-3"
            >
              {/* Symbol header */}
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

              {/* Retrain button */}
              <div className="relative group border-t border-[#374151] pt-3">
                <button
                  disabled
                  className="w-full px-3 py-1.5 rounded bg-[#374151] text-[#9CA3AF] text-xs cursor-not-allowed"
                >
                  Retrain
                </button>
                <div className="absolute left-0 bottom-full mb-1 hidden group-hover:block bg-[#374151] text-[#9CA3AF] text-[10px] rounded px-2 py-1 whitespace-nowrap z-10">
                  Run retrain_model.py manually
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
