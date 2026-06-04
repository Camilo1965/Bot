"use client";

import { useEffect, useState } from "react";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { useToast } from "@/components/ui/Toast";
import { WebPushPrompt, pushNotify } from "@/components/WebPushPrompt";
import { fmtTimeAgo, severityColor, TABULAR } from "@/lib/format";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Severity = "info" | "warn" | "critical";

interface Alert {
  id: string;
  ts: string;
  source: string;
  message: string;
  severity: Severity;
  acked: boolean;
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | Severity>("all");
  const toast = useToast();

  useEffect(() => {
    fetch(`${API}/api/alerts?status=unread`)
      .then((r) => r.json())
      .then((data) => {
        const arr = Array.isArray(data) ? data : [];
        setAlerts(arr);
        setLoading(false);
        arr.slice(0, 3).forEach((a) => pushNotify(`ClawdBot · ${a.severity}`, a.message, a.id));
      })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, []);

  async function ackOne(id: string) {
    try {
      await fetch(`${API}/api/alerts/${id}/ack`, { method: "POST" });
      setAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch {
      toast.push({ type: "error", title: "Ack failed" });
    }
  }

  async function ackAll() {
    const ids = alerts.map((a) => a.id);
    try {
      await Promise.all(ids.map((id) => fetch(`${API}/api/alerts/${id}/ack`, { method: "POST" })));
      setAlerts([]);
      toast.push({ type: "success", title: `Acked ${ids.length} alert${ids.length === 1 ? "" : "s"}` });
    } catch {
      toast.push({ type: "error", title: "Bulk ack failed" });
    }
  }

  const filtered = alerts.filter((a) => filter === "all" || a.severity === filter);
  const counts = {
    critical: alerts.filter((a) => a.severity === "critical").length,
    warn: alerts.filter((a) => a.severity === "warn").length,
    info: alerts.filter((a) => a.severity === "info").length,
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[#F3F4F6]">Alerts</h1>
          <p className="text-xs text-[#9CA3AF] mt-0.5">
            {alerts.length} unread · {counts.critical} critical · {counts.warn} warn · {counts.info} info
          </p>
        </div>
        <div className="flex items-center gap-2">
          <WebPushPrompt />
          {alerts.length > 0 && (
            <button
              onClick={ackAll}
              className="px-3 py-1.5 rounded border border-[#374151] bg-[#0a0e15] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] hover:border-[#4B5563] transition-colors"
            >
              Ack All ({alerts.length})
            </button>
          )}
        </div>
      </div>

      {/* Filter pills */}
      <div className="flex gap-1 flex-wrap">
        {(["all", "critical", "warn", "info"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              filter === f
                ? "bg-[#3B82F6]/20 text-[#3B82F6] border border-[#3B82F6]/30"
                : "bg-[#10151D] border border-[#374151] text-[#9CA3AF] hover:text-[#F3F4F6]"
            }`}
          >
            {f.toUpperCase()}{f !== "all" && counts[f] > 0 ? ` (${counts[f]})` : ""}
          </button>
        ))}
      </div>

      {error && <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>}

      {loading && (
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-14" />)}
        </div>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="bg-[#10151D] border border-[#374151] rounded-lg">
          <EmptyState
            icon="✓"
            title={alerts.length === 0 ? "No unread alerts" : "No alerts match this filter"}
            body={alerts.length === 0 ? "All clear — bot is silent." : `Switch filter to see ${alerts.length} alert${alerts.length === 1 ? "" : "s"}.`}
          />
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="bg-[#10151D] border border-[#374151] rounded-lg divide-y divide-[#374151]/50">
          {filtered.map((a) => (
            <div key={a.id} className="px-4 py-3 hover:bg-white/5 transition-colors group">
              <div className="flex items-start gap-3">
                <span className={`mt-0.5 px-2 py-0.5 rounded text-[10px] font-semibold shrink-0 border ${severityColor(a.severity)}`}>
                  {a.severity.toUpperCase()}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-[#F3F4F6] break-words">{a.message}</div>
                  <div className={`flex gap-3 mt-1 text-[10px] font-mono ${TABULAR} text-[#9CA3AF]`}>
                    <span>{fmtTimeAgo(a.ts)}</span>
                    <span>· {a.source}</span>
                  </div>
                </div>
                <button
                  onClick={() => ackOne(a.id)}
                  className="px-2.5 py-1 rounded text-[10px] font-semibold bg-[#374151] text-[#9CA3AF] hover:bg-[#10B981]/20 hover:text-[#10B981] transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                >
                  ACK
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
