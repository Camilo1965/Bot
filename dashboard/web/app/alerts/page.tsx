"use client";

import { useEffect, useState } from "react";
import { WebPushPrompt, pushNotify } from "@/components/WebPushPrompt";

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

function SeverityBadge({ severity }: { severity: Severity }) {
  const styles: Record<Severity, string> = {
    info: "bg-[#3B82F6]/20 text-[#3B82F6]",
    warn: "bg-yellow-500/20 text-yellow-400",
    critical: "bg-[#EF4444]/20 text-[#EF4444]",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${styles[severity]}`}>
      {severity.toUpperCase()}
    </span>
  );
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [acking, setAcking] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetch(`${API}/api/alerts?status=unread`)
      .then((r) => r.json())
      .then((data) => {
        const arr = Array.isArray(data) ? data : [];
        setAlerts(arr);
        setLoading(false);
        arr.slice(0, 3).forEach((a) => pushNotify(`ClawdBot · ${a.severity}`, a.message, a.id));
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  async function ackAlert(id: string) {
    setAcking((prev) => new Set(prev).add(id));
    try {
      await fetch(`${API}/api/alerts/${id}/ack`, { method: "POST" });
      setAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch (e) {
      console.error("Ack failed", e);
    } finally {
      setAcking((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  async function ackAll() {
    const ids = alerts.map((a) => a.id);
    ids.forEach((id) => setAcking((prev) => new Set(prev).add(id)));
    try {
      await Promise.all(ids.map((id) => fetch(`${API}/api/alerts/${id}/ack`, { method: "POST" })));
      setAlerts([]);
    } catch (e) {
      console.error("Bulk ack failed", e);
    } finally {
      setAcking(new Set());
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-xl font-semibold text-[#F3F4F6]">Alerts</h1>
        <div className="flex items-center gap-2">
          <WebPushPrompt />
          {alerts.length > 0 && (
            <button
              onClick={ackAll}
              className="px-3 py-1.5 rounded bg-[#374151] text-[#9CA3AF] text-sm hover:text-[#F3F4F6] transition-colors"
            >
              Ack All ({alerts.length})
            </button>
          )}
        </div>
      </div>

      {loading && (
        <div className="text-[#9CA3AF] text-sm text-center py-8">Loading alerts…</div>
      )}
      {error && (
        <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>
      )}

      {!loading && !error && alerts.length === 0 && (
        <div className="bg-[#10151D] border border-[#374151] rounded-lg p-10 flex flex-col items-center gap-2">
          <div className="text-[#10B981] text-2xl">✓</div>
          <div className="text-[#F3F4F6] text-sm font-medium">No unread alerts</div>
          <div className="text-[#9CA3AF] text-xs">All clear — no pending notifications.</div>
        </div>
      )}

      {!loading && !error && alerts.length > 0 && (
        <div className="bg-[#10151D] border border-[#374151] rounded-lg divide-y divide-[#374151]/50">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`flex items-start gap-4 px-4 py-3 hover:bg-white/5 transition-colors ${
                alert.severity === "critical" ? "border-l-2 border-[#EF4444]" : ""
              }`}
            >
              {/* Timestamp */}
              <div className="shrink-0 text-[11px] font-mono text-[#9CA3AF] mt-0.5 w-32">
                {new Date(alert.ts).toLocaleString([], {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>

              {/* Severity + source + message */}
              <div className="flex-1 min-w-0 space-y-0.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <SeverityBadge severity={alert.severity} />
                  <span className="text-[#9CA3AF] text-xs">{alert.source}</span>
                </div>
                <div className="text-sm text-[#F3F4F6]">{alert.message}</div>
              </div>

              {/* Ack button */}
              <button
                onClick={() => ackAlert(alert.id)}
                disabled={acking.has(alert.id)}
                className="shrink-0 px-3 py-1 rounded bg-[#374151] text-[#9CA3AF] text-xs hover:text-[#F3F4F6] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {acking.has(alert.id) ? "…" : "Ack"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
