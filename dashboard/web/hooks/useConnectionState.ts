"use client";

import { useEffect, useState } from "react";

export type ConnState = "connecting" | "live" | "stale" | "down";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useConnectionState(pollMs = 10000) {
  const [state, setState] = useState<ConnState>("connecting");
  const [lastOk, setLastOk] = useState<number | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function ping() {
      const start = performance.now();
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 5000);
        const res = await fetch(`${API}/`, { signal: ctrl.signal, cache: "no-store" });
        clearTimeout(t);
        if (cancelled) return;
        if (res.ok) {
          setLatencyMs(Math.round(performance.now() - start));
          setLastOk(Date.now());
          setState("live");
          return;
        }
        setState("stale");
      } catch {
        if (cancelled) return;
        setLatencyMs(null);
        setState((prev) => (prev === "live" ? "stale" : "down"));
      }
    }

    ping();
    const id = setInterval(ping, pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [pollMs]);

  return { state, lastOk, latencyMs };
}
