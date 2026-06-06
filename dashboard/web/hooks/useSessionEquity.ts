"use client";

import { useEffect, useState } from "react";
import { useWs } from "@/hooks/WsProvider";

export interface SessionPoint {
  ts: string;
  equity: number;
  balance: number;
  pnl: number;
}

const MAX_POINTS = 600;

export function useSessionEquity(): SessionPoint[] {
  const ws = useWs();
  const [points, setPoints] = useState<SessionPoint[]>([]);

  useEffect(() => {
    if (!ws.lastEquityTick) return;
    const t = ws.lastEquityTick;
    setPoints((prev) => {
      const next: SessionPoint = {
        ts: t.ts,
        equity: t.equity,
        balance: t.balance,
        pnl: t.equity - t.balance,
      };
      if (prev.length > 0 && prev[prev.length - 1].ts === next.ts) return prev;
      return [...prev.slice(-MAX_POINTS + 1), next];
    });
  }, [ws.lastEquityTick]);

  return points;
}

export function usePositionPriceBuffers(): Record<string, number[]> {
  const ws = useWs();
  const [buffers, setBuffers] = useState<Record<string, number[]>>({});

  useEffect(() => {
    const snap = ws.lastPositionsSnapshot;
    if (!Array.isArray(snap)) return;
    setBuffers((prev) => {
      const next = { ...prev };
      const seen = new Set<string>();
      for (const p of snap) {
        const id = String(p?.trade_id ?? p?.symbol ?? "");
        if (!id) continue;
        seen.add(id);
        const price = Number(p?.current_price);
        if (!isFinite(price) || price <= 0) continue;
        const buf = next[id] ? [...next[id]] : [];
        if (buf.length === 0 || buf[buf.length - 1] !== price) {
          buf.push(price);
          if (buf.length > 200) buf.shift();
        }
        next[id] = buf;
      }
      for (const k of Object.keys(next)) {
        if (!seen.has(k)) delete next[k];
      }
      return next;
    });
  }, [ws.lastPositionsSnapshot]);

  return buffers;
}
