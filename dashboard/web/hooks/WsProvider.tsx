"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = `${API.replace("http", "ws")}/ws/stream`;

export type WsMessage =
  | { type: "hello"; connected_clients: number }
  | { type: "ping" }
  | { type: "equity.tick"; equity: number; balance: number; daily_pnl_pct: number; open_positions_count: number; kill_switch_active?: boolean; ts: string }
  | { type: "positions.snapshot"; positions: any[] }
  | { type: "signal.update"; symbol: string; cal_prob: number; threshold: number; raw_prob?: number; decision?: string; ts: string }
  | { type: "position.opened"; trade_id: string; symbol: string; ts: string }
  | { type: "position.closed"; trade_id: string; pnl_usd: number; reason: string; ts: string }
  | { type: "alert.new"; id?: string; severity?: string; message?: string; ts: string }
  | { type: "kill_switch.changed"; active: boolean; reason?: string; ts: string }
  | { type: "event"; [k: string]: any };

type Listener = (msg: WsMessage) => void;
type ConnState = "connecting" | "open" | "closed";

interface WsContextValue {
  state: ConnState;
  lastEquityTick: Extract<WsMessage, { type: "equity.tick" }> | null;
  lastPositionsSnapshot: any[] | null;
  signals: Record<string, Extract<WsMessage, { type: "signal.update" }>>;
  subscribe: (listener: Listener) => () => void;
}

const Ctx = createContext<WsContextValue | null>(null);

export function useWs(): WsContextValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useWs must be used inside <WsProvider>");
  return v;
}

/**
 * Convenience hook: subscribe to a specific event type, get latest message.
 */
export function useWsEvent<T extends WsMessage["type"]>(
  type: T,
  handler: (msg: Extract<WsMessage, { type: T }>) => void
) {
  const { subscribe } = useWs();
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  useEffect(() => {
    return subscribe((msg) => {
      if (msg.type === type) handlerRef.current(msg as Extract<WsMessage, { type: T }>);
    });
  }, [subscribe, type]);
}

export function WsProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ConnState>("connecting");
  const [lastEquityTick, setLastEquityTick] = useState<Extract<WsMessage, { type: "equity.tick" }> | null>(null);
  const [lastPositionsSnapshot, setLastPositionsSnapshot] = useState<any[] | null>(null);
  const [signals, setSignals] = useState<Record<string, Extract<WsMessage, { type: "signal.update" }>>>({});
  const listenersRef = useRef<Set<Listener>>(new Set());
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef<number>(0);

  const subscribe = useCallback((listener: Listener) => {
    listenersRef.current.add(listener);
    return () => { listenersRef.current.delete(listener); };
  }, []);

  const dispatch = useCallback((msg: WsMessage) => {
    if (msg.type === "equity.tick") setLastEquityTick(msg);
    else if (msg.type === "positions.snapshot") setLastPositionsSnapshot(msg.positions ?? []);
    else if (msg.type === "signal.update") {
      setSignals((prev) => ({ ...prev, [msg.symbol]: msg }));
    }
    listenersRef.current.forEach((l) => {
      try { l(msg); } catch {}
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      setState("connecting");
      let ws: WebSocket;
      try {
        ws = new WebSocket(WS_URL);
      } catch {
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setState("open");
        reconnectAttempts.current = 0;
      };
      ws.onclose = () => {
        if (cancelled) return;
        setState("closed");
        scheduleReconnect();
      };
      ws.onerror = () => {
        try { ws.close(); } catch {}
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data) as WsMessage;
          dispatch(msg);
        } catch {}
      };
    }

    function scheduleReconnect() {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      const attempt = ++reconnectAttempts.current;
      const delay = Math.min(30000, 1000 * Math.pow(1.5, attempt));
      reconnectTimer.current = setTimeout(connect, delay);
    }

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      try { wsRef.current?.close(); } catch {}
    };
  }, [dispatch]);

  const value = useMemo<WsContextValue>(() => ({
    state, lastEquityTick, lastPositionsSnapshot, signals, subscribe,
  }), [state, lastEquityTick, lastPositionsSnapshot, signals, subscribe]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
