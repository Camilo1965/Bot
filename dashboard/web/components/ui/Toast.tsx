"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

export type ToastType = "info" | "success" | "warn" | "error";

interface Toast {
  id: number;
  type: ToastType;
  title: string;
  body?: string;
  duration: number;
}

interface ToastContext {
  push: (t: Omit<Toast, "id" | "duration"> & { duration?: number }) => void;
}

const Ctx = createContext<ToastContext>({ push: () => {} });

export function useToast() {
  return useContext(Ctx);
}

export function ToastHost({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const push = useCallback((t: Omit<Toast, "id" | "duration"> & { duration?: number }) => {
    const id = Date.now() + Math.random();
    const duration = t.duration ?? 5000;
    setToasts((prev) => [...prev, { ...t, id, duration }]);
    setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), duration);
  }, []);

  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm w-[calc(100vw-2rem)] sm:w-96">
        {toasts.map((t) => (
          <ToastCard key={t.id} toast={t} onClose={() => setToasts((p) => p.filter((x) => x.id !== t.id))} />
        ))}
      </div>
    </Ctx.Provider>
  );
}

const STYLE: Record<ToastType, { bar: string; bg: string; icon: string }> = {
  info:    { bar: "bg-[#3B82F6]", bg: "bg-[#10151D]", icon: "ℹ" },
  success: { bar: "bg-[#10B981]", bg: "bg-[#10151D]", icon: "✓" },
  warn:    { bar: "bg-[#F59E0B]", bg: "bg-[#10151D]", icon: "!" },
  error:   { bar: "bg-[#EF4444]", bg: "bg-[#10151D]", icon: "✕" },
};

function ToastCard({ toast, onClose }: { toast: Toast; onClose: () => void }) {
  const [exiting, setExiting] = useState(false);
  const s = STYLE[toast.type];

  useEffect(() => {
    const t = setTimeout(() => setExiting(true), toast.duration - 200);
    return () => clearTimeout(t);
  }, [toast.duration]);

  return (
    <div
      className={`relative flex border border-[#374151] rounded-lg shadow-2xl overflow-hidden transition-all duration-200 ${
        exiting ? "opacity-0 translate-x-4" : "opacity-100 translate-x-0"
      } ${s.bg}`}
    >
      <div className={`w-1 ${s.bar}`} />
      <div className="flex-1 px-3 py-2.5 min-w-0">
        <div className="flex items-start gap-2">
          <span className={`text-sm font-bold ${s.bar.replace("bg-", "text-")} shrink-0`}>{s.icon}</span>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-[#F3F4F6] truncate">{toast.title}</div>
            {toast.body && <div className="text-xs text-[#9CA3AF] mt-0.5 break-words">{toast.body}</div>}
          </div>
          <button
            onClick={onClose}
            className="text-[#6B7280] hover:text-[#F3F4F6] text-xs leading-none shrink-0 ml-1"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}
