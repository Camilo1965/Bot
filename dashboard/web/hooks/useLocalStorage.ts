"use client";

import { useEffect, useState } from "react";

/**
 * useState mirrored to localStorage. Hydrates from storage on mount
 * to avoid SSR mismatch. JSON-encoded, supports any serialisable value.
 */
export function useLocalStorage<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(initial);

  useEffect(() => {
    try {
      const raw = typeof window !== "undefined" ? localStorage.getItem(key) : null;
      if (raw !== null) setValue(JSON.parse(raw) as T);
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    try {
      if (typeof window !== "undefined") localStorage.setItem(key, JSON.stringify(value));
    } catch {}
  }, [key, value]);

  return [value, setValue];
}
