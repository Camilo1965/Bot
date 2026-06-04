"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Smoothly animate a numeric value when it changes.
 * Useful for KPI counters — money/equity/etc — to feel "live."
 */
export function useAnimatedNumber(target: number, durationMs = 400): number {
  const [value, setValue] = useState(target);
  const startRef = useRef(target);
  const startTimeRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (target === value) return;
    startRef.current = value;
    startTimeRef.current = null;

    const tick = (now: number) => {
      if (startTimeRef.current === null) startTimeRef.current = now;
      const t = Math.min(1, (now - startTimeRef.current) / durationMs);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);
      const v = startRef.current + (target - startRef.current) * eased;
      setValue(v);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs]);

  return value;
}
