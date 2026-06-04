"use client";

import { useEffect, useRef, useState } from "react";
import { useAnimatedNumber } from "@/hooks/useAnimatedNumber";

interface Props {
  value: number | null | undefined;
  format: (n: number) => string;
  className?: string;
  duration?: number;
  pulseOnChange?: boolean;
}

export function AnimatedValue({ value, format, className = "", duration = 400, pulseOnChange = true }: Props) {
  const numeric = value ?? 0;
  const animated = useAnimatedNumber(numeric, duration);
  const [pulse, setPulse] = useState(false);
  const prevRef = useRef(numeric);

  useEffect(() => {
    if (!pulseOnChange) return;
    if (prevRef.current !== numeric) {
      prevRef.current = numeric;
      setPulse(true);
      const t = setTimeout(() => setPulse(false), 300);
      return () => clearTimeout(t);
    }
  }, [numeric, pulseOnChange]);

  if (value === null || value === undefined) return <span className={className}>—</span>;

  return (
    <span className={`${className} ${pulse ? "number-pulse" : ""}`}>{format(animated)}</span>
  );
}
