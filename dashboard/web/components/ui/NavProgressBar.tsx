"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

/**
 * Top-edge progress bar that animates on route changes.
 * Fires whenever the path changes — gives instant perceived feedback
 * even before the new page's data fetches start.
 */
export function NavProgressBar() {
  const path = usePathname();
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setVisible(true);
    setProgress(20);
    const t1 = setTimeout(() => setProgress(70), 80);
    const t2 = setTimeout(() => setProgress(95), 250);
    const t3 = setTimeout(() => {
      setProgress(100);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        setVisible(false);
        setProgress(0);
      }, 200);
    }, 500);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [path]);

  return (
    <div
      className={`fixed top-0 left-0 right-0 h-0.5 z-50 pointer-events-none transition-opacity duration-200 ${
        visible ? "opacity-100" : "opacity-0"
      }`}
    >
      <div
        className="h-full bg-gradient-to-r from-[#3B82F6] via-[#60A5FA] to-[#3B82F6] transition-all duration-200 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
