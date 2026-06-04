"use client";

import { useEffect, useState } from "react";

/**
 * Returns true when the page is visible AND focused.
 * Used to pause polling when tab is in the background.
 */
export function usePagePresence(): boolean {
  const [active, setActive] = useState<boolean>(() => {
    if (typeof document === "undefined") return true;
    return !document.hidden;
  });

  useEffect(() => {
    function onChange() { setActive(!document.hidden); }
    document.addEventListener("visibilitychange", onChange);
    window.addEventListener("focus", () => setActive(true));
    window.addEventListener("blur", () => setActive(false));
    return () => {
      document.removeEventListener("visibilitychange", onChange);
    };
  }, []);

  return active;
}
