"use client";

import { useEffect, useState } from "react";

/**
 * Bumps when the user presses `r` (handled in AppShell) or when
 * pageRefreshFn is called manually. Pages can put this in a useEffect
 * dep array to trigger a re-fetch.
 */
export function useRefreshKey(): [number, () => void] {
  const [key, setKey] = useState(0);

  useEffect(() => {
    function handler() { setKey((k) => k + 1); }
    window.addEventListener("clawdbot:refresh", handler);
    return () => window.removeEventListener("clawdbot:refresh", handler);
  }, []);

  return [key, () => setKey((k) => k + 1)];
}
