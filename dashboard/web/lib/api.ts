/**
 * API base URL — always same origin as the frontend (Next.js proxies
 * /api/* and /ws/* to FastAPI on port 8000 via next.config.ts rewrites).
 * Works from localhost, LAN IP, Tailscale, or any public tunnel without
 * rebuilding or changing config.
 */
export const API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.host}`
    : "http://localhost:3000";
