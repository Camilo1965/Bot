/**
 * Runtime API base URL — derives from the browser's hostname so the dashboard
 * works from localhost, LAN IP, Tailscale IP, or any tunnel without rebuilding.
 *
 * FastAPI runs on port 8000; Next.js on 3000. Both are on the same host.
 */
export const API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://localhost:8000";
