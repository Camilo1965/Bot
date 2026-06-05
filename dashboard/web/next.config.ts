import type { NextConfig } from "next";

const FASTAPI = process.env.FASTAPI_INTERNAL_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // Proxy all /api/* and /ws/* calls to FastAPI — single port exposed (3000).
      // WebSocket upgrades (/ws/stream) are also proxied by Next.js 15.
      { source: "/api/:path*", destination: `${FASTAPI}/api/:path*` },
      { source: "/ws/:path*", destination: `${FASTAPI}/ws/:path*` },
    ];
  },
};

export default nextConfig;
