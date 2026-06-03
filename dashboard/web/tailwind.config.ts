import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Design tokens from ClawdBot spec
        bg: {
          base: "#0A0E14",
          surface: "#10151D",
          elevated: "#1E2530",
        },
        border: {
          default: "#374151",
        },
        text: {
          primary: "#F3F4F6",
          secondary: "#9CA3AF",
        },
        accent: {
          blue: "#3B82F6",
          green: "#10B981",
          red: "#EF4444",
          yellow: "#F59E0B",
          purple: "#8B5CF6",
          critical: "#DC2626",
        },
        pnl: {
          positive: "#10B981",
          negative: "#EF4444",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Courier New", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        card: "12px",
      },
    },
  },
  plugins: [],
};

export default config;
