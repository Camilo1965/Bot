"use client";

import { useMemo } from "react";

interface Props {
  symbol: string;
  interval?: string;
  height?: number | string;
}

function tvSymbol(s: string): string {
  const base = s.replace("/", "");
  return `BINANCE:${base}.P`;
}

const tvInterval: Record<string, string> = {
  "1m": "1",
  "5m": "5",
  "15m": "15",
  "30m": "30",
  "1h": "60",
  "4h": "240",
  "1d": "D",
};

export function TradingViewChart({ symbol, interval = "15m", height = 720 }: Props) {
  const src = useMemo(() => {
    const params = new URLSearchParams({
      symbol: tvSymbol(symbol),
      interval: tvInterval[interval] || "15",
      hidesidetoolbar: "0",
      hidetoptoolbar: "0",
      symboledit: "0",
      saveimage: "1",
      toolbarbg: "rgba(16,21,29,1)",
      studies: "[]",
      hideideas: "1",
      theme: "dark",
      style: "1",
      timezone: "Etc/UTC",
      withdateranges: "1",
      hidevolume: "0",
      utm_source: "clawdbot",
      utm_medium: "widget",
    });
    return `https://s.tradingview.com/widgetembed/?${params.toString()}`;
  }, [symbol, interval]);

  const cssHeight = typeof height === "number" ? `${height}px` : height;

  return (
    <div
      className="tradingview-widget-container relative w-full"
      style={{ height: cssHeight, minHeight: 320 }}
    >
      <iframe
        title={`TradingView ${symbol}`}
        src={src}
        allow="fullscreen"
        loading="lazy"
        style={{
          width: "100%",
          height: "100%",
          border: 0,
          display: "block",
          backgroundColor: "#10151D",
        }}
      />
    </div>
  );
}
