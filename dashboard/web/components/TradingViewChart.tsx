"use client";

import { useEffect, useRef } from "react";

interface Props {
  symbol: string;
  interval?: string;
  height?: number;
}

function tvSymbol(s: string): string {
  const base = s.replace("/", "").replace("USDT", "USDT");
  return `BINANCE:${base}.P`;
}

const tvInterval: Record<string, string> = {
  "5m": "5",
  "15m": "15",
  "30m": "30",
  "1h": "60",
  "4h": "240",
  "1d": "D",
};

export function TradingViewChart({ symbol, interval = "15m", height = 480 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const containerId = `tv_${symbol.replace("/", "_")}_${interval}`;

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = "";

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.async = true;
    script.type = "text/javascript";
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: tvSymbol(symbol),
      interval: tvInterval[interval] || "15",
      timezone: "Etc/UTC",
      theme: "dark",
      style: "1",
      locale: "en",
      backgroundColor: "#10151D",
      gridColor: "#374151",
      hide_top_toolbar: false,
      withdateranges: true,
      allow_symbol_change: false,
      hide_volume: false,
      support_host: "https://www.tradingview.com",
    });
    containerRef.current.appendChild(script);
  }, [symbol, interval]);

  return (
    <div
      id={containerId}
      ref={containerRef}
      className="tradingview-widget-container"
      style={{ height, width: "100%" }}
    />
  );
}
