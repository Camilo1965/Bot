"use client";

import { useMemo } from "react";
import { fmtMoney, fmtPct, pnlColor, TABULAR } from "@/lib/format";

export interface LivePosition {
  trade_id: string;
  symbol: string;
  side: string;
  entry_price: number;
  current_price: number;
  sl_price: number;
  tp_price: number;
  pnl_usd: number;
  pnl_pct: number;
  size_usd?: number;
  ml_confidence?: number;
  peak_price?: number;
  current_sl?: number;
  trailing_active?: boolean;
  opened_at?: string;
  timeframe?: string;
}

interface Props {
  position: LivePosition;
  priceHistory: number[];
  onClickClose?: () => void;
  onClickChart?: () => void;
}

const CHART_W = 320;
const CHART_H = 110;
const PADDING_Y = 6;

function buildPath(values: number[], yMin: number, yMax: number): string {
  if (values.length < 2) return "";
  const yRange = Math.max(1e-9, yMax - yMin);
  const xStep = CHART_W / Math.max(1, values.length - 1);
  return values
    .map((v, i) => {
      const x = i * xStep;
      const y = CHART_H - PADDING_Y - ((v - yMin) / yRange) * (CHART_H - 2 * PADDING_Y);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export function LivePositionCard({ position, priceHistory, onClickClose, onClickChart }: Props) {
  const isLong = position.side?.toLowerCase() === "long";
  const sideColor = isLong ? "#10B981" : "#EF4444";
  const lineColor = position.pnl_usd >= 0 ? "#10B981" : "#EF4444";

  const series = useMemo(() => {
    const buf = priceHistory.length > 0 ? priceHistory : [position.entry_price, position.current_price];
    const yMin = Math.min(position.sl_price, position.tp_price, position.entry_price, ...buf);
    const yMax = Math.max(position.sl_price, position.tp_price, position.entry_price, ...buf);
    const pad = (yMax - yMin) * 0.05 || Math.abs(yMin) * 0.001 || 0.0001;
    return {
      path: buildPath(buf, yMin - pad, yMax + pad),
      yMin: yMin - pad,
      yMax: yMax + pad,
      yRange: Math.max(1e-9, yMax - yMin + 2 * pad),
    };
  }, [priceHistory, position.entry_price, position.current_price, position.sl_price, position.tp_price]);

  function yOf(v: number): number {
    return CHART_H - PADDING_Y - ((v - series.yMin) / series.yRange) * (CHART_H - 2 * PADDING_Y);
  }

  const yEntry = yOf(position.entry_price);
  const ySl = yOf(position.sl_price);
  const yTp = yOf(position.tp_price);
  const yNow = yOf(position.current_price);

  return (
    <div className="bg-[#0a0e15] border border-[#374151] rounded-lg p-3 space-y-2 hover:border-[#4B5563] transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-[#F3F4F6] text-sm font-semibold">{position.symbol}</span>
          <span
            className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
            style={{ backgroundColor: `${sideColor}22`, color: sideColor }}
          >
            {position.side?.toUpperCase()}
          </span>
          {position.trailing_active && (
            <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold bg-[#3B82F6]/15 text-[#3B82F6]">TRAIL</span>
          )}
        </div>
        <div className="text-right">
          <div className={`font-mono ${TABULAR} text-sm ${pnlColor(position.pnl_usd)}`}>{fmtMoney(position.pnl_usd)}</div>
          <div className={`font-mono ${TABULAR} text-[10px] ${pnlColor(position.pnl_pct)} opacity-80`}>{fmtPct(position.pnl_pct)}</div>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        width="100%"
        height={CHART_H}
        preserveAspectRatio="none"
        className="block"
      >
        <line x1="0" y1={ySl} x2={CHART_W} y2={ySl} stroke="#EF4444" strokeOpacity="0.6" strokeWidth="0.7" strokeDasharray="3 3" />
        <line x1="0" y1={yEntry} x2={CHART_W} y2={yEntry} stroke="#9CA3AF" strokeOpacity="0.6" strokeWidth="0.7" strokeDasharray="2 4" />
        <line x1="0" y1={yTp} x2={CHART_W} y2={yTp} stroke="#10B981" strokeOpacity="0.6" strokeWidth="0.7" strokeDasharray="3 3" />
        {series.path && (
          <path d={series.path} fill="none" stroke={lineColor} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
        )}
        <circle cx={CHART_W - 1} cy={yNow} r="3" fill={lineColor} />
        <text x="2" y={ySl - 2} fill="#EF4444" opacity="0.7" fontSize="9" fontFamily="ui-monospace, monospace">SL {position.sl_price.toFixed(4)}</text>
        <text x="2" y={yEntry - 2} fill="#9CA3AF" opacity="0.7" fontSize="9" fontFamily="ui-monospace, monospace">E {position.entry_price.toFixed(4)}</text>
        <text x="2" y={yTp - 2} fill="#10B981" opacity="0.7" fontSize="9" fontFamily="ui-monospace, monospace">TP {position.tp_price.toFixed(4)}</text>
      </svg>

      <div className={`flex justify-between items-center text-[10px] font-mono ${TABULAR} text-[#9CA3AF]`}>
        <span>now <span className="text-[#F3F4F6]">{position.current_price.toFixed(4)}</span></span>
        {position.ml_confidence !== undefined && (
          <span>ML <span className="text-[#3B82F6]">{(position.ml_confidence * 100).toFixed(1)}%</span></span>
        )}
        {position.timeframe && <span className="uppercase">{position.timeframe}</span>}
      </div>

      {(onClickChart || onClickClose) && (
        <div className="flex gap-1 pt-1">
          {onClickChart && (
            <button
              onClick={onClickChart}
              className="flex-1 px-2 py-1 rounded text-[10px] font-semibold bg-[#374151] text-[#9CA3AF] hover:bg-[#3B82F6]/15 hover:text-[#3B82F6] transition-colors"
            >
              Chart →
            </button>
          )}
          {onClickClose && (
            <button
              onClick={onClickClose}
              className="flex-1 px-2 py-1 rounded text-[10px] font-semibold bg-[#374151] text-[#9CA3AF] hover:bg-[#EF4444]/15 hover:text-[#EF4444] transition-colors"
            >
              Close
            </button>
          )}
        </div>
      )}
    </div>
  );
}
