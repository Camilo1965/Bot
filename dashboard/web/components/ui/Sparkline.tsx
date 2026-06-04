interface Props {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillOpacity?: number;
  strokeWidth?: number;
}

export function Sparkline({
  data,
  width = 100,
  height = 28,
  color,
  fillOpacity = 0.15,
  strokeWidth = 1.5,
}: Props) {
  if (!data || data.length < 2) {
    return (
      <svg width={width} height={height} className="text-[#374151]">
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
      </svg>
    );
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const trend = data[data.length - 1] - data[0];
  const c = color ?? (trend >= 0 ? "#10B981" : "#EF4444");

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 2) - 1;
    return [x, y] as const;
  });
  const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const area = `${path} L${width},${height} L0,${height} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <path d={area} fill={c} opacity={fillOpacity} />
      <path d={path} fill="none" stroke={c} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={points[points.length - 1][0]} cy={points[points.length - 1][1]} r="2" fill={c} />
    </svg>
  );
}
