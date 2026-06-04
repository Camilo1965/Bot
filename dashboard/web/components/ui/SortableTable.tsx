"use client";

import { useMemo, useState } from "react";

export interface Column<T> {
  key: keyof T | string;
  label: string;
  align?: "left" | "right" | "center";
  sortable?: boolean;
  sortBy?: (row: T) => number | string;
  render?: (row: T) => React.ReactNode;
  className?: string;
  width?: string;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T, idx: number) => string;
  onRowClick?: (row: T) => void;
  emptyLabel?: string;
  initialSort?: { col: string; dir: "asc" | "desc" };
  stickyHeader?: boolean;
  compact?: boolean;
}

export function SortableTable<T>({
  rows, columns, rowKey, onRowClick, emptyLabel = "No data.", initialSort, stickyHeader = false, compact = false,
}: Props<T>) {
  const [sort, setSort] = useState<{ col: string; dir: "asc" | "desc" } | null>(initialSort ?? null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => String(c.key) === sort.col);
    if (!col) return rows;
    const get = col.sortBy ?? ((r: T) => (r as any)[col.key]);
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (va === vb) return 0;
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });
  }, [rows, sort, columns]);

  function toggleSort(key: string) {
    setSort((prev) => {
      if (!prev || prev.col !== key) return { col: key, dir: "desc" };
      if (prev.dir === "desc") return { col: key, dir: "asc" };
      return null;
    });
  }

  const cell = compact ? "py-1.5" : "py-2.5";

  if (rows.length === 0) {
    return (
      <div className="text-center text-[#9CA3AF] py-10 text-sm">{emptyLabel}</div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className={stickyHeader ? "sticky top-0 bg-[#10151D] z-10" : ""}>
          <tr className="text-[10px] uppercase tracking-widest text-[#9CA3AF] border-b border-[#374151]">
            {columns.map((c) => {
              const isActive = sort?.col === String(c.key);
              return (
                <th
                  key={String(c.key)}
                  className={`pb-2 pr-4 font-medium ${c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : ""} ${c.sortable ? "cursor-pointer select-none hover:text-[#F3F4F6]" : ""} ${c.className ?? ""}`}
                  style={c.width ? { width: c.width } : undefined}
                  onClick={c.sortable ? () => toggleSort(String(c.key)) : undefined}
                >
                  <span className={`inline-flex items-center gap-1 ${isActive ? "text-[#3B82F6]" : ""}`}>
                    {c.label}
                    {c.sortable && (
                      <span className="text-[8px] opacity-60">
                        {isActive ? (sort!.dir === "desc" ? "▼" : "▲") : "↕"}
                      </span>
                    )}
                  </span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr
              key={rowKey(row, i)}
              className={`border-b border-[#374151]/40 hover:bg-white/5 transition-colors ${onRowClick ? "cursor-pointer" : ""}`}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
            >
              {columns.map((c) => (
                <td
                  key={String(c.key)}
                  className={`${cell} pr-4 ${c.align === "right" ? "text-right" : c.align === "center" ? "text-center" : ""} ${c.className ?? ""}`}
                >
                  {c.render ? c.render(row) : String((row as any)[c.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
