import type { ReactNode } from "react";

import { EmptyState } from "./States";

export interface Column<T> {
  key: string;
  label: string;
  width?: string;
  render: (row: T) => ReactNode;
}

export function DataTable<T>({
  rows,
  columns,
  rowKey,
  emptyTitle,
}: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  emptyTitle?: string;
}) {
  if (!rows.length) return <EmptyState title={emptyTitle} />;
  return (
    <div className="panel overflow-hidden">
      <div className="max-h-[620px] overflow-auto">
        <table className="w-full min-w-[900px] border-collapse text-left text-sm">
          <thead className="sticky top-0 z-10 bg-slate-50/95 backdrop-blur">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.key}
                  style={{ width: column.width }}
                  className="border-b border-slate-200 px-4 py-3 text-[11px] font-bold uppercase tracking-[0.08em] text-slate-500"
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={rowKey(row)}
                className="border-b border-slate-100 bg-white transition-colors last:border-0 hover:bg-slate-50/80"
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className="max-w-md px-4 py-3 align-top text-slate-650"
                  >
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
