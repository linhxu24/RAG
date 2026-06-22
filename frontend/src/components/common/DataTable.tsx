import { ChevronDown, ChevronUp, ChevronsUpDown, Search } from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";

import { EmptyState } from "./States";

export interface Column<T> {
  key: string;
  label: string;
  width?: string;
  render: (row: T) => ReactNode;
  sortValue?: (row: T) => unknown;
  searchValue?: (row: T) => unknown;
  sortable?: boolean;
}

export function DataTable<T>({
  rows,
  columns,
  rowKey,
  emptyTitle,
  defaultSort,
  pageSize = 25,
  searchable = true,
}: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  emptyTitle?: string;
  defaultSort?: { key: string; direction?: "asc" | "desc" };
  pageSize?: number;
  searchable?: boolean;
}) {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState<{
    key: string;
    direction: "asc" | "desc";
  } | null>(
    defaultSort
      ? { key: defaultSort.key, direction: defaultSort.direction || "asc" }
      : null,
  );
  const filteredRows = useMemo(() => {
    const normalized = search.trim().toLocaleLowerCase("vi");
    if (!normalized) return rows;
    return rows.filter((row) =>
      columns.some((column) => {
        const value = column.searchValue?.(row) ?? column.sortValue?.(row);
        return String(value ?? "").toLocaleLowerCase("vi").includes(normalized);
      }),
    );
  }, [columns, rows, search]);
  const sortedRows = useMemo(() => {
    if (!sort) return filteredRows;
    const column = columns.find((item) => item.key === sort.key);
    if (!column?.sortValue) return filteredRows;
    return [...filteredRows].sort((left, right) => {
      const leftValue = column.sortValue?.(left);
      const rightValue = column.sortValue?.(right);
      const comparison = compareValues(leftValue, rightValue);
      return sort.direction === "asc" ? comparison : -comparison;
    });
  }, [columns, filteredRows, sort]);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));
  const visibleRows = sortedRows.slice((page - 1) * pageSize, page * pageSize);

  useEffect(() => setPage(1), [search, pageSize]);
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

  if (!rows.length) return <EmptyState title={emptyTitle} />;
  return (
    <div className="panel overflow-hidden">
      {(searchable || sortedRows.length > pageSize) && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-3 py-2.5">
          {searchable ? (
            <label className="control flex max-w-sm items-center gap-2 px-3">
              <Search size={15} className="text-slate-400" />
              <input
                className="w-full bg-transparent py-2 text-xs outline-none"
                placeholder="Tìm trong bảng..."
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
          ) : (
            <span />
          )}
          <div className="text-xs text-slate-500">
            {sortedRows.length} / {rows.length} bản ghi
          </div>
        </div>
      )}
      <div className="max-h-[calc(100vh-210px)] min-h-48 overflow-auto">
        <table className="w-full min-w-[820px] border-collapse text-left text-xs">
          <thead className="sticky top-0 z-10 bg-slate-50/95 backdrop-blur">
            <tr>
              {columns.map((column) => {
                const canSort = column.sortable !== false && Boolean(column.sortValue);
                const active = sort?.key === column.key;
                const Icon = active
                  ? sort.direction === "asc"
                    ? ChevronUp
                    : ChevronDown
                  : ChevronsUpDown;
                return (
                  <th
                    key={column.key}
                    style={{ width: column.width }}
                    className="border-b border-slate-200 px-3 py-2.5 text-[10px] font-bold uppercase tracking-[0.08em] text-slate-500"
                  >
                    <button
                      type="button"
                      disabled={!canSort}
                      className="flex items-center gap-1.5 disabled:cursor-default"
                      onClick={() => {
                        if (!canSort) return;
                        setPage(1);
                        setSort((current) =>
                          current?.key === column.key
                            ? {
                                key: column.key,
                                direction: current.direction === "asc" ? "desc" : "asc",
                              }
                            : { key: column.key, direction: "asc" },
                        );
                      }}
                    >
                      {column.label}
                      {canSort && <Icon size={12} className={active ? "text-teal-600" : ""} />}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr
                key={rowKey(row)}
                className="border-b border-slate-100 bg-white transition-colors last:border-0 hover:bg-slate-50/80"
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className="max-w-md px-3 py-2.5 align-top text-slate-650"
                  >
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!sortedRows.length ? (
        <div className="border-t border-slate-100">
          <EmptyState title="Không có bản ghi phù hợp" />
        </div>
      ) : totalPages > 1 ? (
        <div className="flex items-center justify-between border-t border-slate-200 px-3 py-2.5 text-xs text-slate-500">
          <span>
            Trang {page}/{totalPages}
          </span>
          <div className="flex gap-2">
            <button
              className="secondary-button px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={page === 1}
              onClick={() => setPage((current) => Math.max(1, current - 1))}
            >
              Trước
            </button>
            <button
              className="secondary-button px-3 py-1.5 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={page === totalPages}
              onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
            >
              Sau
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function compareValues(left: unknown, right: unknown): number {
  if (left == null && right == null) return 0;
  if (left == null) return 1;
  if (right == null) return -1;
  if (typeof left === "number" && typeof right === "number") return left - right;
  if (typeof left === "boolean" && typeof right === "boolean") {
    return Number(left) - Number(right);
  }
  const leftDate = typeof left === "string" ? Date.parse(left) : Number.NaN;
  const rightDate = typeof right === "string" ? Date.parse(right) : Number.NaN;
  if (!Number.isNaN(leftDate) && !Number.isNaN(rightDate)) return leftDate - rightDate;
  return String(left).localeCompare(String(right), "vi", {
    numeric: true,
    sensitivity: "base",
  });
}
