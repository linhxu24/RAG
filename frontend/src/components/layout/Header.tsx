import { useQuery } from "@tanstack/react-query";
import { CircleGauge, RefreshCw } from "lucide-react";
import { useLocation } from "react-router-dom";

import { getSystemHealth } from "../../api/observability";
import { StatusBadge } from "../common/StatusBadge";
import { navigationItemForPath } from "./navigation";

export function Header() {
  const location = useLocation();
  const current = navigationItemForPath(location.pathname);
  const CurrentIcon = current.icon;
  const health = useQuery({
    queryKey: ["system-health"],
    queryFn: getSystemHealth,
    refetchInterval: 30_000,
    retry: 1,
  });
  const data = health.data;
  return (
    <header className="fixed left-14 right-0 top-0 z-30 flex h-15 items-center justify-between border-b border-slate-200 bg-white/92 px-4 backdrop-blur-xl">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-teal-50 text-teal-700">
          <CurrentIcon size={16} />
        </span>
        <div>
          <div className="text-xs font-bold text-slate-800">{current.shortLabel}</div>
          <div className="hidden text-[10px] text-slate-400 sm:block">
            {current.description}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="mr-1 hidden items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 lg:flex">
          <CircleGauge size={12} />
          Health
        </span>
        <StatusBadge status={data?.postgresql?.status ?? "unknown"} compact />
        <span className="hidden md:inline-flex">
          <StatusBadge status={data?.pgvector?.status ?? "unknown"} compact />
        </span>
        <span className="hidden lg:inline-flex">
          <StatusBadge status={data?.ollama?.status ?? "unknown"} compact />
        </span>
        <button
          aria-label="Refresh system health"
          className="ml-1 rounded-lg border border-slate-200 p-1.5 text-slate-500 hover:bg-slate-50"
          onClick={() => void health.refetch()}
        >
          <RefreshCw size={14} className={health.isFetching ? "animate-spin" : ""} />
        </button>
      </div>
    </header>
  );
}
