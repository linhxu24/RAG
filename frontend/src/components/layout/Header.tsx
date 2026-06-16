import { useQuery } from "@tanstack/react-query";
import { CircleGauge, RefreshCw } from "lucide-react";

import { getSystemHealth } from "../../api/observability";
import { StatusBadge } from "../common/StatusBadge";

export function Header() {
  const health = useQuery({
    queryKey: ["system-health"],
    queryFn: getSystemHealth,
    refetchInterval: 30_000,
    retry: 1,
  });
  const data = health.data;
  return (
    <header className="fixed left-[252px] right-0 top-0 z-30 flex h-[72px] items-center justify-between border-b border-slate-200 bg-white/90 px-6 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <CircleGauge size={20} className="text-teal-600" />
        <div>
          <div className="text-sm font-bold text-slate-800">RAG Operations Workspace</div>
          <div className="text-xs text-slate-400">Testing, evaluation and pipeline diagnostics</div>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="mr-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">
          System health
        </span>
        <StatusBadge status={data?.postgresql?.status ?? "unknown"} compact />
        <StatusBadge status={data?.pgvector?.status ?? "unknown"} compact />
        <StatusBadge status={data?.ollama?.status ?? "unknown"} compact />
        <button
          aria-label="Refresh system health"
          className="ml-1 rounded-lg border border-slate-200 p-2 text-slate-500 hover:bg-slate-50"
          onClick={() => void health.refetch()}
        >
          <RefreshCw size={14} className={health.isFetching ? "animate-spin" : ""} />
        </button>
      </div>
    </header>
  );
}
