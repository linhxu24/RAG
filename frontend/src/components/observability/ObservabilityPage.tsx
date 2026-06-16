import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Bot,
  Database,
  Gauge,
  HardDrive,
  Layers3,
  Timer,
} from "lucide-react";

import {
  getObservabilityMetrics,
  getObservabilityDiagnostics,
  getRecentErrors,
  getSystemHealth,
  type RecentError,
} from "../../api/observability";
import { formatDate, formatLatency, formatPercent, truncate } from "../../utils/format";
import { DataTable, type Column } from "../common/DataTable";
import { MetricCard } from "../common/MetricCard";
import { ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

export function ObservabilityPage() {
  const health = useQuery({
    queryKey: ["observability-health"],
    queryFn: getSystemHealth,
    refetchInterval: 30_000,
  });
  const metrics = useQuery({
    queryKey: ["observability-metrics"],
    queryFn: getObservabilityMetrics,
  });
  const errors = useQuery({
    queryKey: ["observability-errors"],
    queryFn: getRecentErrors,
  });
  const diagnostics = useQuery({
    queryKey: ["observability-diagnostics"],
    queryFn: getObservabilityDiagnostics,
  });
  const data = metrics.data || {};
  return (
    <PageContainer
      title="Observability"
      description="System health và request metrics được đọc độc lập từ backend traces, không gọi live chatbot."
    >
      <h2 className="mb-3 text-sm font-bold text-slate-800">System health</h2>
      {health.isLoading ? (
        <LoadingState />
      ) : health.isError ? (
        <ErrorState error={health.error} onRetry={() => void health.refetch()} />
      ) : (
        <div className="grid grid-cols-6 gap-4">
          <HealthCard
            label="PostgreSQL"
            status={health.data?.postgresql?.status}
            value={truncate(String(health.data?.postgresql?.version || "Unknown"), 26)}
            icon={Database}
          />
          <HealthCard
            label="pgvector"
            status={health.data?.pgvector?.status}
            value="Vector search"
            icon={Layers3}
          />
          <HealthCard
            label="Ollama"
            status={health.data?.ollama?.status}
            value={`${health.data?.ollama?.models?.length || 0} models`}
            icon={Bot}
          />
          <HealthCard
            label="Embedding"
            status={health.data?.embedding_model?.status}
            value={String(health.data?.embedding_model?.model || "Unknown")}
            icon={Activity}
          />
          <HealthCard
            label="Reranker"
            status={health.data?.reranker?.status}
            value={String(health.data?.reranker?.model || "Unknown")}
            icon={Gauge}
          />
          <HealthCard
            label="Assets"
            status={health.data?.assets?.status}
            value={String(health.data?.assets?.directory || "Unknown")}
            icon={HardDrive}
          />
        </div>
      )}
      <h2 className="mb-3 mt-6 text-sm font-bold text-slate-800">Request metrics today</h2>
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total requests" value={data.total_requests ?? "—"} icon={Activity} />
        <MetricCard label="Success rate" value={formatPercent(data.success_rate)} icon={Gauge} />
        <MetricCard
          label="Error rate"
          value={formatPercent(data.error_rate)}
          icon={AlertTriangle}
          accent="rose"
        />
        <MetricCard
          label="Average latency"
          value={formatLatency(data.average_latency_ms)}
          icon={Timer}
          accent="amber"
        />
        <MetricCard label="p50 latency" value={formatLatency(data.p50_latency_ms)} icon={Timer} />
        <MetricCard label="p95 latency" value={formatLatency(data.p95_latency_ms)} icon={Timer} />
        <MetricCard label="p99 latency" value={formatLatency(data.p99_latency_ms)} icon={Timer} />
        <MetricCard
          label="Fallback rate"
          value={formatPercent(data.fallback_rate)}
          icon={Activity}
          accent="violet"
        />
      </div>
      <h2 className="mb-3 mt-6 text-sm font-bold text-slate-800">Diagnostic alerts</h2>
      <div className="grid grid-cols-2 gap-3">
        {(diagnostics.data?.alerts || []).map((alert) => (
          <div
            className={`rounded-xl border p-4 ${
              alert.severity === "critical"
                ? "border-rose-200 bg-rose-50"
                : "border-amber-200 bg-amber-50"
            }`}
            key={alert.code}
          >
            <div className="text-xs font-bold text-slate-800">{alert.title}</div>
            <div className="mt-1 text-xs text-slate-600">{alert.detail}</div>
          </div>
        ))}
        {diagnostics.data && diagnostics.data.alerts.length === 0 ? (
          <div className="panel col-span-2 p-4 text-xs text-emerald-700">
            No diagnostic threshold violations in recent traces.
          </div>
        ) : null}
      </div>
      <h2 className="mb-3 mt-6 text-sm font-bold text-slate-800">Recent errors</h2>
      {errors.isError ? (
        <ErrorState error={errors.error} />
      ) : (
        <ErrorsTable rows={errors.data?.items || []} />
      )}
    </PageContainer>
  );
}

function HealthCard({
  label,
  status,
  value,
  icon: Icon,
}: {
  label: string;
  status?: string;
  value: string;
  icon: typeof Database;
}) {
  return (
    <div className="panel p-4">
      <div className="flex items-start justify-between gap-2">
        <span className="rounded-xl bg-slate-100 p-2.5 text-slate-600">
          <Icon size={18} />
        </span>
        <StatusBadge status={status || "unknown"} compact />
      </div>
      <div className="mt-4 text-sm font-bold text-slate-800">{label}</div>
      <div className="mt-1 truncate text-[10px] text-slate-400" title={value}>
        {value}
      </div>
    </div>
  );
}

function ErrorsTable({ rows }: { rows: RecentError[] }) {
  const columns: Column<RecentError>[] = [
    { key: "time", label: "time", render: (row) => formatDate(row.time) },
    {
      key: "trace",
      label: "trace_id",
      render: (row) => <span className="mono text-[10px]">{truncate(row.trace_id, 18)}</span>,
    },
    { key: "step", label: "step", render: (row) => row.step },
    { key: "type", label: "error_type", render: (row) => row.error_type },
    {
      key: "message",
      label: "message",
      render: (row) => <span className="text-rose-600">{row.message || "—"}</span>,
    },
  ];
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) => `${row.trace_id}-${row.step}-${row.time}`}
      emptyTitle="No recent errors"
    />
  );
}
