import { useQuery } from "@tanstack/react-query";
import {
  Boxes,
  Clock3,
  FileCheck2,
  FileX2,
  Image,
  Layers3,
  Rows3,
  ScanText,
  X,
} from "lucide-react";
import { useState } from "react";

import {
  getIngestionRun,
  getIngestionSummary,
  listIngestionRuns,
} from "../../api/ingestion";
import type { IngestionRun } from "../../types";
import { formatDate, formatLatency, formatPercent, truncate } from "../../utils/format";
import { DataTable, type Column } from "../common/DataTable";
import { MetricCard } from "../common/MetricCard";
import { ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

export function IngestionMonitorPage() {
  const [selected, setSelected] = useState<string>();
  const runs = useQuery({ queryKey: ["ingestion-runs"], queryFn: listIngestionRuns });
  const summary = useQuery({
    queryKey: ["ingestion-summary"],
    queryFn: getIngestionSummary,
  });
  const detail = useQuery({
    queryKey: ["ingestion-run", selected],
    queryFn: () => getIngestionRun(selected!),
    enabled: Boolean(selected),
  });
  const columns: Column<IngestionRun>[] = [
    {
      key: "run",
      label: "run_id",
      render: (row) => (
        <button
          className="mono text-[10px] font-bold text-teal-700 hover:underline"
          onClick={() => setSelected(row.run_id)}
        >
          {truncate(row.run_id, 18)}
        </button>
      ),
    },
    { key: "doc", label: "file_name", render: (row) => row.file_name || row.doc_id },
    { key: "status", label: "status", render: (row) => <StatusBadge status={row.status} /> },
    { key: "started", label: "started_at", render: (row) => formatDate(row.started_at) },
    { key: "ended", label: "ended_at", render: (row) => formatDate(row.ended_at) },
    {
      key: "latency",
      label: "latency",
      render: (row) => formatLatency(row.total_latency_ms),
    },
    {
      key: "error",
      label: "error",
      render: (row) => (
        <span className="line-clamp-3 text-xs text-rose-600">{row.error_message || "—"}</span>
      ),
    },
  ];
  const metrics = summary.data || {};
  return (
    <PageContainer
      title="Ingestion Monitor"
      description="Theo dõi parsing, table extraction, assets, embeddings và ingestion quality."
    >
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total documents" value={metrics.total_documents ?? "—"} icon={Boxes} />
        <MetricCard
          label="Parse success"
          value={formatPercent(metrics.parse_success_rate)}
          icon={FileCheck2}
        />
        <MetricCard
          label="Parse failed"
          value={metrics.parse_failed_count ?? "—"}
          icon={FileX2}
          accent="rose"
        />
        <MetricCard
          label="Chunks created"
          value={metrics.chunks_created ?? "—"}
          icon={ScanText}
          accent="blue"
        />
        <MetricCard label="Tables detected" value={metrics.tables_detected ?? "—"} icon={Layers3} />
        <MetricCard label="Table rows" value={metrics.table_rows_created ?? "—"} icon={Rows3} />
        <MetricCard label="Assets extracted" value={metrics.assets_extracted ?? "—"} icon={Image} />
        <MetricCard
          label="Average latency"
          value={formatLatency(metrics.average_ingestion_latency_ms)}
          icon={Clock3}
          accent="amber"
        />
      </div>
      <div className="mt-6">
        {runs.isLoading ? (
          <LoadingState />
        ) : runs.isError ? (
          <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
        ) : (
          <DataTable
            rows={runs.data?.items || []}
            columns={columns}
            rowKey={(row) => row.run_id}
            emptyTitle="Chưa có ingestion run"
          />
        )}
      </div>
      {selected && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-slate-950/30 p-8 backdrop-blur-sm">
          <div className="panel max-h-[86vh] w-full max-w-3xl overflow-y-auto">
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
              <div>
                <div className="font-bold text-slate-900">Pipeline timeline</div>
                <div className="mono text-[10px] text-slate-400">{selected}</div>
              </div>
              <button onClick={() => setSelected(undefined)}>
                <X size={19} className="text-slate-500" />
              </button>
            </div>
            <div className="p-6">
              {detail.isLoading ? (
                <LoadingState />
              ) : detail.isError ? (
                <ErrorState error={detail.error} />
              ) : detail.data ? (
                <Timeline run={detail.data} />
              ) : null}
            </div>
          </div>
        </div>
      )}
    </PageContainer>
  );
}

function Timeline({ run }: { run: IngestionRun }) {
  const expected = [
    "upload_file",
    "docling_parse",
    "normalize_blocks",
    "asset_masking",
    "table_processing",
    "chunking",
    "embedding",
    "quality_check",
    "save_to_postgres",
  ];
  const backendSteps = new Map(run.timeline.map((item) => [item.step_name, item]));
  const aliases: Record<string, string> = {
    docling_parse: "parse",
    asset_masking: "asset_storage",
    chunking: "chunking_embedding",
    embedding: "chunking_embedding",
    quality_check: "quality_checks",
  };
  return (
    <div className="relative ml-3 border-l-2 border-slate-200 pl-7">
      {expected.map((name) => {
        const step = backendSteps.get(name) || backendSteps.get(aliases[name]);
        return (
          <div key={name} className="relative mb-5 last:mb-0">
            <span
              className={`absolute -left-[37px] top-1 h-4 w-4 rounded-full border-4 border-white ${
                step?.status === "failed"
                  ? "bg-rose-500"
                  : step
                    ? "bg-emerald-500"
                    : "bg-slate-300"
              }`}
            />
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <div className="mono text-xs font-bold text-slate-700">{name}</div>
                  {step?.error_message && (
                    <div className="mt-1 text-xs text-rose-600">{step.error_message}</div>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className="mono text-xs text-slate-500">
                    {step ? `${step.latency_ms} ms` : "—"}
                  </span>
                  <StatusBadge status={step?.status || "pending"} compact />
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
