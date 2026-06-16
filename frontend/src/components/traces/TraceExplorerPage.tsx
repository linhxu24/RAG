import { useQuery } from "@tanstack/react-query";
import {
  ChevronRight,
  Clock3,
  FileSearch,
  Search,
  Waypoints,
} from "lucide-react";
import { useEffect, useState } from "react";

import { getTrace, listTraces } from "../../api/traces";
import type { TraceRecord, TraceStep } from "../../types";
import { formatDate, formatLatency, formatPercent } from "../../utils/format";
import { JsonViewer } from "../common/JsonViewer";
import { EmptyState, ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";

export function TraceExplorerPage() {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string>();
  const [selectedStep, setSelectedStep] = useState<TraceStep>();
  const traces = useQuery({ queryKey: ["traces"], queryFn: listTraces });
  const detail = useQuery({
    queryKey: ["trace", selectedId],
    queryFn: () => getTrace(selectedId!),
    enabled: Boolean(selectedId),
  });
  useEffect(() => setSelectedStep(undefined), [selectedId]);
  const openSearch = () => {
    if (search.trim()) setSelectedId(search.trim());
  };
  return (
    <PageContainer
      title="Trace Explorer"
      description="Tìm trace, xem final answer và inspect input/output của từng pipeline step."
    >
      <div className="mb-5 flex gap-3">
        <div className="relative flex-1">
          <Search
            size={17}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
          />
          <input
            className="control py-3 pl-10 pr-4 text-sm"
            placeholder="Nhập trace_id..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && openSearch()}
          />
        </div>
        <button className="primary-button px-5 text-sm" onClick={openSearch}>
          <FileSearch size={17} /> Inspect Trace
        </button>
      </div>
      <div className="grid grid-cols-[350px_minmax(0,1fr)] gap-5">
        <div className="panel max-h-[calc(100vh-230px)] overflow-y-auto p-3">
          <div className="px-2 pb-3 pt-1 text-xs font-bold uppercase tracking-wider text-slate-400">
            Recent traces
          </div>
          {traces.isLoading ? (
            <LoadingState />
          ) : traces.isError ? (
            <ErrorState error={traces.error} />
          ) : !traces.data?.items.length ? (
            <EmptyState title="No traces yet" />
          ) : (
            <div className="space-y-1.5">
              {traces.data.items.map((trace) => (
                <button
                  key={trace.trace_id}
                  className={`w-full rounded-xl border p-3 text-left transition ${
                    selectedId === trace.trace_id
                      ? "border-teal-300 bg-teal-50"
                      : "border-transparent hover:border-slate-200 hover:bg-slate-50"
                  }`}
                  onClick={() => {
                    setSelectedId(trace.trace_id);
                    setSearch(trace.trace_id);
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <StatusBadge status={trace.status} compact />
                    <span className="text-[10px] text-slate-400">
                      {formatLatency(trace.total_latency_ms)}
                    </span>
                  </div>
                  <div className="mt-2 line-clamp-3 text-xs font-semibold leading-5 text-slate-700">
                    {trace.query || "Hidden query"}
                  </div>
                  <div className="mono mt-2 truncate text-[9px] text-slate-400">
                    {trace.trace_id}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
        <div>
          {!selectedId ? (
            <EmptyState
              title="Select a trace"
              description="Chọn recent trace hoặc nhập trace_id để xem timeline."
            />
          ) : detail.isLoading ? (
            <LoadingState label="Đang tải trace..." />
          ) : detail.isError ? (
            <ErrorState error={detail.error} onRetry={() => void detail.refetch()} />
          ) : detail.data ? (
            <TraceDetail
              trace={detail.data}
              selectedStep={selectedStep}
              onSelectStep={setSelectedStep}
            />
          ) : null}
        </div>
      </div>
    </PageContainer>
  );
}

function TraceDetail({
  trace,
  selectedStep,
  onSelectStep,
}: {
  trace: TraceRecord;
  selectedStep?: TraceStep;
  onSelectStep: (step: TraceStep) => void;
}) {
  return (
    <div className="space-y-5">
      <div className="panel p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              Trace summary
            </div>
            <h2 className="mt-2 max-w-3xl text-lg font-bold leading-7 text-slate-900">
              {trace.query || "Query unavailable"}
            </h2>
          </div>
          <StatusBadge status={trace.status} />
        </div>
        <div className="mt-5 grid grid-cols-4 gap-3">
          {[
            ["Intent", trace.intent || "—"],
            ["Confidence", formatPercent(trace.confidence)],
            ["Total latency", formatLatency(trace.total_latency_ms)],
            ["Created", formatDate(trace.created_at)],
          ].map(([label, value]) => (
            <div key={label} className="rounded-xl bg-slate-50 p-3">
              <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                {label}
              </div>
              <div className="mt-1 text-xs font-bold text-slate-700">{value}</div>
            </div>
          ))}
        </div>
        {(trace.error || trace.failed_step) && (
          <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
            Failed at <strong>{trace.failed_step}</strong>: {trace.error}
          </div>
        )}
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_430px] gap-5">
        <div className="panel p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-bold text-slate-800">
            <Waypoints size={16} className="text-teal-600" /> Pipeline timeline
          </div>
          <div className="relative ml-2 border-l-2 border-slate-200 pl-6">
            {(trace.steps || []).map((step) => (
              <button
                key={step.step_id || step.step_name}
                className="relative mb-3 w-full rounded-xl border border-slate-200 bg-white p-3 text-left hover:border-teal-300 hover:bg-teal-50/40"
                onClick={() => onSelectStep(step)}
              >
                <span
                  className={`absolute -left-[33px] top-4 h-3.5 w-3.5 rounded-full border-4 border-white ${
                    step.status === "failed"
                      ? "bg-rose-500"
                      : step.status === "skipped"
                        ? "bg-slate-400"
                        : "bg-emerald-500"
                  }`}
                />
                <div className="flex items-center justify-between gap-3">
                  <div className="mono text-xs font-bold text-slate-700">
                    {step.step_name}
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1 text-[10px] text-slate-400">
                      <Clock3 size={11} /> {step.latency_ms} ms
                    </span>
                    <StatusBadge status={step.status} compact />
                    <ChevronRight size={14} className="text-slate-400" />
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-5">
          <div className="panel p-5">
            <div className="mb-3 text-sm font-bold text-slate-800">Step detail</div>
            {selectedStep ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="mono text-xs font-bold">{selectedStep.step_name}</span>
                  <StatusBadge status={selectedStep.status} />
                </div>
                {selectedStep.error_message && (
                  <div className="rounded-xl bg-rose-50 p-3 text-xs text-rose-700">
                    {selectedStep.error_message}
                  </div>
                )}
                <div>
                  <div className="mb-1 text-[10px] font-bold uppercase text-slate-400">
                    Input summary
                  </div>
                  <JsonViewer value={selectedStep.input} maxHeight="220px" />
                </div>
                <div>
                  <div className="mb-1 text-[10px] font-bold uppercase text-slate-400">
                    Output summary
                  </div>
                  <JsonViewer value={selectedStep.output} maxHeight="260px" />
                </div>
              </div>
            ) : (
              <EmptyState title="Select a pipeline step" />
            )}
          </div>
          <details className="panel overflow-hidden">
            <summary className="cursor-pointer px-5 py-4 text-sm font-bold text-slate-800">
              Final answer
            </summary>
            <div className="border-t border-slate-100 p-4">
              <JsonViewer value={trace.final_answer || {}} maxHeight="360px" />
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
