import { Braces, Database, Timer, Waypoints } from "lucide-react";

import type { ChatResponse, TraceRecord } from "../../types";
import { formatLatency, formatPercent } from "../../utils/format";
import { EmptyState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";

export function DebugPanel({
  response,
  trace,
  loading,
}: {
  response?: ChatResponse;
  trace?: TraceRecord;
  loading: boolean;
}) {
  if (!response && !loading) {
    return (
      <aside className="h-full overflow-y-auto border-l border-slate-200 bg-slate-50 p-4">
        <EmptyState
          title="Chưa có request"
          description="Metadata của request gần nhất sẽ hiển thị tại đây."
        />
      </aside>
    );
  }
  const answer = response?.answer || response?.message;
  const failedStep = trace?.failed_step || response?.error?.failed_step;
  const steps = trace?.steps || [];
  const chunks = answer?.items?.filter((item) => item.chunk_id).length || 0;
  const rows = answer?.items?.filter((item) => item.row_id).length || 0;
  const debug = response?.debug;
  const fields = [
    ["trace_id", response?.trace_id || "pending"],
    ["intent", trace?.intent || response?.intent || debug?.intent || "—"],
    [
      "confidence",
      trace?.confidence != null
        ? formatPercent(trace.confidence)
        : debug?.confidence != null
          ? formatPercent(debug.confidence)
          : "—",
    ],
    ["answer_type", debug?.answer_type || "—"],
    [
      "total_latency_ms",
      formatLatency(trace?.total_latency_ms || debug?.total_latency_ms || debug?.latency_ms),
    ],
    ["retrieval_used", String(Boolean(debug?.retrieval_used || chunks || rows))],
    ["chunks used", String(chunks || debug?.chunks_used?.length || 0)],
    ["rows used", String(rows || debug?.rows_used?.length || 0)],
    ["assets returned", String(answer?.assets?.length || 0)],
    ["json_valid", String(debug?.json_valid ?? !response?.error)],
    ["failed_step", failedStep || "—"],
    ["model used", debug?.model_used || "—"],
  ];
  return (
    <aside className="h-full overflow-y-auto border-l border-slate-200 bg-slate-50 p-4">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <div className="text-sm font-bold text-slate-800">Request Debug</div>
          <div className="text-[11px] text-slate-400">Latest request metadata</div>
        </div>
        <StatusBadge status={loading ? "running" : trace?.status || "completed"} compact />
      </div>
      <div className="space-y-2">
        {fields.map(([label, value]) => (
          <div
            key={label}
            className="rounded-xl border border-slate-200 bg-white px-3 py-2.5"
          >
            <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
              {label}
            </div>
            <div className="mono mt-1 break-all text-[11px] font-semibold text-slate-700">
              {value}
            </div>
          </div>
        ))}
      </div>
      {steps.length > 0 && (
        <div className="mt-5">
          <div className="mb-2 flex items-center gap-2 text-xs font-bold text-slate-700">
            <Waypoints size={15} className="text-teal-600" /> Pipeline steps
          </div>
          <div className="space-y-1.5">
            {steps.map((step) => (
              <div
                key={step.step_id || step.step_name}
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-2.5 py-2"
              >
                <div className="min-w-0">
                  <div className="truncate text-[10px] font-bold text-slate-600">
                    {step.step_name}
                  </div>
                  <div className="mt-0.5 flex items-center gap-1 text-[9px] text-slate-400">
                    <Timer size={9} /> {step.latency_ms} ms
                  </div>
                </div>
                <StatusBadge status={step.status} compact />
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="mt-5 grid grid-cols-3 gap-2 text-center">
        {[
          [Database, chunks + rows, "sources"],
          [Braces, steps.length, "steps"],
          [Timer, answer?.assets?.length || 0, "assets"],
        ].map(([Icon, value, label]) => {
          const IconComponent = Icon as typeof Database;
          return (
            <div key={String(label)} className="rounded-xl bg-slate-900 p-3 text-white">
              <IconComponent className="mx-auto text-teal-300" size={15} />
              <div className="mt-1 text-lg font-bold">{String(value)}</div>
              <div className="text-[9px] uppercase tracking-wide text-slate-400">
                {String(label)}
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
