import { useQuery } from "@tanstack/react-query";
import {
  BrainCircuit,
  ChevronRight,
  Clock3,
  FileJson,
  FileSearch,
  Link2,
  Search,
  ScanSearch,
  Sigma,
  Waypoints,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { getTrace, listTraces } from "../../api/traces";
import type { TraceRecord, TraceStep } from "../../types";
import { formatDate, formatLatency, formatPercent } from "../../utils/format";
import { JsonViewer } from "../common/JsonViewer";
import { EmptyState, ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { PageContainer } from "../layout/PageContainer";
import {
  extractEntityFlow,
  tokenStatsForStep,
  tokenSummaryRows,
  type EntityFlowView,
} from "./tracePresentation";

export function TraceExplorerPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTraceId = searchParams.get("trace_id") || "";
  const [search, setSearch] = useState(initialTraceId);
  const [selectedId, setSelectedId] = useState<string | undefined>(
    initialTraceId || undefined,
  );
  const [selectedStep, setSelectedStep] = useState<TraceStep>();
  const [isStepDetailOpen, setIsStepDetailOpen] = useState(false);
  const [isFinalResultOpen, setIsFinalResultOpen] = useState(false);
  const traces = useQuery({ queryKey: ["traces"], queryFn: listTraces });
  const detail = useQuery({
    queryKey: ["trace", selectedId],
    queryFn: () => getTrace(selectedId!),
    enabled: Boolean(selectedId),
  });
  useEffect(() => {
    setSelectedStep(undefined);
    setIsStepDetailOpen(false);
    setIsFinalResultOpen(false);
  }, [selectedId]);
  const openSearch = () => {
    const traceId = search.trim();
    if (traceId) {
      setSelectedId(traceId);
      setSearchParams({ trace_id: traceId });
    }
  };
  return (
    <PageContainer
      title="Trace Explorer"
      description="Tìm trace, xem final answer và inspect input/output của từng pipeline step."
    >
      <div className="panel mb-4 p-4">
        <div className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-400">
          Inspect Trace
        </div>
        <div className="flex flex-col gap-3 sm:flex-row">
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
          <button className="primary-button px-5 py-3 text-sm" onClick={openSearch}>
            <FileSearch size={17} /> Inspect Trace
          </button>
        </div>
      </div>

      {selectedId && detail.data && <TraceSummary trace={detail.data} />}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div className="panel h-[320px] overflow-y-auto p-3">
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
                  className={`h-[84px] w-full rounded-xl border p-2.5 text-left transition ${
                    selectedId === trace.trace_id
                      ? "border-teal-300 bg-teal-50"
                      : "border-transparent hover:border-slate-200 hover:bg-slate-50"
                  }`}
                  onClick={() => {
                    setSelectedId(trace.trace_id);
                    setSearch(trace.trace_id);
                    setSearchParams({ trace_id: trace.trace_id });
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <StatusBadge status={trace.status} compact />
                    <span className="text-[10px] text-slate-400">
                      {formatLatency(trace.total_latency_ms)}
                    </span>
                  </div>
                  <div className="mt-1.5 line-clamp-2 text-xs font-semibold leading-4 text-slate-700">
                    {trace.query || "Hidden query"}
                  </div>
                  <div className="mono mt-1.5 truncate text-[9px] text-slate-400">
                    {trace.trace_id}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="min-w-0">
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
            <PipelineTimeline
              trace={detail.data}
              selectedStep={selectedStep}
              onSelectStep={(step) => {
                setSelectedStep(step);
                setIsStepDetailOpen(true);
              }}
            />
          ) : null}
        </div>
      </div>

      {selectedId && detail.data && (
        <>
          <StepDetail
            selectedStep={selectedStep}
            isOpen={isStepDetailOpen}
            onToggle={() => setIsStepDetailOpen((value) => !value)}
          />
          <FinalResult
            trace={detail.data}
            isOpen={isFinalResultOpen}
            onToggle={() => setIsFinalResultOpen((value) => !value)}
          />
        </>
      )}
    </PageContainer>
  );
}

function TraceSummary({ trace }: { trace: TraceRecord }) {
  const plannedTasks = extractPlannedTasks(trace.steps || []);
  const tokenRows = tokenSummaryRows(trace.steps || []);
  const entityFlow = extractEntityFlow(trace.steps || []);
  return (
    <div className="panel mb-4 p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
            Trace summary
          </div>
          <h2 className="mt-2 max-w-5xl text-lg font-bold leading-7 text-slate-900">
            {trace.query || "Query unavailable"}
          </h2>
        </div>
        <StatusBadge status={trace.status} />
      </div>
      <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[
          ["Intent", trace.intent || "—"],
          ["Confidence", formatPercent(trace.confidence)],
          ["Total latency", formatLatency(trace.total_latency_ms)],
          ["Created", formatDate(trace.created_at)],
        ].map(([label, value]) => (
          <div key={label} className="rounded-lg bg-slate-50 p-3">
            <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
              {label}
            </div>
            <div className="mt-1 text-xs font-bold text-slate-700">{value}</div>
          </div>
        ))}
      </div>
      {tokenRows.length > 0 && (
        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
          {tokenRows.map((row) => (
            <div
              key={row.label}
              className="rounded-lg border border-indigo-100 bg-indigo-50/60 p-3"
            >
              <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-wider text-indigo-500">
                <Sigma size={13} /> {row.label} Tokens
              </div>
              <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                <TokenMetric label="Input" value={row.inputTokens} />
                <TokenMetric label="Output" value={row.outputTokens} />
                <TokenMetric label="Total" value={row.totalTokens} />
              </div>
              {row.note && (
                <div className="mt-2 text-[10px] text-indigo-500">{row.note}</div>
              )}
            </div>
          ))}
        </div>
      )}
      {plannedTasks.length > 0 && (
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
            Planned tasks
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {plannedTasks.map((task, index) => (
              <span
                key={`${task.intent}-${index}`}
                className="rounded-lg border border-teal-200 bg-teal-50 px-2.5 py-1.5 text-[10px] font-semibold text-teal-800"
                title={
                  task.effectiveQuery
                    ? `Planner: ${task.query}\nEffective: ${task.effectiveQuery}`
                    : task.query
                }
              >
                {task.intent}: {task.query}
                {task.effectiveQuery && task.effectiveQuery !== task.query
                  ? ` → ${task.effectiveQuery}`
                  : ""}
              </span>
            ))}
          </div>
        </div>
      )}
      {entityFlow && <EntityMemoryFlow flow={entityFlow} />}
      {(trace.error || trace.failed_step) && (
        <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
          Failed at <strong>{trace.failed_step}</strong>: {trace.error}
        </div>
      )}
    </div>
  );
}

function EntityMemoryFlow({ flow }: { flow: EntityFlowView }) {
  const memoryNames = [
    ...stringList(flow.memoryState?.active_product_names),
    ...stringList(flow.memoryState?.active_service_names),
  ];
  return (
    <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
        <BrainCircuit size={14} className="text-violet-600" /> Entity & memory flow
      </div>
      <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-2">
        <div className="rounded-lg border border-violet-100 bg-violet-50/60 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-violet-700">
              <ScanSearch size={13} /> Span extraction
            </span>
            <span className="text-[9px] text-violet-500">
              {flow.provider || "unknown"}
              {flow.degraded ? " · degraded" : ""}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {flow.spans.length ? (
              flow.spans.map((span, index) => (
                <span
                  key={`${span.label}-${span.text}-${index}`}
                  className="rounded-md border border-violet-200 bg-white px-2 py-1 text-[10px] text-violet-800"
                  title={`${span.source || "unknown"} · ${formatSpanScore(span.score)}`}
                >
                  <strong>{span.label}</strong>: {span.text}
                </span>
              ))
            ) : (
              <span className="text-[10px] text-slate-400">Không có explicit span.</span>
            )}
          </div>
          {flow.error && (
            <div className="mt-2 text-[10px] text-rose-600">{flow.error}</div>
          )}
        </div>

        <div className="rounded-lg border border-teal-100 bg-teal-50/60 p-3">
          <div className="flex items-center gap-1.5 text-[10px] font-bold text-teal-700">
            <Link2 size={13} /> Context binding
          </div>
          <div className="mt-2 space-y-2">
            {flow.decisions.length ? (
              flow.decisions.map((decision) => (
                <div
                  key={decision.taskId}
                  className="rounded-md border border-teal-100 bg-white/80 p-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-1 text-[9px]">
                    <strong className="text-slate-700">{decision.intent}</strong>
                    <span className="text-teal-600">
                      {decision.referenceMode} · {decision.bindingSource}
                    </span>
                  </div>
                  <div className="mt-1 text-[10px] text-slate-600">
                    {entityList(decision.before)} →{" "}
                    <strong>{entityList(decision.after)}</strong>
                  </div>
                  {decision.reason && (
                    <div className="mt-1 text-[9px] text-slate-400">{decision.reason}</div>
                  )}
                </div>
              ))
            ) : (
              <span className="text-[10px] text-slate-400">Không có binding decision.</span>
            )}
          </div>
        </div>

        <div className="rounded-lg border border-blue-100 bg-blue-50/60 p-3">
          <div className="flex items-center justify-between gap-2 text-[10px] font-bold text-blue-700">
            <span>Conversation memory</span>
            <span>{flow.memoryTurnCount ?? 0} turns loaded</span>
          </div>
          <div className="mt-2 space-y-1.5 text-[10px]">
            <MemoryValue label="Active topic" value={flow.memoryState?.active_topic} />
            <MemoryValue label="Active domain" value={flow.memoryState?.active_domain} />
            <MemoryValue
              label="Remembered entities"
              value={memoryNames.length ? memoryNames.join(", ") : undefined}
            />
            <MemoryValue
              label="Last intents"
              value={stringList(flow.memoryState?.last_intents).join(", ") || undefined}
            />
          </div>
        </div>

        <div className="rounded-lg border border-amber-100 bg-amber-50/60 p-3">
          <div className="text-[10px] font-bold text-amber-700">
            Canonical tasks & gates
          </div>
          <div className="mt-2 space-y-2">
            {flow.canonicalTasks.length ? (
              flow.canonicalTasks.map((task) => (
                <div
                  key={task.taskId}
                  className="rounded-md border border-amber-100 bg-white/80 p-2 text-[10px]"
                >
                  <div className="flex justify-between gap-2">
                    <strong>{task.intent}</strong>
                    <span className="text-amber-600">
                      {task.resolutionStatus || "unknown"}
                    </span>
                  </div>
                  <div className="mt-1 text-slate-600">
                    Entity: {entityList(task.entityNames)}
                  </div>
                  <div className="mt-1 break-all text-slate-400">
                    IDs: {entityList(task.resolvedIds)}
                  </div>
                  {task.effectiveQuery && (
                    <div className="mt-1 text-slate-500">
                      Effective: {task.effectiveQuery}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <span className="text-[10px] text-slate-400">
                Không có canonical entity; đây có thể là list/filter task.
              </span>
            )}
          </div>
          {flow.gates.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {flow.gates.map((gate) => (
                <span
                  key={gate.name}
                  className="rounded-md border border-amber-200 bg-white px-2 py-1 text-[9px] text-amber-800"
                >
                  {gate.name}: {gate.status} · {gate.validTaskIds.length} valid ·{" "}
                  {gate.violationCount} violations
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MemoryValue({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="flex justify-between gap-3 rounded-md bg-white/70 px-2 py-1.5">
      <span className="text-slate-400">{label}</span>
      <span className="max-w-[65%] text-right font-semibold text-slate-700">
        {value == null || value === "" ? "—" : String(value)}
      </span>
    </div>
  );
}

function PipelineTimeline({
  trace,
  selectedStep,
  onSelectStep,
}: {
  trace: TraceRecord;
  selectedStep?: TraceStep;
  onSelectStep: (step: TraceStep) => void;
}) {
  return (
        <div className="panel h-[320px] overflow-y-auto p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-bold text-slate-800">
            <Waypoints size={16} className="text-teal-600" /> Pipeline timeline
          </div>
          <div className="relative ml-2 border-l-2 border-slate-200 pl-6">
            {(trace.steps || []).map((step) => (
              <button
                key={step.step_id || step.step_name}
                className={`relative mb-3 w-full rounded-xl border p-3 text-left transition ${
                  selectedStep === step
                    ? "border-teal-300 bg-teal-50"
                    : "border-slate-200 bg-white hover:border-teal-300 hover:bg-teal-50/40"
                }`}
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
                    <StepTokenPill step={step} />
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
  );
}

function StepDetail({
  selectedStep,
  isOpen,
  onToggle,
}: {
  selectedStep?: TraceStep;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="panel mt-5 overflow-hidden">
      <button
        className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
        onClick={onToggle}
      >
        <span className="text-sm font-bold text-slate-800">Step detail</span>
        <ChevronRight
          size={16}
          className={`text-slate-400 transition ${isOpen ? "rotate-90" : ""}`}
        />
      </button>
      {isOpen && selectedStep ? (
        <div className="max-h-[620px] space-y-3 overflow-y-auto border-t border-slate-100 p-5">
          <div className="flex items-center justify-between gap-3">
            <span className="mono truncate text-xs font-bold">{selectedStep.step_name}</span>
            <StatusBadge status={selectedStep.status} />
          </div>
          {selectedStep.error_message && (
            <div className="rounded-lg bg-rose-50 p-3 text-xs text-rose-700">
              {selectedStep.error_message}
            </div>
          )}
          <StepTokenPanel step={selectedStep} />
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <div className="min-w-0">
              <div className="mb-1 text-[10px] font-bold uppercase text-slate-400">
                Input summary
              </div>
              <JsonViewer value={selectedStep.input} maxHeight="360px" />
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[10px] font-bold uppercase text-slate-400">
                Output summary
              </div>
              <JsonViewer value={selectedStep.output} maxHeight="360px" />
            </div>
          </div>
        </div>
      ) : isOpen ? (
        <div className="border-t border-slate-100 p-5">
          <EmptyState title="Select a pipeline step" />
        </div>
      ) : null}
    </div>
  );
}

function FinalResult({
  trace,
  isOpen,
  onToggle,
}: {
  trace: TraceRecord;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="panel mt-5 overflow-hidden">
      <button
        className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left"
        onClick={onToggle}
      >
        <span className="flex items-center gap-2 text-sm font-bold text-slate-800">
          <FileJson size={16} className="text-teal-600" /> Final Result
        </span>
        <ChevronRight
          size={16}
          className={`text-slate-400 transition ${isOpen ? "rotate-90" : ""}`}
        />
      </button>
      {isOpen && (
        <div className="border-t border-slate-100 p-4">
          <JsonViewer value={trace.final_answer || {}} maxHeight="520px" />
        </div>
      )}
    </div>
  );
}

function extractPlannedTasks(
  steps: TraceStep[],
): Array<{ intent: string; query: string; effectiveQuery?: string }> {
  const planning = steps.find((step) => step.step_name === "task_planning");
  const canonical = steps.find(
    (step) => step.step_name === "task_canonicalization",
  )?.output;
  const effectiveByTask = new Map<string, string>();
  if (canonical && Array.isArray(canonical.tasks)) {
    canonical.tasks.forEach((task) => {
      if (!isRecord(task)) return;
      const taskId = String(task.task_id || "").trim();
      const effectiveQuery = String(task.effective_query || "").trim();
      if (taskId && effectiveQuery) effectiveByTask.set(taskId, effectiveQuery);
    });
  }
  const output = planning?.output;
  if (!output || !Array.isArray(output.tasks)) return [];
  return output.tasks.flatMap((task) => {
    if (!task || typeof task !== "object") return [];
    const record = task as Record<string, unknown>;
    const taskId = String(record.task_id || "").trim();
    const intent = String(record.intent || "").trim();
    const query = String(record.planner_query || record.query || "").trim();
    return intent && query
      ? [{ intent, query, effectiveQuery: effectiveByTask.get(taskId) }]
      : [];
  });
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map(String).filter(Boolean);
}

function entityList(values: string[]) {
  return values.length ? values.join(", ") : "∅";
}

function formatSpanScore(value?: number) {
  return value == null ? "no score" : `${(value * 100).toFixed(1)}%`;
}

function TokenMetric({
  label,
  value,
}: {
  label: string;
  value?: number;
}) {
  return (
    <div className="rounded-md bg-white/70 p-2">
      <div className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <div className="mt-1 font-bold text-slate-800">{formatTokenCount(value)}</div>
    </div>
  );
}

function StepTokenPill({ step }: { step: TraceStep }) {
  const stats = tokenStatsForStep(step);
  if (!stats) return null;
  return (
    <span
      className="flex items-center gap-1 rounded-full bg-indigo-50 px-2 py-1 text-[10px] font-semibold text-indigo-600"
      title={`Input: ${formatTokenCount(stats.inputTokens)}, Output: ${formatTokenCount(stats.outputTokens)}`}
    >
      <Sigma size={11} />
      {formatTokenCount(stats.totalTokens)} tok
    </span>
  );
}

function StepTokenPanel({ step }: { step: TraceStep }) {
  const stats = tokenStatsForStep(step);
  if (!stats) return null;
  return (
    <div className="rounded-lg border border-indigo-100 bg-indigo-50/70 p-3">
      <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-indigo-600">
        <Sigma size={14} /> Token usage
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <TokenMetric label="Input" value={stats.inputTokens} />
        <TokenMetric label="Output" value={stats.outputTokens} />
        <TokenMetric label="Total" value={stats.totalTokens} />
      </div>
      {stats.attempts?.length ? (
        <div className="mt-3 space-y-1 text-[10px] text-indigo-700">
          {stats.attempts.map((attempt, index) => (
            <div key={index} className="rounded-md bg-white/70 px-2 py-1">
              Attempt {index + 1}: input{" "}
              {formatTokenCount(numberValue(attempt.prompt_eval_count))}, output{" "}
              {formatTokenCount(numberValue(attempt.eval_count))}
              {attempt.model ? ` · ${String(attempt.model)}` : ""}
            </div>
          ))}
        </div>
      ) : null}
      {stats.note && (
        <div className="mt-2 text-[10px] text-indigo-500">{stats.note}</div>
      )}
    </div>
  );
}

function numberValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatTokenCount(value?: number) {
  if (value === undefined || value === null) return "—";
  return Math.round(value).toLocaleString("vi-VN");
}
