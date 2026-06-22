import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Braces,
  CheckCircle2,
  Database,
  Gauge,
  Image,
  Link2,
  Network,
  Play,
  Route,
  ScanSearch,
  ShieldCheck,
  Target,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getEvaluationSummary,
  listEvaluationDatasets,
  listEvaluationResults,
  listEvaluationRuns,
  runEvaluation,
} from "../../api/evaluation";
import type {
  ConversationEvaluationMetrics,
  DiagnosticAlert,
  EvaluationCaseResult,
  EvaluationRun,
} from "../../types";
import {
  formatDate,
  formatLatency,
  formatPercent,
  truncate,
} from "../../utils/format";
import { DataTable, type Column } from "../common/DataTable";
import { MetricCard } from "../common/MetricCard";
import { EmptyState, ErrorState, LoadingState } from "../common/States";
import { StatusBadge } from "../common/StatusBadge";
import { useToast } from "../common/Toast";
import { PageContainer } from "../layout/PageContainer";

const DATASET_PRESETS = {
  basic: {
    label: "Basic grounded cases",
    path: undefined,
    name: "dental_basic_eval",
    version: "2.0",
    description: "Bộ configured mặc định cho router, retrieval và generation.",
  },
  conversation: {
    label: "Multi-turn scenarios",
    path: "eval_datasets/dental_conversation_scenarios.json",
    name: "dental_conversation_scenarios",
    version: "1.0",
    description: "10 scenario, mỗi scenario dùng chung session_id qua 4-6 lượt.",
  },
  semantic: {
    label: "Semantic variants",
    path: "eval_datasets/dental_semantic_groups.json",
    name: "dental_semantic_groups",
    version: "1.0",
    description: "Các cách diễn đạt khác nhau nhưng cùng nghĩa nghiệp vụ.",
  },
} as const;

type DatasetPresetKey = keyof typeof DATASET_PRESETS;

export function EvaluationDashboardPage() {
  const client = useQueryClient();
  const notify = useToast();
  const [mode, setMode] = useState<"router" | "e2e" | "all">("all");
  const [profile, setProfile] = useState<"deterministic" | "production">("deterministic");
  const [dataVersion, setDataVersion] = useState("product-pdf-v2");
  const [datasetPreset, setDatasetPreset] = useState<DatasetPresetKey>("basic");
  const [selectedRunId, setSelectedRunId] = useState<string>();
  const summary = useQuery({
    queryKey: ["evaluation-summary"],
    queryFn: getEvaluationSummary,
  });
  const runs = useQuery({ queryKey: ["evaluation-runs"], queryFn: listEvaluationRuns });
  const datasets = useQuery({
    queryKey: ["evaluation-datasets"],
    queryFn: listEvaluationDatasets,
  });
  const results = useQuery({
    queryKey: ["evaluation-results", selectedRunId],
    queryFn: () => listEvaluationResults(selectedRunId),
    enabled: Boolean(selectedRunId),
  });
  useEffect(() => {
    if (!selectedRunId && summary.data?.latest_run_id) {
      setSelectedRunId(summary.data.latest_run_id);
    }
  }, [selectedRunId, summary.data?.latest_run_id]);
  const preset = DATASET_PRESETS[datasetPreset];
  const mutation = useMutation({
    mutationFn: () =>
      runEvaluation({
        mode,
        dataVersion,
        profile,
        datasetPath: preset.path,
        datasetName: preset.name,
        datasetVersion: preset.version,
      }),
    onSuccess: async (response) => {
      notify("Evaluation completed.");
      await Promise.all([
        client.invalidateQueries({ queryKey: ["evaluation-summary"] }),
        client.invalidateQueries({ queryKey: ["evaluation-runs"] }),
        client.invalidateQueries({ queryKey: ["evaluation-results"] }),
        client.invalidateQueries({ queryKey: ["evaluation-datasets"] }),
      ]);
      if (
        response &&
        typeof response === "object" &&
        "eval_run_id" in response &&
        typeof response.eval_run_id === "string"
      ) {
        setSelectedRunId(response.eval_run_id);
      }
    },
    onError: (error) => notify(error.message, "error"),
  });
  const metrics = summary.data;
  const latestRun = runs.data?.items.find(
    (run) => run.eval_run_id === metrics?.latest_run_id,
  );
  const activeDataset = datasets.data?.items.find(
    (dataset) => dataset.dataset_id === latestRun?.dataset_id,
  );
  const selectedRun = runs.data?.items.find(
    (run) => run.eval_run_id === selectedRunId,
  );
  const cards = [
    ["E2E Pass Rate", formatPercent(metrics?.e2e_pass_rate), CheckCircle2, qualityAccent(metrics?.e2e_pass_rate)],
    ["Router Accuracy", formatPercent(metrics?.router_accuracy), Route, qualityAccent(metrics?.router_accuracy)],
    ["Retrieval Recall@5", formatPercent(metrics?.retrieval_recall_at_5), Gauge, qualityAccent(metrics?.retrieval_recall_at_5)],
    ["Faithfulness", formatPercent(metrics?.faithfulness_rate), ShieldCheck, qualityAccent(metrics?.faithfulness_rate)],
    ["Answer Correctness", formatPercent(metrics?.answer_correctness), Target, qualityAccent(metrics?.answer_correctness)],
    ["Safety Pass", formatPercent(metrics?.safety_pass_rate), ShieldCheck, qualityAccent(metrics?.safety_pass_rate)],
    ["JSON Validity", formatPercent(metrics?.json_validity_rate), Braces, qualityAccent(metrics?.json_validity_rate)],
    ["Asset Resolve", formatPercent(metrics?.asset_resolve_rate), Image, qualityAccent(metrics?.asset_resolve_rate)],
    ["Fallback Rate", formatPercent(metrics?.fallback_rate), Activity, riskAccent(metrics?.fallback_rate)],
    ["No-result Rate", formatPercent(metrics?.no_result_rate), AlertTriangle, riskAccent(metrics?.no_result_rate)],
    ["Unsupported Claims", formatPercent(metrics?.unsupported_claim_rate), AlertTriangle, riskAccent(metrics?.unsupported_claim_rate)],
    ["p95 Latency", formatLatency(metrics?.p95_latency_ms), Timer, "amber"],
  ] as const;
  return (
    <PageContainer
      title="Evaluation Dashboard"
      description="Grounded metrics, per-case failures và diagnostics được lưu trong PostgreSQL theo từng evaluation run."
      actions={
        <button
          className="primary-button px-4 py-2.5 text-xs"
          disabled={mutation.isPending}
          onClick={() => mutation.mutate()}
        >
          <Play size={14} />
          {mutation.isPending ? "Running..." : "Run Evaluation"}
        </button>
      }
    >
      <EvaluationControls
        mode={mode}
        profile={profile}
        dataVersion={dataVersion}
        datasetPreset={datasetPreset}
        onModeChange={setMode}
        onProfileChange={setProfile}
        onDataVersionChange={setDataVersion}
        onDatasetPresetChange={setDatasetPreset}
      />
      {summary.isLoading ? (
        <LoadingState />
      ) : summary.isError ? (
        <ErrorState error={summary.error} onRetry={() => void summary.refetch()} />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
            {cards.map(([label, value, Icon, accent]) => (
              <MetricCard
                key={label}
                label={label}
                value={value}
                icon={Icon}
                accent={accent}
              />
            ))}
          </div>
          <CoveragePanel
            coverage={metrics?.coverage || {}}
            dataset={activeDataset}
          />
          <DiagnosticsPanel alerts={metrics?.diagnostics?.alerts || []} />
          <ConversationQualityPanel
            metrics={metrics?.conversation}
            selectedRunMetrics={selectedRun?.metrics?.conversation}
          />
          <div className="mt-5 grid grid-cols-1 gap-5 xl:grid-cols-2">
            <MetricHistory history={metrics?.history || []} />
            <PerIntentChart values={metrics?.per_intent || {}} />
          </div>
        </>
      )}
      <div className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-[0.75fr_1.6fr]">
        <section>
          <h2 className="mb-3 text-sm font-bold text-slate-800">Evaluation runs</h2>
          {runs.isError ? (
            <ErrorState error={runs.error} />
          ) : runs.isLoading ? (
            <LoadingState />
          ) : (
            <RunsTable rows={runs.data?.items || []} />
          )}
        </section>
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-800">Latest case results</h2>
            <select
              className="control w-72 px-3 py-2 text-[10px]"
              value={selectedRunId || ""}
              onChange={(event) => setSelectedRunId(event.target.value || undefined)}
              aria-label="Evaluation run"
            >
              <option value="">Chọn evaluation run</option>
              {(runs.data?.items || []).map((run) => (
                <option key={run.eval_run_id} value={run.eval_run_id}>
                  {run.dataset_name || "dataset"} · {formatDate(run.started_at)}
                </option>
              ))}
            </select>
          </div>
          {results.isError ? (
            <ErrorState error={results.error} />
          ) : results.isLoading ? (
            <LoadingState />
          ) : (
            <CaseResultsTable rows={results.data?.items || []} />
          )}
        </section>
      </div>
    </PageContainer>
  );
}

function EvaluationControls({
  mode,
  profile,
  dataVersion,
  datasetPreset,
  onModeChange,
  onProfileChange,
  onDataVersionChange,
  onDatasetPresetChange,
}: {
  mode: "router" | "e2e" | "all";
  profile: "deterministic" | "production";
  dataVersion: string;
  datasetPreset: DatasetPresetKey;
  onModeChange: (value: "router" | "e2e" | "all") => void;
  onProfileChange: (value: "deterministic" | "production") => void;
  onDataVersionChange: (value: string) => void;
  onDatasetPresetChange: (value: DatasetPresetKey) => void;
}) {
  const preset = DATASET_PRESETS[datasetPreset];
  return (
    <div className="panel mb-5 p-4">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
          Dataset
          <select
            className="control mt-1.5 px-3 py-2.5 text-xs normal-case"
            value={datasetPreset}
            onChange={(event) =>
              onDatasetPresetChange(event.target.value as DatasetPresetKey)
            }
          >
            {Object.entries(DATASET_PRESETS).map(([key, value]) => (
              <option key={key} value={key}>
                {value.label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
          Evaluation mode
          <select
            className="control mt-1.5 px-3 py-2.5 text-xs normal-case"
            value={mode}
            onChange={(event) => onModeChange(event.target.value as typeof mode)}
          >
            <option value="router">Router baseline</option>
            <option value="e2e">End-to-end</option>
            <option value="all">All grounded metrics</option>
          </select>
        </label>
        <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
          Runtime profile
          <select
            className="control mt-1.5 px-3 py-2.5 text-xs normal-case"
            value={profile}
            onChange={(event) =>
              onProfileChange(event.target.value as typeof profile)
            }
          >
            <option value="deterministic">Deterministic profile</option>
            <option value="production">Production profile</option>
          </select>
        </label>
        <label className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
          Data version
          <input
            className="control mt-1.5 px-3 py-2.5 text-xs normal-case"
            value={dataVersion}
            onChange={(event) => onDataVersionChange(event.target.value)}
          />
        </label>
      </div>
      <div className="mt-3 text-[10px] text-slate-500">
        <strong>{preset.label}:</strong> {preset.description}
        {preset.path ? ` · ${preset.path}` : ""}
      </div>
    </div>
  );
}

function CoveragePanel({
  coverage,
  dataset,
}: {
  coverage: Record<string, number>;
  dataset?: { name: string; version: string; metadata?: Record<string, unknown> };
}) {
  const values = [
    ["Retrieval ground truth", coverage.retrieval_ground_truth],
    ["Answer ground truth", coverage.answer_ground_truth],
    ["Faithfulness applicable", coverage.faithfulness_applicable],
    ["Safety applicable", coverage.safety_applicable],
  ];
  return (
    <div className="panel mt-5 p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
          <Database size={16} className="text-blue-600" /> Ground-truth coverage
        </div>
        <span className="text-xs text-slate-400">
          {dataset ? `${dataset.name} v${dataset.version}` : "No dataset"}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-3">
        {values.map(([label, value]) => (
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3" key={label}>
            <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
              {label}
            </div>
            <div className="mt-2 text-xl font-bold text-slate-800">
              {formatPercent(value as number | undefined)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConversationQualityPanel({
  metrics,
  selectedRunMetrics,
}: {
  metrics?: ConversationEvaluationMetrics;
  selectedRunMetrics?: ConversationEvaluationMetrics;
}) {
  const values = selectedRunMetrics || metrics;
  if (!values) return null;
  const scenarios = values.scenarios || [];
  const providers = Object.entries(values.entity_span_provider_counts || {});
  return (
    <div className="panel mt-5 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
            <Network size={16} className="text-violet-600" /> Multi-turn & entity binding
          </div>
          <div className="mt-1 text-[10px] text-slate-400">
            Chấm từ trace thật của GLiNER, context binder, memory và task planner.
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {providers.map(([provider, count]) => (
            <span
              key={provider}
              className="rounded-full border border-violet-200 bg-violet-50 px-2 py-1 text-[10px] text-violet-700"
            >
              {provider}: {count}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <ConversationMetric
          icon={Link2}
          label="Entity binding"
          value={formatPercent(values.entity_binding_accuracy)}
          count={values.entity_binding_case_count}
        />
        <ConversationMetric
          icon={BrainCircuit}
          label="Follow-up memory"
          value={formatPercent(values.follow_up_success_rate)}
          count={values.follow_up_case_count}
        />
        <ConversationMetric
          icon={Network}
          label="Multi-task"
          value={formatPercent(values.multi_task_success_rate)}
          count={values.multi_task_case_count}
        />
        <ConversationMetric
          icon={ScanSearch}
          label="NER degraded"
          value={formatPercent(values.entity_span_degraded_rate)}
          risk
        />
        <ConversationMetric
          icon={CheckCircle2}
          label="Scenario pass"
          value={formatPercent(values.scenario_pass_rate)}
          count={values.scenario_count}
        />
      </div>
      {scenarios.length > 0 && (
        <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3">
          {scenarios.map((scenario) => (
            <div
              key={scenario.scenario_key}
              className={`rounded-lg border p-3 ${
                scenario.passed
                  ? "border-emerald-200 bg-emerald-50/60"
                  : "border-rose-200 bg-rose-50/60"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="text-xs font-bold text-slate-800">
                    {scenario.scenario_title || scenario.scenario_key}
                  </div>
                  <div className="mono mt-1 text-[9px] text-slate-400">
                    {scenario.scenario_key}
                  </div>
                </div>
                <StatusBadge status={scenario.passed ? "success" : "failed"} compact />
              </div>
              <div className="mt-2 flex justify-between text-[10px] text-slate-600">
                <span>
                  {scenario.passed_turns}/{scenario.turn_count} turns
                </span>
                <span>
                  follow-up {formatPercent(scenario.follow_up_success_rate)}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ConversationMetric({
  icon: Icon,
  label,
  value,
  count,
  risk = false,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  count?: number;
  risk?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] font-bold uppercase tracking-wide text-slate-400">
          {label}
        </span>
        <Icon size={14} className={risk ? "text-amber-600" : "text-violet-600"} />
      </div>
      <div className="mt-2 text-xl font-bold text-slate-800">{value}</div>
      {count != null && (
        <div className="mt-1 text-[9px] text-slate-400">{count} applicable cases</div>
      )}
    </div>
  );
}

function DiagnosticsPanel({ alerts }: { alerts: DiagnosticAlert[] }) {
  return (
    <div className="panel mt-5 p-5">
      <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-800">
        <AlertTriangle size={16} className="text-amber-600" /> Diagnostic alerts
      </div>
      {alerts.length ? (
        <div className="grid grid-cols-2 gap-3">
          {alerts.map((alert) => (
            <div
              className={`rounded-xl border p-3 ${
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
        </div>
      ) : (
        <div className="text-xs text-emerald-700">No diagnostic threshold violations.</div>
      )}
    </div>
  );
}

function MetricHistory({ history }: { history: Array<Record<string, any>> }) {
  return (
    <div className="panel p-5">
      <div className="mb-4 flex items-center gap-2 text-sm font-bold text-slate-800">
        <Activity size={16} className="text-teal-600" /> Metrics over time
      </div>
      {history.length ? (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
            <XAxis dataKey="started_at" hide />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="pass_rate" stroke="#1b9b87" strokeWidth={2} />
            <Line
              type="monotone"
              dataKey="retrieval_recall_at_5"
              stroke="#315bdb"
              strokeWidth={2}
            />
            <Line
              type="monotone"
              dataKey="faithfulness_rate"
              stroke="#8b5cf6"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <EmptyState title="No comprehensive evaluation history" />
      )}
    </div>
  );
}

function PerIntentChart({
  values,
}: {
  values: Record<string, Record<string, number | null>>;
}) {
  const data = Object.entries(values).map(([intent, metrics]) => ({
    intent,
    pass_rate: metrics.pass_rate,
    router_accuracy: metrics.router_accuracy,
    retrieval_recall: metrics.retrieval_recall_at_5,
  }));
  return (
    <div className="panel p-5">
      <div className="mb-4 text-sm font-bold text-slate-800">Quality by intent</div>
      {data.length ? (
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e6ebf2" />
            <XAxis dataKey="intent" tick={{ fontSize: 8 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="pass_rate" fill="#1b9b87" />
            <Bar dataKey="router_accuracy" fill="#315bdb" />
            <Bar dataKey="retrieval_recall" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <EmptyState title="No per-intent data" />
      )}
    </div>
  );
}

function RunsTable({ rows }: { rows: EvaluationRun[] }) {
  const columns: Column<EvaluationRun>[] = [
    {
      key: "id",
      label: "run_id",
      render: (row) => <span className="mono text-[10px]">{truncate(row.eval_run_id, 14)}</span>,
      sortValue: (row) => row.eval_run_id,
      searchValue: (row) => row.eval_run_id,
    },
    {
      key: "dataset",
      label: "dataset",
      render: (row) => row.dataset_name || "—",
      sortValue: (row) => row.dataset_name,
      searchValue: (row) => `${row.dataset_name || ""} ${row.dataset_id || ""}`,
    },
    {
      key: "pass",
      label: "pass_rate",
      render: (row) => formatPercent(row.metrics?.e2e?.pass_rate),
      sortValue: (row) => row.metrics?.e2e?.pass_rate,
    },
    {
      key: "coverage",
      label: "retrieval_gt",
      render: (row) => formatPercent(row.metrics?.coverage?.retrieval_ground_truth),
      sortValue: (row) => row.metrics?.coverage?.retrieval_ground_truth,
    },
    {
      key: "status",
      label: "status",
      render: (row) => <StatusBadge status={row.status} />,
      sortValue: (row) => row.status,
      searchValue: (row) => row.status,
    },
    {
      key: "time",
      label: "started",
      render: (row) => formatDate(row.started_at),
      sortValue: (row) => row.started_at,
      searchValue: (row) => formatDate(row.started_at),
    },
  ];
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) => row.eval_run_id}
      emptyTitle="No evaluation runs"
      defaultSort={{ key: "time", direction: "desc" }}
    />
  );
}

function CaseResultsTable({ rows }: { rows: EvaluationCaseResult[] }) {
  const columns: Column<EvaluationCaseResult>[] = [
    {
      key: "query",
      label: "query",
      render: (row) => {
        const conversation = conversationDetails(row);
        const scenarioTitle = textValue(conversation.scenario_title);
        const turnIndex = textValue(conversation.turn_index);
        return (
          <div title={row.answer_text || undefined}>
            {scenarioTitle && (
              <div className="mb-1 text-[9px] font-bold uppercase text-violet-600">
                Turn {turnIndex || "?"} · {scenarioTitle}
              </div>
            )}
            <div>{truncate(row.query, 38)}</div>
            <div className="mt-1 text-[10px] text-slate-400">
              {truncate(row.answer_text, 58)}
            </div>
          </div>
        );
      },
      sortValue: (row) => row.query,
      searchValue: (row) => {
        const conversation = conversationDetails(row);
        return `${row.query} ${row.answer_text || ""} ${conversation.scenario_title || ""}`;
      },
    },
    {
      key: "intent",
      label: "expected → actual",
      render: (row) => (
        <span className="text-[10px]">
          {row.expected_intent || "—"} → {row.actual_intent || "ERROR"}
        </span>
      ),
      sortValue: (row) => row.actual_intent,
      searchValue: (row) => `${row.expected_intent || ""} ${row.actual_intent || ""}`,
    },
    {
      key: "binding",
      label: "entity / context",
      render: (row) => {
        const conversation = conversationDetails(row);
        const expected = stringArray(conversation.expected_entities);
        const actual = stringArray(conversation.actual_entities);
        const sources = stringArray(conversation.binding_sources);
        const bindingScore = row.scores.entity_binding_match;
        const followUpScore = row.scores.follow_up_memory;
        return (
          <div className="max-w-64 text-[10px]">
            <div>
              <span className="text-slate-400">Expected:</span>{" "}
              {expected.join(", ") || "—"}
            </div>
            <div className={bindingScore === 0 ? "text-rose-600" : "text-slate-700"}>
              <span className="text-slate-400">Bound:</span>{" "}
              {actual.join(", ") || "—"}
            </div>
            {(sources.length > 0 || followUpScore != null) && (
              <div className="mt-1 flex flex-wrap gap-1">
                {sources.map((source) => (
                  <span
                    key={source}
                    className="rounded bg-blue-50 px-1.5 py-0.5 text-[9px] text-blue-700"
                  >
                    {source}
                  </span>
                ))}
                {followUpScore != null && (
                  <span
                    className={`rounded px-1.5 py-0.5 text-[9px] ${
                      followUpScore === 1
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-rose-50 text-rose-700"
                    }`}
                  >
                    memory {followUpScore === 1 ? "used" : "missed"}
                  </span>
                )}
              </div>
            )}
          </div>
        );
      },
      sortValue: (row) => row.scores.entity_binding_match,
      searchValue: (row) => {
        const conversation = conversationDetails(row);
        return [
          ...stringArray(conversation.expected_entities),
          ...stringArray(conversation.actual_entities),
          ...stringArray(conversation.binding_sources),
        ].join(" ");
      },
    },
    {
      key: "retrieval",
      label: "retrieval",
      render: (row) => (
        <div className="text-[10px]">
          <div>{formatPercent(row.scores.recall_at_5)}</div>
          <div className="text-slate-400">
            {row.retrieved_ids.length}/{row.expected_ids.length} ids
          </div>
        </div>
      ),
      sortValue: (row) => row.scores.recall_at_5,
    },
    {
      key: "faithfulness",
      label: "faithful",
      render: (row) => formatPercent(row.scores.faithfulness),
      sortValue: (row) => row.scores.faithfulness,
    },
    {
      key: "violations",
      label: "violations",
      render: (row) => (
        <span className={row.violations.length ? "text-rose-600" : "text-emerald-600"}>
          {row.violations.length
            ? row.violations.map((item) => item.type).join(", ")
            : "None"}
        </span>
      ),
      sortValue: (row) => row.violations.length,
      searchValue: (row) => row.violations.map((item) => item.type).join(" "),
    },
    {
      key: "latency",
      label: "latency",
      render: (row) => formatLatency(row.latency_ms),
      sortValue: (row) => row.latency_ms,
    },
    {
      key: "trace",
      label: "trace_id",
      render: (row) =>
        row.trace_id ? (
          <Link
            className="mono text-[9px] font-semibold text-teal-700 hover:underline"
            to={`/traces?trace_id=${encodeURIComponent(row.trace_id)}`}
          >
            {truncate(row.trace_id, 12)}
          </Link>
        ) : (
          "—"
        ),
      sortValue: (row) => row.trace_id,
      searchValue: (row) => row.trace_id,
    },
    {
      key: "result",
      label: "result",
      render: (row) => (
        <StatusBadge
          status={row.passed == null ? row.status : row.passed ? "success" : "failed"}
        />
      ),
      sortValue: (row) => (row.passed == null ? null : Number(row.passed)),
      searchValue: (row) => (row.passed ? "success passed" : "failed"),
    },
  ];
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) => row.result_id}
      emptyTitle="No case-level results"
      defaultSort={{ key: "result", direction: "asc" }}
    />
  );
}

function conversationDetails(row: EvaluationCaseResult): Record<string, unknown> {
  const value = row.details?.conversation;
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

function textValue(value: unknown): string {
  return value == null ? "" : String(value);
}

function qualityAccent(value?: number | null): "teal" | "amber" | "rose" {
  if (value == null) return "amber";
  if (value >= 0.9) return "teal";
  if (value >= 0.75) return "amber";
  return "rose";
}

function riskAccent(value?: number | null): "teal" | "amber" | "rose" {
  if (value == null) return "amber";
  if (value <= 0.05) return "teal";
  if (value <= 0.15) return "amber";
  return "rose";
}
