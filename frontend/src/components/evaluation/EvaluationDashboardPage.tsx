import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Braces,
  CheckCircle2,
  Database,
  Gauge,
  Image,
  Play,
  Route,
  ShieldCheck,
  Target,
  Timer,
} from "lucide-react";
import { useState } from "react";
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

export function EvaluationDashboardPage() {
  const client = useQueryClient();
  const notify = useToast();
  const [mode, setMode] = useState<"router" | "e2e" | "all">("all");
  const [profile, setProfile] = useState<"deterministic" | "production">("deterministic");
  const [dataVersion, setDataVersion] = useState("product-pdf-v2");
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
    queryKey: ["evaluation-results", summary.data?.latest_run_id],
    queryFn: () => listEvaluationResults(summary.data?.latest_run_id),
    enabled: !summary.isLoading,
  });
  const mutation = useMutation({
    mutationFn: () => runEvaluation(mode, dataVersion, profile),
    onSuccess: async () => {
      notify("Evaluation completed.");
      await Promise.all([
        client.invalidateQueries({ queryKey: ["evaluation-summary"] }),
        client.invalidateQueries({ queryKey: ["evaluation-runs"] }),
        client.invalidateQueries({ queryKey: ["evaluation-results"] }),
        client.invalidateQueries({ queryKey: ["evaluation-datasets"] }),
      ]);
    },
    onError: (error) => notify(error.message, "error"),
  });
  const metrics = summary.data;
  const cards = [
    ["E2E Pass Rate", formatPercent(metrics?.e2e_pass_rate), CheckCircle2],
    ["Router Accuracy", formatPercent(metrics?.router_accuracy), Route],
    ["Retrieval Recall@5", formatPercent(metrics?.retrieval_recall_at_5), Gauge],
    ["Faithfulness", formatPercent(metrics?.faithfulness_rate), ShieldCheck],
    ["Answer Correctness", formatPercent(metrics?.answer_correctness), Target],
    ["Safety Pass", formatPercent(metrics?.safety_pass_rate), ShieldCheck],
    ["JSON Validity", formatPercent(metrics?.json_validity_rate), Braces],
    ["Asset Resolve", formatPercent(metrics?.asset_resolve_rate), Image],
    ["Fallback Rate", formatPercent(metrics?.fallback_rate), Activity],
    ["No-result Rate", formatPercent(metrics?.no_result_rate), AlertTriangle],
    ["Unsupported Claims", formatPercent(metrics?.unsupported_claim_rate), AlertTriangle],
    ["p95 Latency", formatLatency(metrics?.p95_latency_ms), Timer],
  ] as const;
  return (
    <PageContainer
      title="Evaluation Dashboard"
      description="Grounded metrics, per-case failures và diagnostics được lưu trong PostgreSQL theo từng evaluation run."
      actions={
        <div className="flex items-center gap-2">
          <select
            className="control w-44 px-3 py-2 text-xs"
            value={mode}
            onChange={(event) => setMode(event.target.value as typeof mode)}
          >
            <option value="router">Router baseline</option>
            <option value="e2e">End-to-end</option>
            <option value="all">All grounded metrics</option>
          </select>
          <input
            className="control w-40 px-3 py-2 text-xs"
            value={dataVersion}
            onChange={(event) => setDataVersion(event.target.value)}
            aria-label="Data version"
          />
          <select
            className="control w-40 px-3 py-2 text-xs"
            value={profile}
            onChange={(event) => setProfile(event.target.value as typeof profile)}
          >
            <option value="deterministic">Deterministic profile</option>
            <option value="production">Production profile</option>
          </select>
          <button
            className="primary-button px-4 py-2.5 text-xs"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            <Play size={14} />
            {mutation.isPending ? "Running..." : "Run Evaluation"}
          </button>
        </div>
      }
    >
      {summary.isLoading ? (
        <LoadingState />
      ) : summary.isError ? (
        <ErrorState error={summary.error} onRetry={() => void summary.refetch()} />
      ) : (
        <>
          <div className="grid grid-cols-6 gap-4">
            {cards.map(([label, value, Icon], index) => (
              <MetricCard
                key={label}
                label={label}
                value={value}
                icon={Icon}
                accent={index >= 8 ? "rose" : index >= 3 ? "violet" : "teal"}
              />
            ))}
          </div>
          <CoveragePanel
            coverage={metrics?.coverage || {}}
            dataset={datasets.data?.items[0]}
          />
          <DiagnosticsPanel alerts={metrics?.diagnostics?.alerts || []} />
          <div className="mt-5 grid grid-cols-2 gap-5">
            <MetricHistory history={metrics?.history || []} />
            <PerIntentChart values={metrics?.per_intent || {}} />
          </div>
        </>
      )}
      <div className="mt-6 grid grid-cols-[0.75fr_1.6fr] gap-5">
        <section>
          <h2 className="mb-3 text-sm font-bold text-slate-800">Evaluation runs</h2>
          {runs.isError ? (
            <ErrorState error={runs.error} />
          ) : (
            <RunsTable rows={runs.data?.items || []} />
          )}
        </section>
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-800">Latest case results</h2>
            <span className="mono text-[10px] text-slate-400">
              {metrics?.latest_run_id || "No comprehensive run"}
            </span>
          </div>
          {results.isError ? (
            <ErrorState error={results.error} />
          ) : (
            <CaseResultsTable rows={results.data?.items || []} />
          )}
        </section>
      </div>
    </PageContainer>
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
    },
    { key: "dataset", label: "dataset", render: (row) => row.dataset_name || "—" },
    {
      key: "pass",
      label: "pass_rate",
      render: (row) => formatPercent(row.metrics?.e2e?.pass_rate),
    },
    {
      key: "coverage",
      label: "retrieval_gt",
      render: (row) => formatPercent(row.metrics?.coverage?.retrieval_ground_truth),
    },
    { key: "status", label: "status", render: (row) => <StatusBadge status={row.status} /> },
    { key: "time", label: "started", render: (row) => formatDate(row.started_at) },
  ];
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) => row.eval_run_id}
      emptyTitle="No evaluation runs"
    />
  );
}

function CaseResultsTable({ rows }: { rows: EvaluationCaseResult[] }) {
  const columns: Column<EvaluationCaseResult>[] = [
    {
      key: "query",
      label: "query",
      render: (row) => (
        <div title={row.answer_text || undefined}>
          <div>{truncate(row.query, 38)}</div>
          <div className="mt-1 text-[10px] text-slate-400">{truncate(row.answer_text, 58)}</div>
        </div>
      ),
    },
    {
      key: "intent",
      label: "expected → actual",
      render: (row) => (
        <span className="text-[10px]">
          {row.expected_intent || "—"} → {row.actual_intent || "ERROR"}
        </span>
      ),
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
    },
    {
      key: "faithfulness",
      label: "faithful",
      render: (row) => formatPercent(row.scores.faithfulness),
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
    },
    {
      key: "latency",
      label: "latency",
      render: (row) => formatLatency(row.latency_ms),
    },
    {
      key: "trace",
      label: "trace_id",
      render: (row) => <span className="mono text-[9px]">{truncate(row.trace_id, 12)}</span>,
    },
    {
      key: "result",
      label: "result",
      render: (row) => (
        <StatusBadge
          status={row.passed == null ? row.status : row.passed ? "success" : "failed"}
        />
      ),
    },
  ];
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(row) => row.result_id}
      emptyTitle="No case-level results"
    />
  );
}
