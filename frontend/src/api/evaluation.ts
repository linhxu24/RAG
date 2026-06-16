import type {
  ApiList,
  EvaluationCase,
  EvaluationCaseResult,
  EvaluationRun,
  EvaluationSummary,
} from "../types";
import { apiRequest } from "./client";

export interface EvaluationDataset {
  dataset_id: string;
  name: string;
  version: string;
  description?: string;
  content_hash?: string | null;
  metadata?: Record<string, unknown>;
}

export const getEvaluationSummary = () =>
  apiRequest<EvaluationSummary>("/api/evaluation/summary");
export const listEvaluationRuns = () =>
  apiRequest<ApiList<EvaluationRun>>("/api/evaluation/runs");
export const listEvaluationCases = () =>
  apiRequest<ApiList<EvaluationCase>>("/api/evaluation/cases");
export const listEvaluationResults = (evalRunId?: string | null) =>
  apiRequest<ApiList<EvaluationCaseResult>>(
    `/api/evaluation/results${evalRunId ? `?eval_run_id=${evalRunId}` : ""}`,
  );
export const listEvaluationDatasets = () =>
  apiRequest<ApiList<EvaluationDataset>>("/api/evaluation/datasets");
export const runEvaluation = (
  mode: "router" | "e2e" | "all",
  dataVersion: string,
  profile: "deterministic" | "production",
) =>
  apiRequest("/evaluation/run", {
    method: "POST",
    timeoutMs: profile === "production" ? 1_800_000 : 300_000,
    body: JSON.stringify({ mode, data_version: dataVersion, profile }),
  });
