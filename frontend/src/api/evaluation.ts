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

export interface RunEvaluationOptions {
  mode: "router" | "e2e" | "all";
  dataVersion: string;
  profile: "deterministic" | "production";
  datasetPath?: string;
  datasetName?: string;
  datasetVersion?: string;
}

export const runEvaluation = (options: RunEvaluationOptions) =>
  apiRequest("/evaluation/run", {
    method: "POST",
    timeoutMs: options.profile === "production" ? 1_800_000 : 300_000,
    body: JSON.stringify({
      mode: options.mode,
      data_version: options.dataVersion,
      profile: options.profile,
      dataset_path: options.datasetPath || null,
      dataset_name: options.datasetName || "dental_basic_eval",
      dataset_version: options.datasetVersion || "2.0",
    }),
  });
