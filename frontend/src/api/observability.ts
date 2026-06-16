import type { ApiList, EvaluationDiagnostics, SystemHealth } from "../types";
import { apiRequest } from "./client";

export interface RecentError {
  time: string;
  trace_id: string;
  step: string;
  error_type: string;
  message?: string;
}

export const getSystemHealth = () =>
  apiRequest<SystemHealth>("/api/observability/health", { timeoutMs: 5_000 });
export const getObservabilityMetrics = () =>
  apiRequest<Record<string, number>>("/api/observability/metrics");
export const getRecentErrors = () =>
  apiRequest<ApiList<RecentError>>("/api/observability/errors");
export const getObservabilityDiagnostics = () =>
  apiRequest<EvaluationDiagnostics>("/api/observability/diagnostics");
