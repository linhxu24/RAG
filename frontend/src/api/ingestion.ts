import type { ApiList, IngestionRun } from "../types";
import { apiRequest } from "./client";

export const listIngestionRuns = () =>
  apiRequest<ApiList<IngestionRun>>("/api/ingestion/runs");
export const getIngestionRun = (id: string) =>
  apiRequest<IngestionRun>(`/api/ingestion/runs/${id}`);
export const getIngestionSummary = () =>
  apiRequest<Record<string, number>>("/api/ingestion/summary");
