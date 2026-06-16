import type { ApiList, TraceRecord } from "../types";
import { apiRequest } from "./client";

export const listTraces = () => apiRequest<ApiList<TraceRecord>>("/api/traces");
export const getTrace = (id: string) =>
  apiRequest<TraceRecord>(`/api/traces/${id}`);
