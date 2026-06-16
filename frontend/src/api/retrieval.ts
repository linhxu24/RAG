import type { RetrievalDebugResponse } from "../types";
import { apiRequest } from "./client";

export interface RetrievalOptions {
  query: string;
  use_structured: boolean;
  use_dense: boolean;
  use_sparse: boolean;
  use_rrf: boolean;
  use_reranker: boolean;
  use_hyde: boolean;
}

export const runRetrievalDebug = (options: RetrievalOptions) =>
  apiRequest<RetrievalDebugResponse>("/api/retrieval/debug", {
    method: "POST",
    timeoutMs: 130_000,
    body: JSON.stringify(options),
  });
