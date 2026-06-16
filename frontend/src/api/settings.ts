import { apiRequest } from "./client";

export interface ControlCenterSettings {
  api_environment: string;
  router_model: string;
  generation_model: string;
  embedding_model: string;
  reranker_model: string;
  top_k_dense: number;
  top_k_sparse: number;
  top_k_final: number;
  rrf_k: number;
  reranker_enabled: boolean;
  hyde_enabled: boolean;
  confidence_threshold: number;
  assets_dir: string;
  public_assets_base_url: string;
  max_context_tokens: number;
  json_retry_count: number;
  read_only: boolean;
}

export const getSettings = () =>
  apiRequest<ControlCenterSettings>("/api/settings");
