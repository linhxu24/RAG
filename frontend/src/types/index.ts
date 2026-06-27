export type StatusValue =
  | "success"
  | "failed"
  | "running"
  | "pending"
  | "active"
  | "archived"
  | "connected"
  | "disconnected"
  | "enabled"
  | "disabled"
  | "available"
  | "missing"
  | "configured"
  | "completed"
  | "review_required"
  | "unknown"
  | string;

export interface AssetRecord {
  asset_id: string;
  asset_token?: string;
  token?: string;
  asset_type?: string;
  type?: string;
  doc_id?: string;
  chunk_id?: string | null;
  local_path?: string | null;
  public_url?: string | null;
  url?: string | null;
  status?: StatusValue;
  local_file_exists?: boolean;
  used_in?: Record<string, unknown>;
}

export interface SourceRecord {
  source_type?: string;
  source_id?: string;
  doc_id?: string | null;
  file_name?: string;
  page_number?: number | null;
  chunk_id?: string | null;
  row_id?: string | null;
}

export interface ChatAnswer {
  text: string;
  assets: AssetRecord[];
  items: Array<Record<string, unknown>>;
  sources: SourceRecord[];
  missing_assets?: string[];
}

export interface ChatDebugInfo {
  enabled?: boolean;
  intent?: string;
  confidence?: number;
  answer_type?: string;
  total_latency_ms?: number;
  latency_ms?: number;
  retrieval_used?: boolean;
  json_valid?: boolean;
  failed_step?: string | null;
  chunks_used?: string[];
  rows_used?: string[];
  assets_returned?: string[];
  model_used?: string;
  retrieval?: Record<string, unknown>;
}

export interface ChatSuggestion {
  suggestion_id: string;
  type: "next_question" | "recommendation";
  label: string;
  query: string;
  target_intent: string;
  reason_code: string;
}

export interface ChatResponse {
  trace_id: string;
  intent?: string;
  answer_type?: string;
  degraded?: boolean;
  answer?: ChatAnswer;
  message?: ChatAnswer;
  suggestions?: ChatSuggestion[];
  debug?: ChatDebugInfo | null;
  error?: {
    type?: string;
    message: string;
    failed_step?: string;
  };
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: string;
  response?: ChatResponse;
  error?: string;
}

export interface DocumentRecord {
  doc_id: string;
  file_name: string;
  file_type?: string | null;
  status: StatusValue;
  version: number;
  detected_document_type?: string | null;
  document_type_confidence?: number | null;
  chunks: number;
  tables: number;
  assets: number;
  created_at?: string | null;
  updated_at?: string | null;
  metadata?: Record<string, unknown>;
  chunk_items?: Array<Record<string, unknown>>;
  table_items?: Array<Record<string, unknown>>;
  asset_items?: AssetRecord[];
  latest_ingestion_run?: IngestionRun | null;
}

export interface IngestionStep {
  step_name: string;
  status: StatusValue;
  latency_ms: number;
  error_message?: string | null;
}

export interface IngestionRun {
  run_id: string;
  doc_id: string;
  file_name?: string | null;
  status: StatusValue;
  started_at?: string | null;
  ended_at?: string | null;
  total_latency_ms: number;
  total_chunks: number;
  total_tables: number;
  total_table_rows: number;
  total_assets: number;
  total_embeddings: number;
  error_message?: string | null;
  quality_report?: Record<string, any>;
  timeline: IngestionStep[];
}

export interface TraceStep {
  step_id?: string;
  step_name: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  latency_ms: number;
  status: StatusValue;
  error_message?: string | null;
  created_at?: string | null;
}

export interface TraceRecord {
  trace_id: string;
  session_id?: string | null;
  query?: string;
  intent?: string | null;
  confidence?: number | null;
  total_latency_ms?: number | null;
  status: StatusValue;
  created_at?: string;
  final_answer?: Record<string, any> | null;
  error?: string | null;
  failed_step?: string | null;
  steps?: TraceStep[];
}

export interface EvaluationSummary {
  e2e_success_rate?: number | null;
  e2e_pass_rate?: number | null;
  router_accuracy?: number | null;
  retrieval_recall_at_5?: number | null;
  retrieval_mrr_at_10?: number | null;
  ndcg_at_10?: number | null;
  retrieval_ground_truth_coverage?: number | null;
  json_validity_rate?: number | null;
  schema_pass_rate?: number | null;
  answer_correctness?: number | null;
  faithfulness_rate?: number | null;
  unsupported_claim_rate?: number | null;
  safety_pass_rate?: number | null;
  asset_resolve_rate?: number | null;
  asset_ground_truth_coverage?: number | null;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  p99_latency_ms?: number | null;
  fallback_rate?: number | null;
  no_result_rate?: number | null;
  clarification_rate?: number | null;
  entity_binding_accuracy?: number | null;
  follow_up_success_rate?: number | null;
  multi_task_success_rate?: number | null;
  entity_span_degraded_rate?: number | null;
  scenario_pass_rate?: number | null;
  latest_run_id?: string | null;
  coverage?: Record<string, number>;
  per_intent?: Record<string, {
    case_count: number;
    pass_rate: number;
    router_accuracy: number;
    retrieval_recall_at_5?: number | null;
    faithfulness_rate?: number | null;
  }>;
  diagnostics?: EvaluationDiagnostics;
  history: Array<Record<string, any>>;
  confusion_matrix?: Record<string, Record<string, number>>;
  conversation?: ConversationEvaluationMetrics;
}

export interface ConversationScenarioMetric {
  scenario_key: string;
  scenario_title?: string | null;
  turn_count: number;
  passed_turns: number;
  passed: boolean;
  follow_up_success_rate?: number | null;
}

export interface ConversationEvaluationMetrics {
  entity_binding_case_count?: number;
  entity_binding_accuracy?: number | null;
  follow_up_case_count?: number;
  follow_up_success_rate?: number | null;
  multi_task_case_count?: number;
  multi_task_success_rate?: number | null;
  entity_span_provider_counts?: Record<string, number>;
  entity_span_degraded_rate?: number | null;
  scenario_count?: number;
  scenario_pass_rate?: number | null;
  scenarios?: ConversationScenarioMetric[];
}

export interface EvaluationRun {
  eval_run_id: string;
  dataset_id?: string | null;
  dataset_name?: string | null;
  pipeline_version: string;
  data_version: string;
  started_at?: string | null;
  ended_at?: string | null;
  status: StatusValue;
  metrics: Record<string, any>;
  config_snapshot?: Record<string, any>;
}

export interface EvaluationCase {
  case_id: string;
  dataset_id: string;
  query: string;
  expected_intent?: string | null;
  actual_intent?: string | null;
  expected_doc_ids?: string[];
  expected_chunk_ids?: string[];
  expected_row_ids?: string[];
  expected_asset_ids?: string[];
  retrieved_source?: string[];
  pass?: boolean;
  trace_id?: string;
  metadata?: Record<string, any>;
}

export interface EvaluationViolation {
  type: string;
  message?: string;
  values?: unknown[];
  expected?: unknown;
  actual?: unknown;
}

export interface EvaluationCaseResult {
  result_id: string;
  eval_run_id: string;
  case_id?: string | null;
  trace_id?: string | null;
  query: string;
  expected_intent?: string | null;
  actual_intent?: string | null;
  status: StatusValue;
  passed?: boolean | null;
  latency_ms?: number | null;
  expected_ids: string[];
  retrieved_ids: string[];
  answer_text?: string | null;
  scores: Record<string, number | null>;
  violations: EvaluationViolation[];
  details: Record<string, any>;
  error_message?: string | null;
  created_at?: string | null;
}

export interface DiagnosticAlert {
  severity: string;
  code: string;
  title: string;
  detail: string;
  value?: number;
  threshold?: number;
}

export interface EvaluationDiagnostics {
  alerts: DiagnosticAlert[];
  latency_by_stage: Record<string, {
    count: number;
    average_ms: number;
    p95_ms: number;
    max_ms: number;
  }>;
  failed_steps: Record<string, number>;
  fallback_rate?: number;
  no_result_rate?: number;
  failed_case_rate?: number;
}

export interface SystemHealth {
  status: StatusValue;
  postgresql?: Record<string, any>;
  pgvector?: Record<string, any>;
  ollama?: Record<string, any>;
  embedding_model?: Record<string, any>;
  reranker?: Record<string, any>;
  assets?: Record<string, any>;
  database?: Record<string, any>;
}

export interface RetrievalItem {
  id: string;
  type: string;
  content: string;
  score: number;
  rank: number;
  source?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  ranks?: Record<string, number>;
  canonical_key?: string | null;
}

export interface RetrievalDebugResponse {
  router: Record<string, any>;
  entities: Record<string, any>;
  plan: Record<string, any>;
  rewrite: Record<string, any>;
  structured: RetrievalItem[];
  dense: RetrievalItem[];
  sparse: RetrievalItem[];
  rrf: RetrievalItem[];
  reranker: RetrievalItem[];
  reranker_used: boolean;
  final_context: {
    items: Array<Record<string, any>>;
    total_chars: number;
  };
}

export interface ProductRecord {
  product_id: string;
  name: string;
  category?: string | null;
  description?: string | null;
  price?: number | null;
  quantity?: number | null;
  asset_id?: string | null;
  source_doc_id: string;
  status: StatusValue;
  version: number;
}

export interface ServiceRecord {
  service_id: string;
  name: string;
  description?: string | null;
  duration_minutes?: number | null;
  price?: number | null;
  source_doc_id: string;
  status: StatusValue;
  version: number;
}

export interface ApiList<T> {
  items: T[];
}
