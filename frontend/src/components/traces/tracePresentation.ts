import type { TraceStep } from "../../types";

export interface EntitySpanView {
  text: string;
  label: string;
  score?: number;
  source?: string;
}

export interface BindingDecisionView {
  taskId: string;
  intent: string;
  referenceMode: string;
  bindingSource: string;
  before: string[];
  after: string[];
  reason?: string;
}

export interface CanonicalTaskView {
  taskId: string;
  intent: string;
  plannerQuery?: string;
  effectiveQuery?: string;
  entityNames: string[];
  resolvedIds: string[];
  resolutionStatus?: string;
}

export interface GateView {
  name: string;
  status: string;
  validTaskIds: string[];
  violationCount: number;
}

export interface EntityFlowView {
  provider?: string;
  degraded: boolean;
  error?: string;
  spans: EntitySpanView[];
  decisions: BindingDecisionView[];
  canonicalTasks: CanonicalTaskView[];
  gates: GateView[];
  memoryTurnCount?: number;
  memoryState?: Record<string, unknown>;
}

export interface TokenStats {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  attempts?: Array<Record<string, unknown>>;
  estimated?: boolean;
  note?: string;
}

export function extractEntityFlow(steps: TraceStep[]): EntityFlowView | null {
  const spanStep = stepByName(steps, "entity_span_extraction");
  const bindingStep = stepByName(steps, "context_binding");
  const memoryStep = stepByName(steps, "memory_load");
  const canonicalStep = stepByName(steps, "task_canonicalization");
  const resolutionStep = stepByName(steps, "entity_resolution");
  const boundGateStep = stepByName(steps, "bound_task_consistency");
  const evidenceGateStep = stepByName(steps, "evidence_consistency");

  const spanOutput = stepOutput(spanStep);
  const bindingOutput = stepOutput(bindingStep);
  const memoryOutput = stepOutput(memoryStep);
  const canonicalOutput = stepOutput(canonicalStep);
  const resolutionOutput = stepOutput(resolutionStep);
  if (
    !spanOutput &&
    !bindingOutput &&
    !memoryOutput &&
    !canonicalOutput &&
    !resolutionOutput
  ) {
    return null;
  }

  const spans = arrayValue(spanOutput?.spans).flatMap((value) => {
    const record = recordValue(value);
    if (!record) return [];
    const text = stringValue(record.text);
    const label = stringValue(record.label);
    if (!text || !label) return [];
    return [{
      text,
      label,
      score: numberValue(record.score),
      source: stringValue(record.source),
    }];
  });

  const decisions = arrayValue(bindingOutput?.decisions).flatMap((value, index) => {
    const record = recordValue(value);
    if (!record) return [];
    return [{
      taskId: stringValue(record.task_id) || `task-${index + 1}`,
      intent: stringValue(record.intent) || "UNKNOWN",
      referenceMode: stringValue(record.reference_mode) || "unknown",
      bindingSource: stringValue(record.binding_source) || "none",
      before: stringList(
        record.rejected_planner_entities ?? record.entities_before,
      ),
      after: stringList(record.entity_names ?? record.entities_after),
      reason: arrayValue(record.reason_codes).length
        ? arrayValue(record.reason_codes).map(String).join(", ")
        : stringValue(record.reason),
    }];
  });

  const resolutions = new Map<string, Record<string, unknown>>();
  arrayValue(resolutionOutput?.resolutions).forEach((value) => {
    const record = recordValue(value);
    const taskId = stringValue(record?.task_id);
    if (record && taskId) resolutions.set(taskId, record);
  });

  const canonicalTasks = arrayValue(canonicalOutput?.tasks).flatMap((value, index) => {
    const record = recordValue(value);
    if (!record) return [];
    const taskId = stringValue(record.task_id) || `task-${index + 1}`;
    const resolution = resolutions.get(taskId);
    return [{
      taskId,
      intent: stringValue(record.intent) || "UNKNOWN",
      plannerQuery: stringValue(record.planner_query),
      effectiveQuery: stringValue(record.effective_query),
      entityNames: stringList(record.entity_names),
      resolvedIds: stringList(record.resolved_ids),
      resolutionStatus:
        stringValue(record.resolution_status) ||
        stringValue(resolution?.status),
    }];
  });

  const gates = [
    gateView("Bound task", stepOutput(boundGateStep)),
    gateView("Evidence", stepOutput(evidenceGateStep)),
  ].filter((value): value is GateView => value !== null);

  const bindingInput = stepInput(bindingStep);
  return {
    provider: stringValue(spanOutput?.provider),
    degraded: Boolean(spanOutput?.degraded),
    error: stringValue(spanOutput?.error),
    spans,
    decisions,
    canonicalTasks,
    gates,
    memoryTurnCount: numberValue(memoryOutput?.turn_count),
    memoryState:
      recordValue(memoryOutput?.state) ||
      recordValue(bindingInput?.state),
  };
}

export function tokenSummaryRows(steps: TraceStep[]) {
  return steps
    .filter((step) =>
      ["task_planning", "synthesis_generation"].includes(step.step_name),
    )
    .map((step) => {
      const stats = tokenStatsForStep(step);
      if (!stats) return null;
      return {
        label:
          step.step_name === "task_planning"
            ? "Task planning"
            : "Synthesis generation",
        ...stats,
      };
    })
    .filter(Boolean) as Array<TokenStats & { label: string }>;
}

export function tokenStatsForStep(step: TraceStep): TokenStats | null {
  if (step.step_name === "synthesis_generation") {
    return synthesisTokenStats(stepOutput(step), step.status);
  }
  if (step.step_name === "task_planning") {
    return planningTokenStats(stepOutput(step), stepInput(step));
  }
  return null;
}

function synthesisTokenStats(
  output?: Record<string, unknown>,
  status?: string,
): TokenStats | null {
  if (!output) return null;
  const attempts = arrayValue(output.attempts).filter(isRecord);
  const fromAttempts = tokenStatsFromAttempts(attempts);
  if (fromAttempts) {
    return {
      ...fromAttempts,
      attempts,
      note: "Đọc từ metadata LLM provider trong trace output.",
    };
  }
  const usage = recordValue(output.usage);
  const fromUsage = tokenStatsFromUsage(usage);
  if (fromUsage) {
    return {...fromUsage, note: "Đọc từ usage của provider."};
  }
  if (status === "skipped") {
    return null;
  }
  return {
    note: "Trace không ghi token usage cho bước synthesis này.",
  };
}

function planningTokenStats(
  output?: Record<string, unknown>,
  input?: Record<string, unknown>,
): TokenStats | null {
  if (!output) return null;
  const metadata = recordValue(output.metadata);
  const attempts = arrayValue(metadata?.attempts).filter(isRecord);
  const fromAttempts = tokenStatsFromAttempts(attempts);
  if (fromAttempts) {
    return {
      ...fromAttempts,
      attempts,
      note: "Đọc từ metadata PlannerLLM trong trace output.",
    };
  }
  const usage = recordValue(metadata?.usage) || recordValue(output.usage);
  const fromUsage = tokenStatsFromUsage(usage);
  if (fromUsage) {
    return {...fromUsage, note: "Đọc từ usage của provider."};
  }
  const promptChars = numberValue(metadata?.llm_prompt_chars);
  if (promptChars !== undefined) {
    const estimatedInput = Math.ceil(promptChars / 4);
    return {
      inputTokens: estimatedInput,
      totalTokens: estimatedInput,
      estimated: true,
      note:
        "Input tokens ước tính từ prompt chars/4; provider không trả output token usage.",
    };
  }
  const source = stringValue(output.source) || "unknown";
  const query = stringValue(input?.query);
  return {
    inputTokens: undefined,
    outputTokens: undefined,
    totalTokens: undefined,
    note:
      source === "heuristic"
        ? `Planner kết thúc bằng heuristic fallback; trace này không có token usage từ provider${query ? ` cho query “${query}”` : ""}.`
        : "Trace không ghi token usage của PlannerLLM.",
  };
}

function tokenStatsFromAttempts(
  attempts: Array<Record<string, unknown>>,
): TokenStats | null {
  if (!attempts.length) return null;
  const inputTokens = sumNumbers(
    attempts.map((attempt) => attempt.prompt_eval_count),
  );
  const outputTokens = sumNumbers(
    attempts.map((attempt) => attempt.eval_count),
  );
  if (inputTokens === undefined && outputTokens === undefined) return null;
  return {
    inputTokens,
    outputTokens,
    totalTokens: (inputTokens || 0) + (outputTokens || 0),
  };
}

function tokenStatsFromUsage(
  usage?: Record<string, unknown>,
): TokenStats | null {
  if (!usage) return null;
  const inputTokens =
    numberValue(usage.prompt_tokens) ?? numberValue(usage.input_tokens);
  const outputTokens =
    numberValue(usage.completion_tokens) ?? numberValue(usage.output_tokens);
  const totalTokens =
    numberValue(usage.total_tokens) ??
    (inputTokens === undefined && outputTokens === undefined
      ? undefined
      : (inputTokens || 0) + (outputTokens || 0));
  if (
    inputTokens === undefined &&
    outputTokens === undefined &&
    totalTokens === undefined
  ) {
    return null;
  }
  return {inputTokens, outputTokens, totalTokens};
}

function gateView(
  name: string,
  output?: Record<string, unknown>,
): GateView | null {
  if (!output) return null;
  return {
    name,
    status: stringValue(output.status) || "unknown",
    validTaskIds: stringList(output.valid_task_ids),
    violationCount: arrayValue(output.violations).length,
  };
}

function stepByName(
  steps: TraceStep[],
  name: string,
): TraceStep | undefined {
  return steps.find((step) => step.step_name === name);
}

function stepOutput(
  step?: TraceStep,
): Record<string, unknown> | undefined {
  return parseRecord(step?.output);
}

function stepInput(
  step?: TraceStep,
): Record<string, unknown> | undefined {
  return parseRecord(step?.input);
}

function parseRecord(value: unknown): Record<string, unknown> | undefined {
  if (isRecord(value)) return value;
  if (typeof value !== "string") return undefined;
  try {
    const parsed = JSON.parse(value);
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringValue(value: unknown): string | undefined {
  if (value == null) return undefined;
  const text = String(value).trim();
  return text || undefined;
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map(String).map((item) => item.trim()).filter(Boolean);
}

function sumNumbers(values: unknown[]) {
  const numbers = values
    .map(numberValue)
    .filter((value): value is number => value !== undefined);
  if (!numbers.length) return undefined;
  return numbers.reduce((total, value) => total + value, 0);
}

function numberValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function recordValue(
  value: unknown,
): Record<string, unknown> | undefined {
  return isRecord(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
