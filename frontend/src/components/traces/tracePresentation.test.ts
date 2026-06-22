import { describe, expect, it } from "vitest";

import type { TraceStep } from "../../types";
import {
  extractEntityFlow,
  tokenSummaryRows,
} from "./tracePresentation";

const productListTraceSteps: TraceStep[] = [
  {
    step_name: "memory_load",
    status: "success",
    latency_ms: 48,
    output: {
      turn_count: 16,
      state: {
        active_topic: "OralWave Pro S2 Electric Toothbrush",
        active_domain: "product",
        active_product_names: ["OralWave Pro S2 Electric Toothbrush"],
        last_intents: ["PRODUCT_LIST", "SERVICE_DETAIL"],
      },
    },
  },
  {
    step_name: "task_planning",
    status: "success",
    latency_ms: 143_984,
    input: {
      query: "Cho tôi danh sách sản phẩm đang có",
      history_turns: 16,
    },
    output: {
      source: "heuristic",
      metadata: {},
      tasks: [
        {
          task_id: "t1",
          intent: "PRODUCT_LIST",
          planner_query: "Cho tôi danh sách sản phẩm đang có",
          planner_entities: [],
        },
      ],
    },
  },
  {
    step_name: "entity_span_extraction",
    status: "success",
    latency_ms: 2_369,
    output: {
      provider: "gliner",
      degraded: false,
      spans: [],
    },
  },
  {
    step_name: "context_binding",
    status: "success",
    latency_ms: 0,
    output: {
      decisions: [
        {
          task_id: "t1",
          intent: "PRODUCT_LIST",
          entity_names: [],
          rejected_planner_entities: [],
          binding_source: "task_filters",
          reference_mode: "no_entity",
          reason_codes: ["filter_only_intent"],
        },
      ],
    },
  },
  {
    step_name: "entity_resolution",
    status: "success",
    latency_ms: 0,
    output: {
      resolutions: [
        {
          task_id: "t1",
          status: "not_applicable",
          entity_names: [],
          resolved_ids: [],
        },
      ],
    },
  },
  {
    step_name: "task_canonicalization",
    status: "success",
    latency_ms: 0,
    output: {
      tasks: [
        {
          task_id: "t1",
          intent: "PRODUCT_LIST",
          planner_query: "Cho tôi danh sách sản phẩm đang có",
          effective_query: "Cho tôi danh sách sản phẩm đang có",
          entity_names: [],
          resolved_ids: [],
          resolution_status: "not_applicable",
        },
      ],
    },
  },
  {
    step_name: "bound_task_consistency",
    status: "success",
    latency_ms: 0,
    output: {
      status: "pass",
      valid_task_ids: ["t1"],
      violations: [],
    },
  },
  {
    step_name: "evidence_consistency",
    status: "success",
    latency_ms: 0,
    output: {
      status: "pass",
      valid_task_ids: ["t1"],
      violations: [],
    },
  },
  {
    step_name: "synthesis_generation",
    status: "skipped",
    latency_ms: 0,
    output: { reason: "Evidence synthesis disabled or no evidence" },
  },
];

describe("trace presentation", () => {
  it("shows missing planner token metadata instead of hiding token usage", () => {
    const rows = tokenSummaryRows(productListTraceSteps);

    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      label: "Task planning",
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    });
    expect(rows[0].note).toContain("heuristic fallback");
  });

  it("builds entity, memory, canonical task and gate flow for filter-only tasks", () => {
    const flow = extractEntityFlow(productListTraceSteps);

    expect(flow).not.toBeNull();
    expect(flow?.provider).toBe("gliner");
    expect(flow?.memoryTurnCount).toBe(16);
    expect(flow?.memoryState?.active_domain).toBe("product");
    expect(flow?.decisions[0]).toMatchObject({
      taskId: "t1",
      intent: "PRODUCT_LIST",
      referenceMode: "no_entity",
      bindingSource: "task_filters",
      before: [],
      after: [],
      reason: "filter_only_intent",
    });
    expect(flow?.canonicalTasks[0]).toMatchObject({
      taskId: "t1",
      intent: "PRODUCT_LIST",
      effectiveQuery: "Cho tôi danh sách sản phẩm đang có",
      entityNames: [],
      resolvedIds: [],
      resolutionStatus: "not_applicable",
    });
    expect(flow?.gates).toEqual([
      {
        name: "Bound task",
        status: "pass",
        validTaskIds: ["t1"],
        violationCount: 0,
      },
      {
        name: "Evidence",
        status: "pass",
        validTaskIds: ["t1"],
        violationCount: 0,
      },
    ]);
  });
});
