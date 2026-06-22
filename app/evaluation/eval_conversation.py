from collections import Counter, defaultdict
from typing import Any

from app.retrieval.normalization import normalize_vietnamese

MEMORY_BINDING_SOURCES = {
    "conversation_state",
    "mixed_context",
    "same_turn_task",
}


def evaluate_conversation_case(
    *,
    case: dict[str, Any],
    trace_steps: list[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, Any], list[dict[str, Any]]]:
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    span_output = _step_output(trace_steps, "entity_span_extraction")
    binding_output = _step_output(trace_steps, "context_binding")
    canonical_output = _step_output(trace_steps, "task_canonicalization")
    memory_output = _step_output(trace_steps, "memory_load")
    planning_output = _step_output(trace_steps, "task_planning")
    legacy_bound_plan = binding_output.get("plan")
    plan = (
        canonical_output
        if canonical_output
        else legacy_bound_plan
        if isinstance(legacy_bound_plan, dict)
        else planning_output
    )
    tasks = plan.get("tasks") if isinstance(plan.get("tasks"), list) else []
    decisions = (
        binding_output.get("decisions")
        if isinstance(binding_output.get("decisions"), list)
        else []
    )
    spans = span_output.get("spans") if isinstance(span_output.get("spans"), list) else []
    expected_entities = [str(value) for value in case.get("expected_entities", []) if value]
    actual_entities = _actual_entities(tasks)
    entity_recall = _entity_recall(expected_entities, actual_entities)
    entity_match = (
        float(entity_recall == 1.0) if entity_recall is not None else None
    )

    requires_follow_up = bool(metadata.get("requires_follow_up_memory"))
    binding_sources = {
        str(decision.get("binding_source"))
        for decision in decisions
        if isinstance(decision, dict) and decision.get("binding_source")
    }
    memory_context_used = bool(binding_sources & MEMORY_BINDING_SOURCES) or any(
        task.get("resolution_status") == "resolved"
        and task.get("binding_source") == "conversation_state"
        for task in tasks
        if isinstance(task, dict)
    )
    follow_up_success = None
    if requires_follow_up:
        follow_up_success = float(
            memory_context_used and (entity_match != 0.0)
        )

    expects_multi_task = bool(metadata.get("expects_multi_task"))
    multi_task_match = (
        float(len(tasks) >= 2) if expects_multi_task else None
    )
    violations: list[dict[str, Any]] = []
    if entity_match == 0.0:
        violations.append(
            {
                "type": "entity_binding_mismatch",
                "expected": expected_entities,
                "actual": actual_entities,
            }
        )
    if follow_up_success == 0.0:
        violations.append(
            {
                "type": "follow_up_context_not_used",
                "expected": expected_entities,
                "actual": actual_entities,
            }
        )
    if multi_task_match == 0.0:
        violations.append(
            {
                "type": "multi_task_not_decomposed",
                "expected": "at least 2 tasks",
                "actual": len(tasks),
            }
        )

    details = {
        "scenario_key": metadata.get("scenario_key"),
        "scenario_title": metadata.get("scenario_title"),
        "turn_index": metadata.get("turn_index"),
        "requires_follow_up_memory": requires_follow_up,
        "expects_multi_task": expects_multi_task,
        "expected_entities": expected_entities,
        "actual_entities": actual_entities,
        "entity_spans": spans,
        "entity_span_provider": span_output.get("provider"),
        "entity_span_degraded": bool(span_output.get("degraded")),
        "binding_decisions": decisions,
        "binding_sources": sorted(binding_sources),
        "memory_turn_count": memory_output.get("turn_count"),
        "memory_state": memory_output.get("state"),
        "memory_context_used": memory_context_used,
        "planned_task_count": len(tasks),
        "planned_tasks": tasks,
    }
    scores = {
        "entity_binding_recall": entity_recall,
        "entity_binding_match": entity_match,
        "follow_up_memory": follow_up_success,
        "multi_task_match": multi_task_match,
    }
    return scores, details, violations


def evaluate_conversation(records: list[dict[str, Any]]) -> dict[str, Any]:
    entity_scores = _score_values(records, "entity_binding_match")
    follow_up_scores = _score_values(records, "follow_up_memory")
    multi_task_scores = _score_values(records, "multi_task_match")
    provider_counts: Counter[str] = Counter()
    degraded_count = 0
    provider_case_count = 0
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        conversation = item.get("details", {}).get("conversation", {})
        provider = conversation.get("entity_span_provider")
        if provider:
            provider_counts[str(provider)] += 1
            provider_case_count += 1
        if conversation.get("entity_span_degraded"):
            degraded_count += 1
        scenario_key = conversation.get("scenario_key")
        if scenario_key:
            scenarios[str(scenario_key)].append(item)

    scenario_rows = []
    for scenario_key, items in sorted(scenarios.items()):
        ordered = sorted(
            items,
            key=lambda item: (
                item.get("details", {})
                .get("conversation", {})
                .get("turn_index")
                or 0
            ),
        )
        scenario_rows.append(
            {
                "scenario_key": scenario_key,
                "scenario_title": (
                    ordered[0]
                    .get("details", {})
                    .get("conversation", {})
                    .get("scenario_title")
                ),
                "turn_count": len(ordered),
                "passed_turns": sum(item.get("passed") is True for item in ordered),
                "passed": all(item.get("passed") is True for item in ordered),
                "follow_up_success_rate": _average(
                    [
                        value
                        for item in ordered
                        if (
                            value := item.get("scores", {}).get("follow_up_memory")
                        )
                        is not None
                    ]
                ),
            }
        )
    return {
        "entity_binding_case_count": len(entity_scores),
        "entity_binding_accuracy": _average(entity_scores),
        "follow_up_case_count": len(follow_up_scores),
        "follow_up_success_rate": _average(follow_up_scores),
        "multi_task_case_count": len(multi_task_scores),
        "multi_task_success_rate": _average(multi_task_scores),
        "entity_span_provider_counts": dict(provider_counts),
        "entity_span_degraded_rate": (
            degraded_count / provider_case_count if provider_case_count else None
        ),
        "scenario_count": len(scenario_rows),
        "scenario_pass_rate": (
            sum(row["passed"] for row in scenario_rows) / len(scenario_rows)
            if scenario_rows
            else None
        ),
        "scenarios": scenario_rows,
    }


def _actual_entities(tasks: list[Any]) -> list[str]:
    values: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        entities = task.get("entity_names")
        if not isinstance(entities, list):
            entities = task.get("entities")
        if isinstance(entities, list):
            values.extend(str(value) for value in entities if value)
        selection = _selection(task)
        mentions = selection.get("mentions")
        if isinstance(mentions, list):
            values.extend(str(value) for value in mentions if value)
    return list(dict.fromkeys(values))


def _entity_recall(expected: list[str], actual: list[str]) -> float | None:
    if not expected:
        return None
    normalized_actual = [normalize_vietnamese(value) for value in actual]
    matched = sum(
        any(
            expected_value == actual_value
            or expected_value in actual_value
            or actual_value in expected_value
            for actual_value in normalized_actual
            if actual_value
        )
        for expected_value in (normalize_vietnamese(value) for value in expected)
        if expected_value
    )
    return matched / len(expected)


def _selection(task: dict[str, Any]) -> dict[str, Any]:
    selection = task.get("selection")
    return selection if isinstance(selection, dict) else {}


def _step_output(
    steps: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    output = next(
        (
            step.get("output")
            for step in steps
            if step.get("step_name") == name
        ),
        {},
    )
    return output if isinstance(output, dict) else {}


def _score_values(
    records: list[dict[str, Any]],
    score_name: str,
) -> list[float]:
    return [
        float(value)
        for item in records
        if (value := item.get("scores", {}).get(score_name)) is not None
    ]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
