from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.run_conversation_scenarios import (
    BASE_URL,
    FAIL_MARK,
    PASS_MARK,
    REQUEST_TIMEOUT_SECONDS,
    build_summary,
    check_rate,
    contains_text,
    filter_scenarios,
    load_scenarios,
    normalize_text,
    post_json,
    resolve_input_path,
    score_turn,
    write_results,
)

DEFAULT_INPUT = "eval_datasets/dental_conversation_scenarios_v2.json"
DEFAULT_OUTPUT = "eval_results/conversation_scenario_results_v2.json"

V2_CHECK_KEYS = (
    "normalize_resolved",
    "entity_inherited_not_reresolved",
    "entity_switched_correctly",
    "no_stale_entity",
    "cardinality_two_resolved",
    "cardinality_one_after_compare",
    "implicit_ref_resolved",
    "multi_task_both_pass_gate",
    "entity_survives_no_entity_turn",
    "clarification_triggered",
    "resolved_after_clarification",
    "no_entity_leaked_to_no_entity_intent",
    "chitchat_clean_state",
)

NORMALIZE_TARGETS = {"normalize_for_match", "normalize_case_insensitive", "initial_entity_resolve"}
INHERIT_TARGETS = {
    "entity_inheritance_after_resolve",
    "deep_follow_up_memory",
    "follow_up_same_entity",
    "follow_up_after_multi_task",
    "follow_up_after_clarification_resolved",
    "faq_inherits_entity_post_clarification",
}
NO_STALE_TARGETS = {
    "follow_up_after_switch",
    "switch_back_to_first_entity",
    "faq_after_compare_inherits_both_entities",
}
ENTITY_SWITCH_TARGETS = {"entity_switch_to_new"}
CLARIFICATION_TARGETS = {"ambiguous_entity_triggers_clarification"}
RESOLVED_AFTER_CLARIFICATION_TARGETS = {"user_provides_exact_name_after_clarification"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2 dental conversation scenarios against a live backend."
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario_key values to run. Default: all scenarios.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to v2 scenarios JSON.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write v2 results JSON.",
    )
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Override BACKEND_URL. Default: BACKEND_URL env var or http://localhost:8000.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to v1 results JSON for delta comparison.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full debug snapshot.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Per-turn HTTP timeout. Use 0 to avoid setting a client timeout.",
    )
    return parser.parse_args()


def extract_debug(response: dict[str, Any]) -> dict[str, Any]:
    raw_debug = (
        response.get("debug")
        or response.get("trace")
        or response.get("orchestration")
        or {}
    )
    if not isinstance(raw_debug, dict):
        raw_debug = {}
    retrieval = (
        raw_debug.get("retrieval")
        if isinstance(raw_debug.get("retrieval"), dict)
        else {}
    )
    bound_plan = (
        retrieval.get("bound_plan")
        if isinstance(retrieval.get("bound_plan"), dict)
        else {}
    )
    planner_plan = (
        retrieval.get("planner_plan")
        if isinstance(retrieval.get("planner_plan"), dict)
        else {}
    )
    bound_gate = (
        retrieval.get("bound_gate")
        if isinstance(retrieval.get("bound_gate"), dict)
        else {}
    )
    debug_absent = not _has_debug_payload(raw_debug)
    binding = raw_debug.get("binding") if isinstance(raw_debug.get("binding"), dict) else {}
    consistency = (
        raw_debug.get("consistency")
        if isinstance(raw_debug.get("consistency"), dict)
        else {}
    )
    bound_task = (
        raw_debug.get("bound_task")
        or binding.get("bound_task")
        or _primary_bound_task(bound_plan, response)
        or {}
    )
    if not isinstance(bound_task, dict):
        bound_task = {}
    decision = (
        raw_debug.get("decision")
        or binding.get("decision")
        or _decision_from_bound_task(bound_task)
        or {}
    )
    if not isinstance(decision, dict):
        decision = {}
    planned_tasks = (
        raw_debug.get("planned_tasks")
        or raw_debug.get("tasks")
        or planner_plan.get("tasks")
        or bound_plan.get("tasks")
        or []
    )
    if not isinstance(planned_tasks, list):
        planned_tasks = []
    violations = (
        raw_debug.get("violations")
        or consistency.get("violations")
        or bound_gate.get("violations")
        or []
    )
    if not isinstance(violations, list):
        violations = []
    resolution_status = (
        raw_debug.get("resolution_status")
        or binding.get("resolution_status")
        or bound_task.get("resolution_status")
        or "unknown"
    )
    resolution_source = (
        raw_debug.get("resolution_source")
        or binding.get("source")
        or binding.get("resolution_source")
        or bound_task.get("binding_source")
        or "unknown"
    )
    gate_status = (
        raw_debug.get("gate_status")
        or consistency.get("status")
        or bound_gate.get("status")
        or "unknown"
    )
    return {
        "debug_absent": debug_absent,
        "raw_debug": raw_debug,
        "resolution_status": resolution_status,
        "resolution_source": resolution_source,
        "gate_status": gate_status,
        "bound_task": bound_task,
        "planned_tasks": planned_tasks,
        "decision": decision,
        "violations": violations,
    }


def _has_debug_payload(raw_debug: dict[str, Any]) -> bool:
    if not raw_debug:
        return False
    if raw_debug.get("enabled") is False and len(raw_debug) == 1:
        return False
    debug_keys = {
        "resolution_status",
        "resolution_source",
        "gate_status",
        "bound_task",
        "planned_tasks",
        "tasks",
        "decision",
        "violations",
        "binding",
        "consistency",
        "retrieval",
    }
    return any(key in raw_debug for key in debug_keys)


def _primary_bound_task(
    bound_plan: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    tasks = bound_plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return {}
    response_intent = response.get("intent")
    matching = [
        task
        for task in tasks
        if isinstance(task, dict) and task.get("intent") == response_intent
    ]
    candidates = matching or [task for task in tasks if isinstance(task, dict)]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda item: item.get("priority", 1))[0]


def _decision_from_bound_task(bound_task: dict[str, Any]) -> dict[str, Any]:
    if not bound_task:
        return {}
    return {
        "reference_mode": bound_task.get("reference_mode"),
        "binding_source": bound_task.get("binding_source"),
        "entity_names": bound_task.get("entity_names"),
        "resolved_ids": bound_task.get("resolved_ids"),
    }


def score_v2_checks(
    turn: dict[str, Any],
    response: dict[str, Any],
    debug: dict[str, Any],
    previous_expected_entities: list[str],
) -> tuple[dict[str, str], list[str], list[str]]:
    checks = {key: "skip" for key in V2_CHECK_KEYS}
    notes: list[str] = []
    failures: list[str] = []
    metadata = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
    target = str(metadata.get("test_target") or "")
    expected_entities = [str(item) for item in turn.get("expected_entities") or []]

    if debug["debug_absent"]:
        notes.append("debug field absent — enable debug mode in backend")
        return checks, notes, failures

    bound_task = debug["bound_task"]
    decision = debug["decision"]
    answer_text = _answer_text(response)
    entity_names = _string_list(bound_task.get("entity_names"))
    resolved_ids = _string_list(bound_task.get("resolved_ids"))
    violations = debug["violations"]

    if target in NORMALIZE_TARGETS:
        checks["normalize_resolved"] = _pass_fail(
            debug["resolution_status"] == "resolved"
            and _contains_all_entities(entity_names, expected_entities)
        )
        _append_failure(failures, checks, "normalize_resolved", target)

    if target in INHERIT_TARGETS:
        checks["entity_inherited_not_reresolved"] = _pass_fail(
            debug["resolution_status"] == "resolved"
            and debug["resolution_source"] == "conversation_state"
        )
        _append_failure(failures, checks, "entity_inherited_not_reresolved", target)

    if target in ENTITY_SWITCH_TARGETS:
        checks["entity_switched_correctly"] = _pass_fail(
            _same_entities(entity_names, expected_entities)
            and not _overlaps_entities(entity_names, previous_expected_entities)
        )
        _append_failure(failures, checks, "entity_switched_correctly", target)

    if target in NO_STALE_TARGETS:
        checks["no_stale_entity"] = _pass_fail(_same_entities(entity_names, expected_entities))
        _append_failure(failures, checks, "no_stale_entity", target)

    if target == "compare_cardinality_two_entities":
        checks["cardinality_two_resolved"] = _pass_fail(
            len(resolved_ids) == 2 and _gate_pass(debug)
        )
        _append_failure(failures, checks, "cardinality_two_resolved", target)

    if target == "detail_after_compare_one_entity":
        checks["cardinality_one_after_compare"] = _pass_fail(
            len(resolved_ids) == 1 and _same_entities(entity_names, expected_entities)
        )
        _append_failure(failures, checks, "cardinality_one_after_compare", target)

    if target == "implicit_reference_to_second_entity":
        checks["implicit_ref_resolved"] = _pass_fail(
            _same_entities(entity_names, ["SilkLine Waxed Dental Floss"])
            and decision.get("reference_mode") == "implicit"
        )
        _append_failure(failures, checks, "implicit_ref_resolved", target)

    if target == "multi_task_service_detail_plus_faq":
        checks["multi_task_both_pass_gate"] = _pass_fail(
            len(debug["planned_tasks"]) == 2
            and _gate_pass(debug)
            and not _has_violation(violations, "effective_query_missing_entity")
        )
        _append_failure(failures, checks, "multi_task_both_pass_gate", target)

    if target == "re_reference_after_clinic_info_turn":
        checks["entity_survives_no_entity_turn"] = _pass_fail(
            debug["resolution_source"] == "conversation_state"
            and _contains_all_entities(entity_names, expected_entities)
        )
        _append_failure(failures, checks, "entity_survives_no_entity_turn", target)

    if target in CLARIFICATION_TARGETS:
        checks["clarification_triggered"] = _pass_fail(
            debug["gate_status"] == "clarify"
            or bool(bound_task.get("clarification_required"))
            or _looks_like_clarification(answer_text)
        )
        _append_failure(failures, checks, "clarification_triggered", target)

    if target in RESOLVED_AFTER_CLARIFICATION_TARGETS:
        checks["resolved_after_clarification"] = _pass_fail(
            debug["resolution_status"] == "resolved" and _gate_pass(debug)
        )
        _append_failure(failures, checks, "resolved_after_clarification", target)

    if target == "intent_switch_no_entity_after_entity_session":
        checks["no_entity_leaked_to_no_entity_intent"] = _pass_fail(
            entity_names == [] and resolved_ids == []
        )
        _append_failure(failures, checks, "no_entity_leaked_to_no_entity_intent", target)

    if target == "graceful_end_after_full_flow":
        checks["chitchat_clean_state"] = _pass_fail(
            response.get("degraded") is False and entity_names == []
        )
        _append_failure(failures, checks, "chitchat_clean_state", target)

    return checks, notes, failures


def build_debug_snapshot(debug: dict[str, Any]) -> dict[str, Any]:
    bound_task = debug["bound_task"]
    return {
        "resolution_status": debug["resolution_status"],
        "resolution_source": debug["resolution_source"],
        "gate_status": debug["gate_status"],
        "entity_names": _string_list(bound_task.get("entity_names")),
        "resolved_ids": _string_list(bound_task.get("resolved_ids")),
        "violations": debug["violations"],
    }


def _answer_text(response: dict[str, Any]) -> str:
    answer = response.get("answer")
    if isinstance(answer, dict) and isinstance(answer.get("text"), str):
        return answer["text"]
    return ""


def _pass_fail(value: bool) -> str:
    return "pass" if value else "fail"


def _append_failure(
    failures: list[str],
    checks: dict[str, str],
    key: str,
    target: str,
) -> None:
    if checks.get(key) == "fail":
        failures.append(f"{key}: failed for test_target={target}")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item)]


def _contains_all_entities(actual: list[str], expected: list[str]) -> bool:
    return all(
        any(_entity_equal(actual_entity, expected_entity) for actual_entity in actual)
        for expected_entity in expected
    )


def _same_entities(actual: list[str], expected: list[str]) -> bool:
    if len(actual) != len(expected):
        return False
    return _contains_all_entities(actual, expected) and _contains_all_entities(expected, actual)


def _overlaps_entities(actual: list[str], previous: list[str]) -> bool:
    return any(
        _entity_equal(actual_entity, previous_entity)
        for actual_entity in actual
        for previous_entity in previous
    )


def _entity_equal(left: str, right: str) -> bool:
    return normalize_text(left) == normalize_text(right)


def _gate_pass(debug: dict[str, Any]) -> bool:
    if debug["gate_status"] == "pass":
        return True
    consistency = debug["raw_debug"].get("consistency")
    if isinstance(consistency, dict):
        reports = consistency.get("reports") or consistency.get("task_reports")
        if isinstance(reports, list) and reports:
            return all(
                isinstance(report, dict) and report.get("status") == "pass"
                for report in reports
            )
    return False


def _has_violation(violations: list[Any], code: str) -> bool:
    return any(isinstance(item, dict) and item.get("code") == code for item in violations)


def _looks_like_clarification(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        contains_text(normalized, phrase)
        for phrase in (
            "ban vui long",
            "ban muon",
            "y ban la",
            "noi ro",
            "lam ro",
            "chon",
            "san pham nao",
            "dich vu nao",
        )
    )


def build_v2_summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_summary(scenarios)
    turns = [turn for scenario in scenarios for turn in scenario["turns"]]
    summary.update(
        {
            "normalize_check_accuracy": v2_rate(turns, ["normalize_resolved"]),
            "entity_switch_accuracy": v2_rate(
                turns,
                [
                    "entity_switched_correctly",
                    "no_stale_entity",
                    "cardinality_one_after_compare",
                    "implicit_ref_resolved",
                ],
            ),
            "clarification_flow_accuracy": v2_rate(
                turns,
                ["clarification_triggered", "resolved_after_clarification"],
            ),
            "gate_regression_clean": not any_effective_query_violation(turns),
        }
    )
    return summary


def v2_rate(turns: list[dict[str, Any]], keys: list[str]) -> float | None:
    applicable: list[str] = []
    for turn in turns:
        checks = turn.get("v2_checks") or {}
        for key in keys:
            status = checks.get(key)
            if status != "skip":
                applicable.append(status)
    if not applicable:
        return None
    return round(sum(1 for status in applicable if status == "pass") / len(applicable), 4)


def any_effective_query_violation(turns: list[dict[str, Any]]) -> bool:
    return any(
        violation.get("code") == "effective_query_missing_entity"
        for turn in turns
        for violation in turn.get("debug_snapshot", {}).get("violations", [])
        if isinstance(violation, dict)
    )


def build_bug_regression(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    affected_turns = []
    debug_present = False
    for scenario in scenarios:
        for turn in scenario["turns"]:
            if "debug field absent" not in " ".join(turn.get("v2_notes", [])):
                debug_present = True
            for violation in turn.get("debug_snapshot", {}).get("violations", []):
                if isinstance(violation, dict) and violation.get("code") == (
                    "effective_query_missing_entity"
                ):
                    affected_turns.append(
                        {
                            "scenario_key": scenario["scenario_key"],
                            "turn_index": turn["turn_index"],
                            "query": turn["query"],
                        }
                    )
    occurrences = len(affected_turns)
    return {
        "effective_query_missing_entity_removed": {
            "description": "Check rằng bug cũ không còn xuất hiện",
            "method": (
                "Scan toàn bộ violations trong debug output — không được có "
                "code = effective_query_missing_entity"
            ),
            "result": "skip" if not debug_present else "pass" if occurrences == 0 else "fail",
            "occurrences": occurrences,
            "affected_turns": affected_turns,
        },
        "normalize_unified": build_normalize_regression(scenarios),
        "entity_stale_binding_fixed": build_stale_binding_regression(scenarios),
    }


def build_normalize_regression(scenarios: list[dict[str, Any]]) -> dict[str, str]:
    scenario = _scenario_by_key(scenarios, "entity_resolve_after_normalize_fix")
    statuses = []
    if scenario:
        for turn in scenario["turns"][:3]:
            checks = turn.get("v2_checks", {})
            for key in ("normalize_resolved", "entity_inherited_not_reresolved"):
                if checks.get(key) != "skip":
                    statuses.append(checks[key])
    return regression_result(
        description="normalize_for_match hoạt động nhất quán",
        method=(
            "Scenario 1 turns 1-3: normalize/inheritance checks must pass for "
            "case/diacritic variants"
        ),
        statuses=statuses,
    )


def build_stale_binding_regression(scenarios: list[dict[str, Any]]) -> dict[str, str]:
    scenario = _scenario_by_key(scenarios, "entity_switch_mid_conversation")
    statuses = []
    if scenario:
        for turn in scenario["turns"][2:5]:
            checks = turn.get("v2_checks", {})
            for key in ("entity_switched_correctly", "no_stale_entity"):
                if checks.get(key) != "skip":
                    statuses.append(checks[key])
    return regression_result(
        description="Không có stale entity sau entity switch",
        method="Scenario 2 turns 3-5: entity switch/no_stale checks must pass",
        statuses=statuses,
    )


def regression_result(
    *,
    description: str,
    method: str,
    statuses: list[str],
) -> dict[str, str]:
    if not statuses:
        result = "skip"
    else:
        result = "pass" if all(status == "pass" for status in statuses) else "fail"
    return {
        "description": description,
        "method": method,
        "result": result,
        "evidence": ", ".join(statuses) if statuses else "no applicable debug checks",
    }


def _scenario_by_key(
    scenarios: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    return next((scenario for scenario in scenarios if scenario["scenario_key"] == key), None)


def build_delta_from_v1(
    compare_path: str | None,
    summary: dict[str, Any],
) -> dict[str, Any] | None:
    if not compare_path:
        return None
    path = Path(compare_path)
    if not path.exists():
        raise FileNotFoundError(f"Compare file not found: {compare_path}")
    with path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    previous_summary = previous.get("summary", {}) if isinstance(previous, dict) else {}
    keys = ("intent_accuracy", "entity_accuracy", "follow_up_accuracy")
    return {
        key: signed_delta(summary.get(key), previous_summary.get(key))
        for key in keys
    } | {"new_checks_introduced": list(V2_CHECK_KEYS)}


def signed_delta(current: Any, previous: Any) -> str | None:
    if current is None or previous is None:
        return None
    try:
        delta = float(current) - float(previous)
    except (TypeError, ValueError):
        return None
    return f"{delta:+.4f}"


def print_turn_progress(turn: dict[str, Any], turn_number: int, total: int) -> None:
    checks = turn["checks"]
    v2 = turn["v2_checks"]
    target = turn.get("test_target") or "-"
    fragments = [
        f"  turn {turn_number}/{total}",
        str(turn["expected_intent"]),
        f"intent {PASS_MARK if checks.get('intent_match') == 'pass' else FAIL_MARK}",
    ]
    for key in V2_CHECK_KEYS:
        if v2.get(key) != "skip":
            fragments.append(f"{key} {_symbol(v2[key])}")
    if checks.get("follow_up_memory_used") != "skip":
        fragments.append(f"follow_up_memory {_symbol(checks['follow_up_memory_used'])}")
    snapshot = turn["debug_snapshot"]
    fragments.append(f"gate={snapshot['gate_status']}")
    fragments.append(f"source={snapshot['resolution_source']}")
    fragments.append(f"target={target}")
    fragments.append(f"{turn['latency_ms']}ms")
    print(" | ".join(fragments), flush=True)


def _symbol(status: str) -> str:
    if status == "pass":
        return PASS_MARK
    if status == "fail":
        return FAIL_MARK
    return "skip"


def summarize_scenario_v2(scenario_result: dict[str, Any]) -> None:
    turns = scenario_result["turns"]
    passed = sum(1 for turn in turns if turn["turn_passed"])
    total = len(turns)
    scenario_result["scenario_pass_rate"] = round(passed / total, 4) if total else None
    scenario_result["follow_up_accuracy"] = check_rate(turns, "follow_up_memory_used")
    scenario_result["multi_task_accuracy"] = check_rate(turns, "multi_task_produced")
    scenario_result["passed"] = passed == total


def main() -> int:
    args = parse_args()
    backend_url = (args.backend_url or os.getenv("BACKEND_URL") or BASE_URL).rstrip("/")
    chat_endpoint = f"{backend_url}/chat"
    request_timeout = None if args.timeout_seconds <= 0 else args.timeout_seconds
    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    try:
        input_path = resolve_input_path(args.input)
        scenarios = filter_scenarios(load_scenarios(input_path), args.scenarios)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error loading inputs: {exc}", file=sys.stderr)
        return 2

    results: dict[str, Any] = {
        "run_id": run_id,
        "scenario_file": str(input_path),
        "backend_url": backend_url,
        "summary": {},
        "bug_regression": {},
        "scenarios": [],
    }

    exit_code = 0
    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_key = str(scenario.get("scenario_key", f"scenario_{scenario_index}"))
        session_id = f"{scenario_key}-v2-{run_id}"
        turns = scenario.get("turns") if isinstance(scenario.get("turns"), list) else []
        scenario_result: dict[str, Any] = {
            "scenario_key": scenario_key,
            "title": scenario.get("title", ""),
            "passed": False,
            "scenario_pass_rate": 0.0,
            "follow_up_accuracy": None,
            "multi_task_accuracy": None,
            "turns": [],
        }
        previous_expected_entities: list[str] = []
        print(f"[v2] scenario {scenario_index}/{len(scenarios)} — {scenario_key}", flush=True)

        for turn_index, turn in enumerate(turns):
            request_payload = {
                "message": turn.get("query", ""),
                "session_id": session_id,
                "debug": True,
            }
            result = post_json(chat_endpoint, request_payload, request_timeout)
            turn_result = score_turn(turn, result, turn_index, request_timeout)
            response = result.data if isinstance(result.data, dict) else {}
            debug = extract_debug(response)
            v2_checks, v2_notes, v2_failures = score_v2_checks(
                turn,
                response,
                debug,
                previous_expected_entities,
            )
            metadata = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
            turn_result["test_target"] = metadata.get("test_target")
            turn_result["v2_checks"] = v2_checks
            turn_result["v2_notes"] = v2_notes
            turn_result["debug_snapshot"] = build_debug_snapshot(debug)
            turn_result["failure_reasons"].extend(v2_failures)
            turn_result["turn_passed"] = turn_result["turn_passed"] and all(
                status in {"pass", "skip"} for status in v2_checks.values()
            )
            scenario_result["turns"].append(turn_result)
            print_turn_progress(turn_result, turn_index + 1, len(turns))
            if args.verbose:
                print(json.dumps(turn_result["debug_snapshot"], ensure_ascii=False, indent=2))
            if result.unreachable:
                print(
                    f"Backend unreachable at {chat_endpoint}: {result.error_message}",
                    file=sys.stderr,
                )
                exit_code = 1
                break
            current_expected = [str(item) for item in turn.get("expected_entities") or []]
            if current_expected:
                previous_expected_entities = current_expected

        summarize_scenario_v2(scenario_result)
        passed_turns = sum(1 for turn in scenario_result["turns"] if turn["turn_passed"])
        print(
            f"  result: {passed_turns}/{len(turns)} turns passed "
            f"{PASS_MARK if scenario_result['passed'] else FAIL_MARK}\n",
            flush=True,
        )
        results["scenarios"].append(scenario_result)
        if exit_code:
            break

    results["summary"] = build_v2_summary(results["scenarios"])
    results["bug_regression"] = build_bug_regression(results["scenarios"])
    if args.compare:
        results["delta_from_v1"] = build_delta_from_v1(args.compare, results["summary"])
    write_results(Path(args.output), results)
    print_final_report(results, Path(args.output))
    return exit_code


def print_final_report(results: dict[str, Any], output_path: Path) -> None:
    print(f"Results written to {output_path}")
    print("\n[v2] Summary")
    for key, value in results["summary"].items():
        print(f"  {key}: {value}")
    bug = results["bug_regression"]["effective_query_missing_entity_removed"]
    symbol = (
        PASS_MARK
        if bug["result"] == "pass"
        else FAIL_MARK
        if bug["result"] == "fail"
        else "skip"
    )
    print(
        "\n[v2] bug_regression: effective_query_missing_entity "
        f"→ {bug['occurrences']} occurrences {symbol}"
    )
    for key, value in results["bug_regression"].items():
        print(f"  {key}: {value['result']} | {value.get('evidence', value.get('method'))}")
    failed = [scenario for scenario in results["scenarios"] if not scenario["passed"]]
    if failed:
        print("\n[v2] Failed scenarios")
        for scenario in failed:
            print(f"  {scenario['scenario_key']}")
            for turn in scenario["turns"]:
                if turn["turn_passed"]:
                    continue
                failed_standard = [
                    key for key, status in turn["checks"].items() if status == "fail"
                ]
                failed_v2 = [
                    key for key, status in turn["v2_checks"].items() if status == "fail"
                ]
                print(
                    f"    turn {turn['turn_index'] + 1}: "
                    f"standard={failed_standard}; v2={failed_v2}; "
                    f"reasons={turn['failure_reasons']}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
