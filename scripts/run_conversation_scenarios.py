from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_INPUT = "dental_conversation_scenarios.json"
DEFAULT_OUTPUT = "eval_results/conversation_scenario_results.json"
REQUEST_TIMEOUT_SECONDS = 60.0

CHECK_KEYS = (
    "intent_match",
    "entity_present",
    "follow_up_memory_used",
    "multi_task_produced",
    "not_degraded",
    "has_answer_text",
    "valid_json_response",
    "no_hallucinated_price",
)

PASS_MARK = "\u2713"
FAIL_MARK = "\u2717"
PRICE_RE = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:\s?(?:k|nghìn|nghin|tr|triệu|"
    r"trieu|vnd|vnđ|đ|₫|\$))\b|(?:₫|\$)\s?\d[\d.,]*",
    re.IGNORECASE,
)


@dataclass
class HTTPResult:
    status_code: int | None
    data: Any
    text: str
    latency_ms: int
    error_message: str | None = None
    timeout: bool = False
    unreachable: bool = False
    malformed_json: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-turn dental conversation scenarios against a live backend."
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario_key values to run. Default: all scenarios.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Path to scenarios JSON. Default: dental_conversation_scenarios.json; "
            "falls back to eval_datasets/dental_conversation_scenarios.json if present."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Path to write results JSON. "
            "Default: eval_results/conversation_scenario_results.json."
        ),
    )
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Override BACKEND_URL. Default: BACKEND_URL env var or http://localhost:8000.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full response JSON.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Per-turn HTTP timeout. Use 0 to avoid setting a client timeout.",
    )
    return parser.parse_args()


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    if raw_path == DEFAULT_INPUT:
        fallback = Path("eval_datasets") / DEFAULT_INPUT
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Scenario file not found: {raw_path}")


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(scenarios, list):
        raise ValueError("Scenario JSON must contain a list at key 'scenarios'.")
    return scenarios


def filter_scenarios(
    scenarios: list[dict[str, Any]],
    selected: str | None,
) -> list[dict[str, Any]]:
    if not selected:
        return scenarios
    keys = {item.strip() for item in selected.split(",") if item.strip()}
    by_key = {str(scenario.get("scenario_key")): scenario for scenario in scenarios}
    missing = sorted(keys - set(by_key))
    if missing:
        raise ValueError(f"Unknown scenario key(s): {', '.join(missing)}")
    return [scenario for scenario in scenarios if scenario.get("scenario_key") in keys]


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float | None,
) -> HTTPResult:
    try:
        import httpx
    except ImportError:
        return post_json_urllib(url, payload, timeout_seconds)

    start = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        return HTTPResult(
            status_code=None,
            data=None,
            text="",
            latency_ms=elapsed_ms(start),
            error_message=str(exc) or "Request timed out.",
            timeout=True,
        )
    except httpx.RequestError as exc:
        return HTTPResult(
            status_code=None,
            data=None,
            text="",
            latency_ms=elapsed_ms(start),
            error_message=str(exc) or "Backend is unreachable.",
            unreachable=True,
        )

    text = response.text
    try:
        data = response.json()
    except ValueError as exc:
        data = None
        malformed = True
        error_message = f"Malformed JSON response: {exc}"
    else:
        malformed = False
        error_message = None

    return HTTPResult(
        status_code=response.status_code,
        data=data,
        text=text,
        latency_ms=elapsed_ms(start),
        error_message=error_message,
        malformed_json=malformed,
    )


def post_json_urllib(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float | None,
) -> HTTPResult:
    start = time.perf_counter()
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        if timeout_seconds is None:
            response_handle = urllib.request.urlopen(request)
        else:
            response_handle = urllib.request.urlopen(request, timeout=timeout_seconds)
        with response_handle as response:
            text = response.read().decode("utf-8", errors="replace")
            status_code = response.status
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        data, malformed, error = parse_response_json(text)
        return HTTPResult(
            status_code=exc.code,
            data=data,
            text=text,
            latency_ms=elapsed_ms(start),
            error_message=error or str(exc),
            malformed_json=malformed,
        )
    except TimeoutError as exc:
        return HTTPResult(
            status_code=None,
            data=None,
            text="",
            latency_ms=elapsed_ms(start),
            error_message=str(exc) or "Request timed out.",
            timeout=True,
        )
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return HTTPResult(
                status_code=None,
                data=None,
                text="",
                latency_ms=elapsed_ms(start),
                error_message=str(exc.reason) or "Request timed out.",
                timeout=True,
            )
        return HTTPResult(
            status_code=None,
            data=None,
            text="",
            latency_ms=elapsed_ms(start),
            error_message=str(exc.reason) or "Backend is unreachable.",
            unreachable=True,
        )

    data, malformed, error = parse_response_json(text)
    return HTTPResult(
        status_code=status_code,
        data=data,
        text=text,
        latency_ms=elapsed_ms(start),
        error_message=error,
        malformed_json=malformed,
    )


def elapsed_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


def parse_response_json(text: str) -> tuple[Any, bool, str | None]:
    try:
        return json.loads(text), False, None
    except ValueError as exc:
        return None, True, f"Malformed JSON response: {exc}"


def score_turn(
    turn: dict[str, Any],
    result: HTTPResult,
    turn_index: int,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    expected_intent = str(turn.get("expected_intent", ""))
    expected_entities = [str(item) for item in turn.get("expected_entities") or []]
    metadata = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
    requires_follow_up = bool(metadata.get("requires_follow_up_memory"))
    expects_multi_task = bool(metadata.get("expects_multi_task"))

    if result.timeout:
        return failed_transport_turn(
            turn,
            turn_index,
            result,
            "timeout",
            timeout_message(timeout_seconds),
        )
    if result.status_code is not None and result.status_code >= 400:
        return failed_transport_turn(
            turn,
            turn_index,
            result,
            "http_error",
            result.error_message or f"HTTP {result.status_code}",
        )
    if result.unreachable:
        return failed_transport_turn(
            turn,
            turn_index,
            result,
            "backend_unreachable",
            result.error_message or "Backend is unreachable.",
        )

    response = result.data if isinstance(result.data, dict) else {}
    answer_text = extract_answer_text(response)
    actual_intent = string_or_none(response.get("intent"))
    degraded = response.get("degraded") if isinstance(response, dict) else None
    found_entities = find_expected_entities(expected_entities, response, answer_text)
    checks = {key: "skip" for key in CHECK_KEYS}
    failure_reasons: list[str] = []

    checks["valid_json_response"] = (
        "pass" if not result.malformed_json and has_expected_schema(response) else "fail"
    )
    if checks["valid_json_response"] == "fail":
        failure_reasons.append(result.error_message or "Response JSON does not match schema.")

    checks["intent_match"] = "pass" if actual_intent == expected_intent else "fail"
    if checks["intent_match"] == "fail":
        failure_reasons.append(f"intent_match: expected {expected_intent}, got {actual_intent}")

    if expected_entities:
        checks["entity_present"] = (
            "pass" if len(found_entities) == len(expected_entities) else "fail"
        )
        if checks["entity_present"] == "fail":
            missing = sorted(set(expected_entities) - set(found_entities))
            failure_reasons.append(f"entity_present: missing {missing}")
    else:
        checks["entity_present"] = "skip"

    if requires_follow_up:
        checks["follow_up_memory_used"] = score_follow_up_memory(
            expected_entities,
            found_entities,
            str(turn.get("query", "")),
        )
        if checks["follow_up_memory_used"] == "fail":
            failure_reasons.append(
                "follow_up_memory_used: expected prior entity in response without re-mention."
            )
    else:
        checks["follow_up_memory_used"] = "skip"

    if expects_multi_task:
        task_count = extract_task_count(response)
        checks["multi_task_produced"] = "pass" if task_count >= 2 else "fail"
        if checks["multi_task_produced"] == "fail":
            failure_reasons.append(f"multi_task_produced: found {task_count} planned task(s).")
    else:
        checks["multi_task_produced"] = "skip"

    checks["not_degraded"] = "pass" if degraded is False else "fail"
    if checks["not_degraded"] == "fail":
        failure_reasons.append(f"not_degraded: response.degraded is {degraded!r}")

    checks["has_answer_text"] = "pass" if isinstance(answer_text, str) and answer_text else "fail"
    if checks["has_answer_text"] == "fail":
        failure_reasons.append("has_answer_text: answer.text is empty or missing.")

    checks["no_hallucinated_price"] = score_price_grounding(response, answer_text)
    if checks["no_hallucinated_price"] == "fail":
        failure_reasons.append("no_hallucinated_price: price in answer not found in items/sources.")

    turn_passed = all(value in {"pass", "skip"} for value in checks.values())
    return {
        "turn_index": turn_index,
        "query": turn.get("query", ""),
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "expected_entities": expected_entities,
        "found_entities": found_entities,
        "requires_follow_up_memory": requires_follow_up,
        "expects_multi_task": expects_multi_task,
        "latency_ms": result.latency_ms,
        "http_status": result.status_code,
        "degraded": degraded,
        "answer_text": answer_text,
        "checks": checks,
        "turn_passed": turn_passed,
        "failure_reasons": failure_reasons,
    }


def failed_transport_turn(
    turn: dict[str, Any],
    turn_index: int,
    result: HTTPResult,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    checks = {key: "fail" for key in CHECK_KEYS}
    return {
        "turn_index": turn_index,
        "query": turn.get("query", ""),
        "expected_intent": turn.get("expected_intent"),
        "actual_intent": None,
        "expected_entities": turn.get("expected_entities") or [],
        "found_entities": [],
        "requires_follow_up_memory": bool(
            (turn.get("metadata") or {}).get("requires_follow_up_memory")
        ),
        "expects_multi_task": bool((turn.get("metadata") or {}).get("expects_multi_task")),
        "latency_ms": result.latency_ms,
        "http_status": result.status_code,
        "degraded": None,
        "answer_text": "",
        "checks": checks,
        "turn_passed": False,
        "failure_reasons": [f"{error_type}: {message}"],
        "error_type": error_type,
        "error_message": message,
        "timeout": result.timeout,
    }


def timeout_message(timeout_seconds: float | None) -> str:
    if timeout_seconds is None:
        return "Request timed out."
    return f"Request exceeded {timeout_seconds:g} seconds."


def has_expected_schema(response: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    answer = response.get("answer")
    return (
        isinstance(response.get("intent"), str)
        and isinstance(answer, dict)
        and isinstance(answer.get("text"), str)
        and isinstance(response.get("degraded"), bool)
    )


def extract_answer_text(response: dict[str, Any]) -> str:
    answer = response.get("answer") if isinstance(response, dict) else None
    if isinstance(answer, dict):
        text = answer.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(answer, str):
        return answer
    return ""


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", without_marks).strip()


def contains_text(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return normalize_text(needle) in normalize_text(haystack)


def find_expected_entities(
    expected_entities: list[str],
    response: dict[str, Any],
    answer_text: str,
) -> list[str]:
    if not expected_entities:
        return []
    entity_haystack = "\n".join([answer_text, *collect_entity_strings(response)])
    return [entity for entity in expected_entities if contains_text(entity_haystack, entity)]


def collect_entity_strings(response: dict[str, Any]) -> list[str]:
    strings: list[str] = []
    raw_entities = response.get("entities")
    if isinstance(raw_entities, list):
        strings.extend(collect_named_values(raw_entities))
    answer = response.get("answer")
    if isinstance(answer, dict):
        items = answer.get("items")
        if isinstance(items, list):
            strings.extend(collect_named_values(items))
    return strings


def collect_named_values(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "name",
                "entity_name",
                "product_name",
                "service_name",
                "question",
                "title",
            } and isinstance(item, str):
                output.append(item)
            elif isinstance(item, dict | list):
                output.extend(collect_named_values(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(collect_named_values(item))
    elif isinstance(value, str):
        output.append(value)
    return output


def score_follow_up_memory(
    expected_entities: list[str],
    found_entities: list[str],
    query: str,
) -> str:
    if not expected_entities:
        return "skip"
    memory_entities = [entity for entity in expected_entities if not contains_text(query, entity)]
    if not memory_entities:
        return "fail"
    return "pass" if all(entity in found_entities for entity in memory_entities) else "fail"


def extract_task_count(response: dict[str, Any]) -> int:
    candidate_counts: list[int] = []
    for key in ("tasks", "planned_tasks", "planner_tasks"):
        value = response.get(key)
        if isinstance(value, list):
            candidate_counts.append(len(value))
    debug = response.get("debug")
    if isinstance(debug, dict):
        collect_task_counts(debug, candidate_counts)
    return max(candidate_counts, default=0)


def collect_task_counts(value: Any, counts: list[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"tasks", "planned_tasks", "planner_tasks"} and isinstance(item, list):
                counts.append(len(item))
            if isinstance(item, dict | list):
                collect_task_counts(item, counts)
    elif isinstance(value, list):
        for item in value:
            collect_task_counts(item, counts)


def score_price_grounding(response: dict[str, Any], answer_text: str) -> str:
    price_terms = PRICE_RE.findall(answer_text)
    if not price_terms:
        return "skip"
    answer = response.get("answer") if isinstance(response, dict) else {}
    evidence_blob = ""
    if isinstance(answer, dict):
        evidence_blob = json.dumps(
            {
                "items": answer.get("items") or [],
                "sources": answer.get("sources") or [],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    normalized_evidence = normalize_price_text(evidence_blob)
    for term in price_terms:
        if not price_term_in_evidence(term, normalized_evidence):
            return "fail"
    return "pass"


def normalize_price_text(value: str) -> str:
    return normalize_text(value).replace(",", "").replace(".", "")


def price_term_in_evidence(term: str, normalized_evidence: str) -> bool:
    normalized_term = normalize_price_text(term)
    if normalized_term and normalized_term in normalized_evidence:
        return True
    digits = re.sub(r"\D", "", normalized_term)
    if not digits:
        return False
    candidates = {digits}
    lower = normalize_text(term)
    if re.search(r"\b(k|nghin|nghìn)\b", lower):
        candidates.add(str(int(digits) * 1000))
    if re.search(r"\b(tr|trieu|triệu)\b", lower):
        candidates.add(str(int(digits) * 1_000_000))
    return any(candidate in normalized_evidence for candidate in candidates)


def summarize_scenario(scenario_result: dict[str, Any]) -> None:
    turns = scenario_result["turns"]
    passed = sum(1 for turn in turns if turn["turn_passed"])
    total = len(turns)
    scenario_result["scenario_pass_rate"] = safe_rate(passed, total)
    scenario_result["follow_up_accuracy"] = check_rate(turns, "follow_up_memory_used")
    scenario_result["multi_task_accuracy"] = check_rate(turns, "multi_task_produced")
    scenario_result["passed"] = passed == total


def build_summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for scenario in scenarios for turn in scenario["turns"]]
    latencies = [int(turn["latency_ms"]) for turn in turns]
    total_turns = len(turns)
    turns_passed = sum(1 for turn in turns if turn["turn_passed"])
    not_degraded_failed = count_check(turns, "not_degraded", "fail")
    return {
        "total_scenarios": len(scenarios),
        "scenarios_passed": sum(1 for scenario in scenarios if scenario["passed"]),
        "scenarios_failed": sum(1 for scenario in scenarios if not scenario["passed"]),
        "total_turns": total_turns,
        "turns_passed": turns_passed,
        "turns_failed": total_turns - turns_passed,
        "intent_accuracy": check_rate(turns, "intent_match", denominator=total_turns),
        "entity_accuracy": check_rate(turns, "entity_present"),
        "follow_up_accuracy": check_rate(turns, "follow_up_memory_used"),
        "multi_task_accuracy": check_rate(turns, "multi_task_produced"),
        "degraded_rate": safe_rate(not_degraded_failed, total_turns),
        "avg_latency_ms": int(round(sum(latencies) / len(latencies))) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 95),
    }


def check_rate(
    turns: list[dict[str, Any]],
    check_key: str,
    denominator: int | None = None,
) -> float | None:
    if denominator is None:
        applicable = [turn for turn in turns if turn["checks"].get(check_key) != "skip"]
        denominator = len(applicable)
    else:
        applicable = turns
    if denominator == 0:
        return None
    passed = sum(1 for turn in applicable if turn["checks"].get(check_key) == "pass")
    return round(passed / denominator, 4)


def count_check(turns: list[dict[str, Any]], check_key: str, status: str) -> int:
    return sum(1 for turn in turns if turn["checks"].get(check_key) == status)


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def percentile(values: list[int], percentile_value: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile_value / 100) * len(ordered)) - 1)
    return ordered[index]


def build_acceptance_mapping(
    scenarios: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, str]]:
    turns = [turn for scenario in scenarios for turn in scenario["turns"]]
    total_turns = len(turns)
    valid_json_passed = count_check(turns, "valid_json_response", "pass")
    not_degraded_passed = count_check(turns, "not_degraded", "pass")
    no_price_failed = count_check(turns, "no_hallucinated_price", "fail")
    entity_failed = count_check(turns, "entity_present", "fail")
    switch_passed, switch_total = explicit_switching_counts(scenarios)

    return [
        criterion(
            "AC-21",
            "Chat endpoint returns structured JSON",
            valid_json_passed == total_turns,
            f"valid_json_response: {valid_json_passed}/{total_turns} turns passed",
        ),
        criterion(
            "AC-22",
            "LLM output is validated",
            valid_json_passed == total_turns and not_degraded_passed == total_turns,
            (
                f"valid_json_response: {valid_json_passed}/{total_turns}; "
                f"not_degraded: {not_degraded_passed}/{total_turns}"
            ),
        ),
        criterion_from_rate(
            "AC-31",
            "One message can produce multiple tasks",
            summary.get("multi_task_accuracy"),
            "multi_task_accuracy",
        ),
        criterion_from_rate(
            "AC-32",
            "Implicit follow-up cannot leave stale planner entities",
            summary.get("follow_up_accuracy"),
            "follow_up_memory_used pass rate",
        ),
        criterion(
            "AC-33",
            "Explicit entity switching resolved against DB",
            switch_total > 0 and switch_passed == switch_total,
            f"entity_present on switching turns: {switch_passed}/{switch_total}",
        ),
        criterion(
            "AC-34",
            "Synthesis cannot return entity conflicting with evidence",
            no_price_failed == 0 and entity_failed == 0,
            (
                f"no_hallucinated_price failures: {no_price_failed}; "
                f"entity_present failures: {entity_failed}"
            ),
        ),
        criterion(
            "AC-35",
            "Conversation eval measures entity binding, follow-up, multi-task, scenario pass rate",
            total_turns > 0 and "scenarios_passed" in summary,
            (
                f"scenarios_passed: {summary.get('scenarios_passed')}/"
                f"{summary.get('total_scenarios')}; "
                f"follow_up_accuracy: {summary.get('follow_up_accuracy')}; "
                f"multi_task_accuracy: {summary.get('multi_task_accuracy')}"
            ),
        ),
    ]


def criterion(
    criterion_id: str,
    description: str,
    passed: bool,
    evidence: str,
) -> dict[str, str]:
    return {
        "criterion_id": criterion_id,
        "description": description,
        "result": "pass" if passed else "fail",
        "evidence": evidence,
    }


def criterion_from_rate(
    criterion_id: str,
    description: str,
    rate: Any,
    label: str,
) -> dict[str, str]:
    if rate is None:
        result = "skip"
        evidence = f"{label}: no applicable turns"
    else:
        result = "pass" if rate == 1.0 else "fail"
        evidence = f"{label}: {rate}"
    return {
        "criterion_id": criterion_id,
        "description": description,
        "result": result,
        "evidence": evidence,
    }


def explicit_switching_counts(scenarios: list[dict[str, Any]]) -> tuple[int, int]:
    passed = 0
    total = 0
    for scenario in scenarios:
        previous_entities: set[str] = set()
        for turn in scenario["turns"]:
            current_entities = set(turn.get("expected_entities") or [])
            if previous_entities and current_entities and current_entities != previous_entities:
                total += 1
                if turn["checks"].get("entity_present") == "pass":
                    passed += 1
            if current_entities:
                previous_entities = current_entities
    return passed, total


def print_turn_progress(
    turn_result: dict[str, Any],
    turn_number: int,
    total_turns: int,
) -> None:
    checks = turn_result["checks"]
    parts = [
        f"  turn {turn_number}/{total_turns}",
        str(turn_result["expected_intent"]),
        check_fragment("intent", turn_result.get("actual_intent"), checks["intent_match"]),
    ]
    if checks["entity_present"] != "skip":
        entity_label = ",".join(turn_result["expected_entities"])
        parts.append(check_fragment("entity", entity_label, checks["entity_present"]))
    if checks["follow_up_memory_used"] != "skip":
        parts.append(check_fragment("follow_up_memory", None, checks["follow_up_memory_used"]))
    if checks["multi_task_produced"] != "skip":
        parts.append(check_fragment("multi_task", None, checks["multi_task_produced"]))
    parts.append(check_fragment("not_degraded", None, checks["not_degraded"]))
    parts.append(f"{turn_result['latency_ms']}ms")
    print(" | ".join(parts), flush=True)


def check_fragment(label: str, value: str | None, status: str) -> str:
    symbol = PASS_MARK if status == "pass" else FAIL_MARK if status == "fail" else "skip"
    if value:
        return f"{label}={value} {symbol}"
    return f"{label} {symbol}"


def write_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gitkeep = output_path.parent / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    backend_url = (args.backend_url or os.getenv("BACKEND_URL") or BASE_URL).rstrip("/")
    chat_endpoint = f"{backend_url}/chat"
    request_timeout = None if args.timeout_seconds <= 0 else args.timeout_seconds
    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    output_path = Path(args.output)

    try:
        input_path = resolve_input_path(args.input)
        scenarios = filter_scenarios(load_scenarios(input_path), args.scenarios)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error loading scenarios: {exc}", file=sys.stderr)
        return 2

    results: dict[str, Any] = {
        "run_id": run_id,
        "backend_url": backend_url,
        "summary": {},
        "scenarios": [],
        "acceptance_criteria_mapping": [],
    }

    exit_code = 0
    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_key = str(scenario.get("scenario_key", f"scenario_{scenario_index}"))
        session_id = f"{scenario_key}-{run_id}"
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
        print(f"[scenario {scenario_index}/{len(scenarios)}] {scenario_key}", flush=True)

        for turn_index, turn in enumerate(turns):
            request_payload = {
                "message": turn.get("query", ""),
                "session_id": session_id,
                "debug": True,
            }
            result = post_json(chat_endpoint, request_payload, request_timeout)
            turn_result = score_turn(turn, result, turn_index, request_timeout)
            scenario_result["turns"].append(turn_result)
            print_turn_progress(turn_result, turn_index + 1, len(turns))
            if args.verbose:
                print(json.dumps(result.data, ensure_ascii=False, indent=2), flush=True)
            if result.unreachable:
                print(
                    f"Backend unreachable at {chat_endpoint}: {result.error_message}",
                    file=sys.stderr,
                )
                exit_code = 1
                break

        summarize_scenario(scenario_result)
        passed_turns = sum(1 for turn in scenario_result["turns"] if turn["turn_passed"])
        print(
            f"  scenario result: {passed_turns}/{len(turns)} turns passed "
            f"{PASS_MARK if scenario_result['passed'] else FAIL_MARK}\n",
            flush=True,
        )
        results["scenarios"].append(scenario_result)
        if exit_code:
            break

    results["summary"] = build_summary(results["scenarios"])
    results["acceptance_criteria_mapping"] = build_acceptance_mapping(
        results["scenarios"],
        results["summary"],
    )
    write_results(output_path, results)
    print_final_report(results, output_path)
    return exit_code


def print_final_report(results: dict[str, Any], output_path: Path) -> None:
    print(f"Results written to {output_path}")
    print("\nSummary")
    for key, value in results["summary"].items():
        print(f"  {key}: {value}")
    print("\nAcceptance criteria mapping")
    for item in results["acceptance_criteria_mapping"]:
        print(
            f"  {item['criterion_id']} | {item['result']} | "
            f"{item['evidence']}"
        )
    failed_scenarios = [scenario for scenario in results["scenarios"] if not scenario["passed"]]
    if not failed_scenarios:
        return
    print("\nFailed scenarios")
    for scenario in failed_scenarios:
        print(f"  {scenario['scenario_key']}")
        for turn in scenario["turns"]:
            if turn["turn_passed"]:
                continue
            failed_checks = [
                key for key, status in turn["checks"].items() if status == "fail"
            ]
            print(
                f"    turn {turn['turn_index'] + 1}: checks={failed_checks}; "
                f"reasons={turn['failure_reasons']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
