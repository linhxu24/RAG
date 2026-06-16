from collections import Counter, defaultdict
from typing import Any

from app.observability.metrics import percentile


def build_diagnostics(
    *,
    case_results: list[dict[str, Any]],
    trace_steps: list[dict[str, Any]],
    retrieval_coverage: float,
) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []
    step_latencies: dict[str, list[float]] = defaultdict(list)
    failed_steps: Counter[str] = Counter()
    for step in trace_steps:
        step_name = str(step.get("step_name") or "unknown")
        step_latencies[step_name].append(float(step.get("latency_ms") or 0))
        if step.get("status") == "failed":
            failed_steps[step_name] += 1

    latency_by_stage = {
        name: {
            "count": len(values),
            "average_ms": sum(values) / len(values),
            "p95_ms": percentile(values, 0.95),
            "max_ms": max(values),
        }
        for name, values in sorted(step_latencies.items())
    }
    for step_name, values in latency_by_stage.items():
        threshold = 25_000 if step_name in {"llm_generation", "router_intent"} else 10_000
        if values["p95_ms"] > threshold:
            alerts.append(
                _alert(
                    "warning",
                    f"slow_{step_name}",
                    f"{step_name} latency is high",
                    f"p95={values['p95_ms']:.0f} ms, threshold={threshold} ms",
                    values["p95_ms"],
                    threshold,
                )
            )
    for step_name, count in failed_steps.items():
        alerts.append(
            _alert(
                "critical" if step_name in {"json_validation", "dense_retrieval"} else "warning",
                f"failed_{step_name}",
                f"{step_name} has failed steps",
                f"{count} failed trace step(s)",
                count,
                0,
            )
        )

    total = len(case_results)
    fallback_rate = (
        sum(bool(item.get("details", {}).get("fallback")) for item in case_results) / total
        if total
        else 0.0
    )
    no_result_rate = (
        sum(bool(item.get("details", {}).get("no_result")) for item in case_results) / total
        if total
        else 0.0
    )
    failed_case_rate = (
        sum(item.get("passed") is False for item in case_results) / total if total else 0.0
    )
    retrieval_empty = sum(
        bool(item.get("expected_ids")) and not item.get("retrieved_ids")
        for item in case_results
    )
    if retrieval_coverage < 0.6:
        alerts.append(
            _alert(
                "warning",
                "low_ground_truth_coverage",
                "Retrieval ground-truth coverage is low",
                f"Only {retrieval_coverage:.1%} of cases can measure retrieval",
                retrieval_coverage,
                0.6,
            )
        )
    if retrieval_empty:
        alerts.append(
            _alert(
                "critical",
                "empty_retrieval",
                "Grounded cases returned no retrieval results",
                f"{retrieval_empty} case(s) had expected sources but retrieved nothing",
                retrieval_empty,
                0,
            )
        )
    if fallback_rate > 0.1:
        alerts.append(
            _alert(
                "warning",
                "high_fallback_rate",
                "Fallback rate is high",
                f"fallback_rate={fallback_rate:.1%}",
                fallback_rate,
                0.1,
            )
        )
    if failed_case_rate > 0.1:
        alerts.append(
            _alert(
                "critical",
                "high_case_failure_rate",
                "Evaluation case failure rate is high",
                f"failed_case_rate={failed_case_rate:.1%}",
                failed_case_rate,
                0.1,
            )
        )
    return {
        "alerts": alerts,
        "latency_by_stage": latency_by_stage,
        "failed_steps": dict(failed_steps),
        "fallback_rate": fallback_rate,
        "no_result_rate": no_result_rate,
        "failed_case_rate": failed_case_rate,
    }


def _alert(
    severity: str,
    code: str,
    title: str,
    detail: str,
    value: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "title": title,
        "detail": detail,
        "value": value,
        "threshold": threshold,
    }
