from typing import Any

from app.observability.metrics import percentile


def evaluate_e2e(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    latencies = [
        float(item["latency_ms"])
        for item in records
        if item.get("latency_ms") is not None
    ]
    return {
        "case_count": total,
        "success_rate": _rate(records, lambda item: item.get("status") == "completed"),
        "pass_rate": _rate(records, lambda item: item.get("passed") is True),
        "fallback_rate": _rate(
            records,
            lambda item: bool(item.get("details", {}).get("fallback")),
        ),
        "no_result_rate": _rate(
            records,
            lambda item: bool(item.get("details", {}).get("no_result")),
        ),
        "clarification_rate": _rate(
            records,
            lambda item: item.get("details", {}).get("actual_answer_type")
            == "clarification",
        ),
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
    }


def _rate(records: list[dict[str, Any]], predicate) -> float:
    return sum(bool(predicate(item)) for item in records) / len(records) if records else 0.0
