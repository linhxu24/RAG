from typing import Any

from app.evaluation.metrics import mean


def evaluate_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(records),
        "json_validity_rate": mean([record.get("json_valid") for record in records]),
        "schema_pass_rate": mean([record.get("schema_pass") for record in records]),
        "answer_correctness": mean(
            [record.get("answer_correctness") for record in records]
        ),
        "faithfulness_rate": mean([record.get("faithfulness") for record in records]),
        "price_grounding_rate": mean(
            [record.get("price_grounded") for record in records]
        ),
        "citation_grounding_rate": mean(
            [record.get("citation_grounded") for record in records]
        ),
        "safety_pass_rate": mean([record.get("safety_pass") for record in records]),
        "unsupported_claim_rate": _inverse_mean(
            [record.get("faithfulness") for record in records]
        ),
    }


def _inverse_mean(values: list[float | None]) -> float | None:
    score = mean(values)
    return 1.0 - score if score is not None else None
