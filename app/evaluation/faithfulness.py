import json
from typing import Any

from app.assets.resolver import detect_asset_tokens
from app.generation.validator import ResponseValidator


def evaluate_answer(
    *,
    answer_text: str,
    answer_payload: dict[str, Any],
    source_text: str,
    retrieved_ids: list[str],
    case: dict[str, Any],
) -> tuple[dict[str, bool | float | None], list[dict[str, Any]]]:
    lowered = answer_text.lower()
    expected_contains = [item.lower() for item in case.get("expected_answer_contains", [])]
    forbidden = [item.lower() for item in case.get("forbidden_answer_contains", [])]
    expected_entities = [item.lower() for item in case.get("expected_entities", [])]
    violations: list[dict[str, Any]] = []

    answer_correctness = None
    if expected_contains or forbidden or expected_entities:
        missing = [item for item in expected_contains if item not in lowered]
        present_forbidden = [item for item in forbidden if item in lowered]
        payload_text = json.dumps(answer_payload, ensure_ascii=False).lower()
        missing_entities = [
            item for item in expected_entities if item not in lowered and item not in payload_text
        ]
        if missing:
            violations.append({"type": "missing_expected_text", "values": missing})
        if present_forbidden:
            violations.append({"type": "forbidden_text", "values": present_forbidden})
        if missing_entities:
            violations.append({"type": "missing_expected_entity", "values": missing_entities})
        answer_correctness = not (missing or present_forbidden or missing_entities)

    expected_answer_type = case.get("expected_answer_type")
    actual_answer_type = answer_payload.get("answer_type")
    answer_type_match = (
        actual_answer_type == expected_answer_type if expected_answer_type else None
    )
    if answer_type_match is False:
        violations.append(
            {
                "type": "answer_type_mismatch",
                "expected": expected_answer_type,
                "actual": actual_answer_type,
            }
        )

    generated_prices = ResponseValidator._price_numbers(answer_text)
    source_numbers = ResponseValidator._all_numbers(source_text)
    unsupported_prices = sorted(generated_prices - source_numbers)
    price_grounded = not unsupported_prices if generated_prices else None
    if unsupported_prices:
        violations.append({"type": "unsupported_price", "values": unsupported_prices})

    sources = answer_payload.get("answer", {}).get("sources", [])
    cited_ids = {str(item.get("source_id")) for item in sources if item.get("source_id")}
    citation_grounded = cited_ids.issubset(set(retrieved_ids)) if cited_ids else None
    if citation_grounded is False:
        violations.append(
            {
                "type": "citation_not_retrieved",
                "values": sorted(cited_ids - set(retrieved_ids)),
            }
        )

    answer = answer_payload.get("answer", {})
    tokens = set(detect_asset_tokens(answer_text))
    resolved_tokens = {
        str(item.get("token"))
        for item in answer.get("assets", [])
        if item.get("token")
    }
    missing_tokens = sorted(tokens - resolved_tokens)
    asset_grounded = not missing_tokens if tokens else None
    if missing_tokens:
        violations.append({"type": "unresolved_asset_token", "values": missing_tokens})
    if answer.get("missing_assets"):
        violations.append(
            {"type": "missing_assets", "values": list(answer.get("missing_assets", []))}
        )
        asset_grounded = False

    medical = bool(case.get("metadata", {}).get("medical"))
    safety = answer_payload.get("safety", {})
    safety_pass = None
    if medical:
        safety_pass = bool(
            safety.get("medical_disclaimer_required")
            or any(term in lowered for term in ("nha sĩ", "bác sĩ", "đi khám", "cấp cứu"))
        )
        if not safety_pass:
            violations.append({"type": "medical_safety_missing"})

    factual_checks = [
        value
        for value in (price_grounded, citation_grounded, asset_grounded)
        if value is not None
    ]
    faithfulness = all(factual_checks) if factual_checks else None
    return (
        {
            "answer_correctness": _bool_score(answer_correctness),
            "answer_type_match": _bool_score(answer_type_match),
            "faithfulness": _bool_score(faithfulness),
            "price_grounded": _bool_score(price_grounded),
            "citation_grounded": _bool_score(citation_grounded),
            "asset_grounded": _bool_score(asset_grounded),
            "safety_pass": _bool_score(safety_pass),
        },
        violations,
    )


def _bool_score(value: bool | None) -> float | None:
    return float(value) if value is not None else None
