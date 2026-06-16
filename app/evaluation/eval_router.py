from collections import Counter, defaultdict
from typing import Any

from app.retrieval.router import IntentRouter


def evaluate_router(
    cases: list[dict[str, Any]],
    *,
    known_products: list[str] | None = None,
    known_services: list[str] | None = None,
) -> dict[str, Any]:
    router = IntentRouter()
    predictions: list[tuple[str, str, float, bool]] = []
    for case in cases:
        result = router.route(
            case["query"],
            known_products=known_products,
            known_services=known_services,
        )
        predictions.append(
            (
                str(case.get("expected_intent", "UNKNOWN")),
                result.intent.value,
                result.confidence,
                result.needs_clarification,
            )
        )
    labels = sorted({item for row in predictions for item in row[:2]})
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_label: dict[str, dict[str, float]] = {}
    for expected, predicted, _, _ in predictions:
        confusion[expected][predicted] += 1
    for label in labels:
        tp = sum(
            expected == label and predicted == label for expected, predicted, *_ in predictions
        )
        fp = sum(
            expected != label and predicted == label for expected, predicted, *_ in predictions
        )
        fn = sum(
            expected == label and predicted != label for expected, predicted, *_ in predictions
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    correct = sum(expected == predicted for expected, predicted, *_ in predictions)
    return {
        "case_count": len(predictions),
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "per_label": per_label,
        "confusion_matrix": {expected: dict(counts) for expected, counts in confusion.items()},
        "low_confidence_rate": sum(score < 0.65 for *_, score, _ in predictions) / len(predictions)
        if predictions
        else 0.0,
        "clarification_rate": sum(clarify for *_, clarify in predictions) / len(predictions)
        if predictions
        else 0.0,
        "predictions": [
            {
                "query": case["query"],
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
            }
            for case, (expected, predicted, confidence, _) in zip(cases, predictions, strict=True)
        ],
    }
