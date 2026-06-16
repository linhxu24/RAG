from typing import Any

from app.evaluation.metrics import hit_at, mean, ndcg_at, recall_at, reciprocal_rank


def retrieval_case_scores(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> dict[str, float | None]:
    expected = set(expected_ids)
    return {
        "hit_at_1": hit_at(retrieved_ids, expected, 1),
        "hit_at_3": hit_at(retrieved_ids, expected, 3),
        "recall_at_5": recall_at(retrieved_ids, expected, 5),
        "recall_at_10": recall_at(retrieved_ids, expected, 10),
        "mrr_at_10": reciprocal_rank(retrieved_ids, expected, 10),
        "ndcg_at_10": ndcg_at(retrieved_ids, expected, 10),
    }


def evaluate_retrieval(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        retrieval_case_scores(
            [str(item) for item in record.get("retrieved_ids", [])],
            [str(item) for item in record.get("expected_ids", [])],
        )
        for record in records
    ]
    eligible = sum(bool(record.get("expected_ids")) for record in records)
    total = len(records)
    return {
        "case_count": total,
        "eligible_case_count": eligible,
        "ground_truth_coverage": eligible / total if total else 0.0,
        **{
            metric: mean([score[metric] for score in scores])
            for metric in (
                "hit_at_1",
                "hit_at_3",
                "recall_at_5",
                "recall_at_10",
                "mrr_at_10",
                "ndcg_at_10",
            )
        },
    }


def evaluate_reranker(records: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [record for record in records if record.get("expected_ids")]
    before = [
        hit_at(record.get("before_ids", []), set(record["expected_ids"]), 1)
        for record in applicable
    ]
    after = [
        hit_at(record.get("after_ids", []), set(record["expected_ids"]), 1)
        for record in applicable
    ]
    improvements = sum(
        after_score is not None
        and before_score is not None
        and after_score > before_score
        for after_score, before_score in zip(after, before, strict=True)
    )
    return {
        "eligible_case_count": len(applicable),
        "top_1_before": mean(before),
        "top_1_after": mean(after),
        "improvement_rate": improvements / len(applicable) if applicable else None,
        "average_latency_added_ms": mean(
            [float(record.get("latency_ms", 0)) for record in applicable]
        ),
    }
