from collections import defaultdict
from typing import Any

from app.retrieval.types import RetrievalResult

_REPRESENTATION_PRIORITY = {
    "product": 0,
    "service": 0,
    "clinic_info": 0,
    "faq": 0,
    "table_row": 1,
    "chunk": 2,
}


def reciprocal_rank_fusion(
    result_sets: dict[str, list[RetrievalResult]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    max_per_source: int | None = None,
) -> list[RetrievalResult]:
    """Fuse multiple ranked lists using weighted Reciprocal Rank Fusion.

    Key behaviours:
    - Items sharing the same ``canonical_key`` are merged into one entry.
      The highest-scoring text representation is kept and scores are summed.
    - Per-source cap: when *max_per_source* is set, at most that many items
      originating from the same ``source_type`` appear in the final list.
    - Per-retriever weighting via *weights*.
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, RetrievalResult] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for retriever_name, results in result_sets.items():
        weight = weights.get(retriever_name, 1.0)
        for rank, result in enumerate(results, start=1):
            key = result.key
            rrf_score = weight / (k + rank)
            scores[key] += rrf_score
            # Keep the most authoritative representation for each canonical
            # entity. A normalized business object or FAQ must not be replaced
            # by a longer row/chunk representation that loses structured data.
            existing = records.get(key)
            if existing is None or _prefer_representation(result, existing):
                records[key] = result
            ranks[key][retriever_name] = rank

    fused: list[RetrievalResult] = []
    for key, score in scores.items():
        original = records[key]
        fused.append(
            RetrievalResult(
                source_type=original.source_type,
                source_id=original.source_id,
                text=original.text,
                score=score,
                raw_json=original.raw_json,
                source=original.source,
                ranks=ranks[key],
                canonical_key=original.canonical_key,
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)

    if max_per_source is not None:
        fused = _cap_per_source(fused, max_per_source)

    return fused


def _prefer_representation(
    candidate: RetrievalResult,
    existing: RetrievalResult,
) -> bool:
    candidate_priority = _REPRESENTATION_PRIORITY.get(candidate.source_type, 99)
    existing_priority = _REPRESENTATION_PRIORITY.get(existing.source_type, 99)
    if candidate_priority != existing_priority:
        return candidate_priority < existing_priority
    return len(candidate.text) > len(existing.text)


def _cap_per_source(
    results: list[RetrievalResult],
    cap: int,
) -> list[RetrievalResult]:
    """Keep at most *cap* items per source_type while preserving order."""
    from collections import Counter

    counts: Counter[str] = Counter()
    capped: list[RetrievalResult] = []
    for item in results:
        if counts[item.source_type] < cap:
            capped.append(item)
            counts[item.source_type] += 1
    return capped


def rrf_fusion_summary(fused: list[RetrievalResult]) -> dict[str, Any]:
    """Build a compact summary dict for tracing/debug."""
    return {
        "count": len(fused),
        "results": [
            {
                "key": item.key,
                "type": item.source_type,
                "score": round(item.score, 6),
                "ranks": item.ranks,
            }
            for item in fused[:20]
        ],
    }
