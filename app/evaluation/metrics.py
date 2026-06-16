import math
from collections.abc import Sequence


def hit_at(retrieved: Sequence[str], expected: set[str], k: int) -> float | None:
    if not expected:
        return None
    return float(bool(set(retrieved[:k]) & expected))


def recall_at(retrieved: Sequence[str], expected: set[str], k: int) -> float | None:
    if not expected:
        return None
    return len(set(retrieved[:k]) & expected) / len(expected)


def reciprocal_rank(
    retrieved: Sequence[str],
    expected: set[str],
    k: int = 10,
) -> float | None:
    if not expected:
        return None
    for rank, identifier in enumerate(retrieved[:k], start=1):
        if identifier in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at(retrieved: Sequence[str], expected: set[str], k: int = 10) -> float | None:
    if not expected:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, identifier in enumerate(retrieved[:k], start=1)
        if identifier in expected
    )
    ideal_count = min(len(expected), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float | None]) -> float | None:
    applicable = [value for value in values if value is not None]
    return sum(applicable) / len(applicable) if applicable else None
