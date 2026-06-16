from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.types import RetrievalResult


def _result(
    identifier: str,
    source_type: str = "chunk",
    canonical_key: str | None = None,
    text: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        source_type=source_type,
        source_id=identifier,
        text=text or identifier,
        score=1.0,
        canonical_key=canonical_key,
    )


def test_rrf_rewards_items_found_by_multiple_retrievers():
    fused = reciprocal_rank_fusion(
        {
            "dense": [_result("a"), _result("b")],
            "sparse": [_result("b"), _result("c")],
        },
        k=60,
    )
    assert fused[0].source_id == "b"
    assert fused[0].ranks == {"dense": 2, "sparse": 1}


def test_rrf_weighted_structured_boosted():
    """structured retriever items should rank higher when weight > 1."""
    fused = reciprocal_rank_fusion(
        {
            "structured": [_result("product_1", source_type="product")],
            "dense_original_chunk": [_result("chunk_1"), _result("chunk_2")],
        },
        k=60,
        weights={"structured": 1.5, "dense_original_chunk": 1.0},
    )
    assert fused[0].source_id == "product_1"


def test_rrf_canonical_dedup_merges_same_key():
    """Items with the same canonical_key should be merged, summing scores."""
    # Two different retrievers find the same product via different row IDs
    product_key = "product:abc-123"
    fused = reciprocal_rank_fusion(
        {
            "dense": [
                _result("row_1", canonical_key=product_key, text="short"),
            ],
            "sparse": [
                _result("row_2", canonical_key=product_key, text="longer description"),
            ],
        },
        k=60,
    )
    # Should be merged into one entry
    product_results = [r for r in fused if r.key == product_key]
    assert len(product_results) == 1
    # Should keep the longer text representation
    assert product_results[0].text == "longer description"
    # Both retrievers should appear in ranks
    assert "dense" in product_results[0].ranks
    assert "sparse" in product_results[0].ranks


def test_rrf_canonical_dedup_keeps_separate_keys():
    """Items with different canonical_keys must remain separate."""
    fused = reciprocal_rank_fusion(
        {
            "dense": [_result("row_1", canonical_key="product:aaa")],
            "sparse": [_result("row_2", canonical_key="product:bbb")],
        },
        k=60,
    )
    keys = {r.key for r in fused}
    assert "product:aaa" in keys
    assert "product:bbb" in keys


def test_rrf_canonical_dedup_prefers_structured_representation():
    product_key = "product:abc-123"
    fused = reciprocal_rank_fusion(
        {
            "structured": [
                _result(
                    "product_1",
                    source_type="product",
                    canonical_key=product_key,
                    text="normalized product",
                )
            ],
            "dense_table_row": [
                _result(
                    "row_1",
                    source_type="table_row",
                    canonical_key=product_key,
                    text="a much longer table row representation of the product",
                )
            ],
        }
    )

    assert fused[0].source_type == "product"
    assert fused[0].source_id == "product_1"
    assert fused[0].ranks == {"structured": 1, "dense_table_row": 1}


def test_rrf_max_per_source_cap():
    """max_per_source should limit how many items of each source_type appear."""
    fused = reciprocal_rank_fusion(
        {
            "dense": [_result(f"chunk_{i}", source_type="chunk") for i in range(10)],
            "sparse": [_result("faq_1", source_type="faq")],
        },
        k=60,
        max_per_source=3,
    )
    chunk_count = sum(1 for r in fused if r.source_type == "chunk")
    assert chunk_count == 3
    # faq should still be present
    assert any(r.source_type == "faq" for r in fused)


def test_rrf_no_cap_when_none():
    """When max_per_source is None, all items pass through."""
    fused = reciprocal_rank_fusion(
        {
            "dense": [_result(f"chunk_{i}", source_type="chunk") for i in range(10)],
        },
        k=60,
        max_per_source=None,
    )
    assert len(fused) == 10
