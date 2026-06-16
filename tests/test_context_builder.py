from app.retrieval.context_builder import ContextBuilder
from app.retrieval.types import RetrievalResult


def _result(
    identifier: str,
    source_type: str = "chunk",
    text: str = "sample text",
    score: float = 0.9,
    canonical_key: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        source_type=source_type,
        source_id=identifier,
        text=text,
        score=score,
        canonical_key=canonical_key,
    )


class TestContextBuilder:
    def test_basic_build(self):
        builder = ContextBuilder(max_chars=10_000, max_items_per_source=4)
        results = [_result("1"), _result("2", text="different text")]
        ctx = builder.build(results)
        assert len(ctx["items"]) == 2
        assert ctx["total_chars"] > 0

    def test_dedup_by_text(self):
        """Duplicate text content should be de-duplicated."""
        builder = ContextBuilder()
        results = [
            _result("1", text="same content"),
            _result("2", text="same content"),
        ]
        ctx = builder.build(results)
        assert len(ctx["items"]) == 1

    def test_dedup_by_canonical_key(self):
        """Items with same canonical_key but different text are de-duplicated."""
        builder = ContextBuilder()
        results = [
            _result("row_1", text="product from dense", canonical_key="product:abc"),
            _result("row_2", text="product from sparse", canonical_key="product:abc"),
        ]
        ctx = builder.build(results)
        assert len(ctx["items"]) == 1

    def test_per_source_cap(self):
        builder = ContextBuilder(max_items_per_source=2)
        results = [
            _result(str(i), text=f"chunk content {i}")
            for i in range(10)
        ]
        ctx = builder.build(results)
        assert len(ctx["items"]) == 2
        assert ctx["source_counts"]["chunk"] == 2

    def test_limits_can_be_disabled_for_direct_sql_lists(self):
        builder = ContextBuilder(max_chars=20, max_items_per_source=2, max_item_chars=10)
        results = [
            _result(
                str(index),
                source_type="service",
                text=f"service description {index}",
            )
            for index in range(6)
        ]

        ctx = builder.build(results, apply_limits=False)

        assert len(ctx["items"]) == 6
        assert ctx["source_counts"]["service"] == 6

    def test_skip_oversized_items(self):
        """Items exceeding max_item_chars should be skipped, not abort build."""
        builder = ContextBuilder(max_chars=100_000, max_item_chars=50)
        results = [
            _result("short", text="short text"),
            _result("long", text="x" * 100),  # exceeds 50 char limit
            _result("also_short", text="also short"),
        ]
        ctx = builder.build(results)
        ids = [item["source_id"] for item in ctx["items"]]
        assert "short" in ids
        assert "also_short" in ids
        assert "long" not in ids
        assert ctx["skipped_long"] == 1

    def test_budget_exhaustion_continues(self):
        """When char budget runs out, remaining items should be skipped gracefully."""
        builder = ContextBuilder(max_chars=30, max_items_per_source=10)
        results = [
            _result("1", text="twelve chars"),   # 12 chars
            _result("2", text="more twelve!"),    # 12 chars, total 24
            _result("3", text="this is a longer text that exceeds budget"),
            _result("4", text="tiny"),            # 4 chars, total 28
        ]
        ctx = builder.build(results)
        assert ctx["total_chars"] <= 30

    def test_source_priority_sorting(self):
        """Products and services should appear before chunks in context."""
        builder = ContextBuilder()
        results = [
            _result("c1", source_type="chunk", text="chunk text"),
            _result("p1", source_type="product", text="product text"),
            _result("f1", source_type="faq", text="faq text"),
        ]
        ctx = builder.build(results)
        types = [item["source_type"] for item in ctx["items"]]
        assert types.index("product") < types.index("faq")
        assert types.index("faq") < types.index("chunk")

    def test_empty_input(self):
        builder = ContextBuilder()
        ctx = builder.build([])
        assert ctx["items"] == []
        assert ctx["total_chars"] == 0

    def test_skipped_dup_count(self):
        builder = ContextBuilder()
        results = [
            _result("1", text="same"),
            _result("2", text="same"),
            _result("3", text="same"),
        ]
        ctx = builder.build(results)
        assert len(ctx["items"]) == 1
        assert ctx["skipped_dup"] == 2
