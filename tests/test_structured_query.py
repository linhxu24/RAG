from decimal import Decimal

from app.retrieval import structured_query


def test_product_query_parses_price_sort(monkeypatch):
    monkeypatch.setattr(
        structured_query,
        "_category_codes",
        lambda _session, _query: ["ELECTRIC_TOOTHBRUSH"],
    )
    monkeypatch.setattr(
        structured_query,
        "_product_ids",
        lambda _session, _query: [],
    )

    result = structured_query.parse_product_query(
        object(),
        "Cho tôi bàn chải điện dưới 2 triệu, giá từ thấp đến cao",
    )

    assert result.category_codes == ("ELECTRIC_TOOTHBRUSH",)
    assert result.price_max == Decimal("2000000")
    assert result.sort_by == "price"
    assert result.sort_direction == "asc"
    assert result.needs_clarification is False


def test_product_query_requests_sort_direction(monkeypatch):
    monkeypatch.setattr(structured_query, "_category_codes", lambda *_: [])
    monkeypatch.setattr(structured_query, "_product_ids", lambda *_: [])

    result = structured_query.parse_product_query(
        object(),
        "Sắp xếp danh sách sản phẩm theo giá",
    )

    assert result.needs_clarification is True
    assert "tăng dần hay giảm dần" in (result.clarification_message or "")
