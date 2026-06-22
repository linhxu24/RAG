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


def test_product_list_ignores_empty_sort_payload(monkeypatch):
    monkeypatch.setattr(structured_query, "_category_codes", lambda *_: [])
    monkeypatch.setattr(structured_query, "_product_ids", lambda *_: [])

    result = structured_query.parse_product_query(
        object(),
        "Cho tôi danh sách sản phẩm đang có",
        sort={"field": None, "direction": None},
    )

    assert result.needs_clarification is False
    assert result.sort_by == "category"
    assert result.sort_direction == "asc"


def test_service_query_parses_implant_category(monkeypatch):
    monkeypatch.setattr(
        structured_query,
        "_service_category_match",
        lambda _session, _query: (["IMPLANT"], ["implant", "trong rang"]),
    )
    monkeypatch.setattr(
        structured_query,
        "_service_ids",
        lambda _session, _query: [],
    )

    result = structured_query.parse_service_query(
        object(),
        "Dịch vụ nào liên quan đến implant?",
    )

    assert result.category_codes == ("IMPLANT",)
    assert result.category_terms == ("implant", "trong rang")
    assert result.needs_clarification is False


def test_service_query_parses_duration_sort(monkeypatch):
    monkeypatch.setattr(structured_query, "_service_category_match", lambda *_: ([], []))
    monkeypatch.setattr(structured_query, "_service_ids", lambda *_: [])

    result = structured_query.parse_service_query(
        object(),
        "Sắp xếp dịch vụ theo thời gian ngắn nhất",
    )

    assert result.sort_by == "duration"
    assert result.sort_direction == "asc"
    assert result.needs_clarification is False


def test_service_price_filter_does_not_become_duration_filter(monkeypatch):
    monkeypatch.setattr(
        structured_query,
        "_service_category_match",
        lambda _session, _query: (["IMPLANT"], ["implant"]),
    )
    monkeypatch.setattr(structured_query, "_service_ids", lambda *_: [])

    result = structured_query.parse_service_query(
        object(),
        "Dịch vụ nào liên quan đến implant và dưới 2 triệu?",
    )

    assert result.category_codes == ("IMPLANT",)
    assert result.price_max == Decimal("2000000")
    assert result.duration_max is None


def test_product_query_uses_llm_constraints(monkeypatch):
    monkeypatch.setattr(structured_query, "_category_codes", lambda *_: [])
    monkeypatch.setattr(structured_query, "_product_ids", lambda *_: [])

    result = structured_query.parse_product_query(
        object(),
        "Có sản phẩm làm trắng răng còn hàng không?",
        constraints={
            "feature_terms": ["làm trắng răng"],
            "stock": "available",
            "price": {"max": "500k"},
        },
        sort={"field": "price", "direction": "asc"},
        limit=12,
    )

    assert result.feature_terms == ("làm trắng răng", "lam trang")
    assert result.quantity_min == 1
    assert result.price_max == Decimal("500000")
    assert result.sort_by == "price"
    assert result.limit == 12


def test_service_query_uses_llm_constraints(monkeypatch):
    monkeypatch.setattr(
        structured_query,
        "_service_category_match",
        lambda _session, query: (["IMPLANT"], ["implant"]) if "implant" in query else ([], []),
    )
    monkeypatch.setattr(structured_query, "_service_ids", lambda *_: [])

    result = structured_query.parse_service_query(
        object(),
        "Có dịch vụ implant nào không?",
        constraints={
            "category_terms": ["implant"],
            "feature_terms": ["phục hình cố định"],
            "duration": {"max": 120},
        },
    )

    assert result.category_codes == ("IMPLANT",)
    assert result.category_terms == ("implant",)
    assert result.feature_terms == ("phục hình cố định", "implant")
    assert result.duration_max == 120
