from app.config import Settings
from app.ner.entity_span_extractor import EntitySpanExtractor


def test_gliner_warmup_reports_disabled_without_loading_model():
    extractor = EntitySpanExtractor(Settings(enable_gliner_ner=False))

    result = extractor.warmup()

    assert result["enabled"] is False
    assert result["loaded"] is False
    assert result["error"] is None


def test_fallback_span_extractor_matches_explicit_service_name():
    extractor = EntitySpanExtractor(Settings(enable_gliner_ner=False))

    result = extractor.extract(
        "Dịch vụ tẩy trắng răng giá bao nhiêu?",
        known_services=["Tẩy trắng răng tại phòng khám"],
    )

    service_spans = [span for span in result.spans if span.label == "service_name"]
    assert service_spans
    assert service_spans[0].metadata["catalog_name"] == (
        "Tẩy trắng răng tại phòng khám"
    )


def test_fallback_span_extractor_does_not_invent_entity_for_follow_up():
    extractor = EntitySpanExtractor(Settings(enable_gliner_ner=False))

    result = extractor.extract(
        "Mất bao lâu?",
        known_services=[
            "Tẩy trắng răng tại phòng khám",
            "Mặt dán sứ Veneer",
        ],
    )

    assert not [
        span
        for span in result.spans
        if span.label in {"service_name", "product_name"}
    ]


def test_fallback_span_extractor_matches_mixed_reference_explicit_entity():
    extractor = EntitySpanExtractor(Settings(enable_gliner_ner=False))

    result = extractor.extract(
        "So sánh nó với EnamelGuard Sensitive Toothpaste",
        known_products=[
            "FreshMint Total Protection Toothpaste",
            "EnamelGuard Sensitive Toothpaste",
        ],
    )

    product_spans = [span for span in result.spans if span.label == "product_name"]
    assert [span.metadata["catalog_name"] for span in product_spans] == [
        "EnamelGuard Sensitive Toothpaste"
    ]
