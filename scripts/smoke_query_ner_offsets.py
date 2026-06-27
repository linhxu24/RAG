"""Smoke-check query-time fallback NER returns explicit offsets."""

from app.config import get_settings
from app.ner.entity_span_extractor import EntitySpanExtractor


def main() -> None:
    settings = get_settings().model_copy(update={"enable_gliner_ner": False})
    extractor = EntitySpanExtractor(settings)
    query = "Cho tôi giá AquaJet Mini Water Flosser"
    result = extractor.extract(
        query,
        known_products=["AquaJet Mini Water Flosser"],
        known_services=[],
    )
    span = next(item for item in result.spans if item.label == "product_name")
    assert span.start is not None
    assert span.end is not None
    assert query[span.start : span.end] == "AquaJet Mini Water Flosser"
    print(span.as_dict())


if __name__ == "__main__":
    main()
