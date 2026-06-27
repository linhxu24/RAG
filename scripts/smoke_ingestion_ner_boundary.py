"""Smoke-check ingestion-time entity extraction schema."""

from app.config import get_settings
from app.ner.entity_span_extractor import EntitySpanExtractor


def main() -> None:
    settings = get_settings().model_copy(update={"enable_gliner_ner": False})
    extractor = EntitySpanExtractor(settings)
    result = extractor.extract_for_ingestion(
        text_blocks=[
            {
                "text": "AquaJet Mini Water Flosser là máy tăm nước đang bán.",
                "page_number": 1,
            }
        ],
        table_rows=[
            {
                "_entity_type": "product",
                "name": "AquaJet Mini Water Flosser",
                "category": "Máy tăm nước",
                "brand": "AquaJet",
            }
        ],
        known_products=["AquaJet Mini Water Flosser"],
        known_services=[],
    )
    payload = result.as_dict()
    labels = set(payload["labels"])
    assert "product_name" in labels
    assert payload["total_mentions"] >= 1
    assert all("source_type" in mention for mention in payload["mentions"])
    print(payload)


if __name__ == "__main__":
    main()
