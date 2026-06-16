from app.ingestion.document_classifier import classify_document


def test_auto_detects_product_catalog_from_tables():
    result = classify_document(
        requested_type="auto",
        table_classifications=[
            {
                "entity_type": "product",
                "confidence": 0.96,
                "inferred_entity_type": "product",
                "inferred_confidence": 0.96,
            }
        ],
        text="",
    )

    assert result.document_type == "product_catalog"
    assert result.confidence == 0.96
    assert result.requires_review is False


def test_explicit_type_conflict_requires_review():
    result = classify_document(
        requested_type="service",
        table_classifications=[
            {
                "entity_type": "service",
                "inferred_entity_type": "product",
                "inferred_confidence": 0.96,
            }
        ],
        text="",
    )

    assert result.document_type == "service_catalog"
    assert result.requires_review is True
