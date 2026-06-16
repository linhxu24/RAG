from typing import Any

from app.ingestion.review_policy import split_review_reasons


def build_quality_report(
    *,
    chunks: list[str],
    tables_found: int,
    table_rows_created: int,
    assets_found: int,
    assets_resolved: int,
    products_created: int,
    services_created: int,
    embedding_success: int,
    embedding_failed: int,
    embedding_backend: str,
    table_classifications: list[dict[str, Any]],
    classification_threshold: float,
    review_reasons: list[str],
    asset_extraction_errors: list[dict[str, Any]],
    stage_traces: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings: list[str] = []
    empty_chunks = sum(not chunk.strip() for chunk in chunks)
    if empty_chunks:
        warnings.append(f"{empty_chunks} empty chunks detected")
    if assets_found != assets_resolved:
        warnings.append("Some extracted assets were not persisted")
    if embedding_failed:
        warnings.append(f"{embedding_failed} embeddings failed")
    if embedding_backend == "deterministic_hash_fallback":
        warnings.append("Embedding model unavailable; deterministic hash fallback was used")
    if asset_extraction_errors:
        warnings.append(f"{len(asset_extraction_errors)} assets could not be extracted")
    unknown_tables = sum(not item.get("entity_type") for item in table_classifications)
    low_confidence_tables = sum(
        float(item.get("confidence") or 0) < classification_threshold
        for item in table_classifications
    )
    reason_split = split_review_reasons(review_reasons)
    return {
        "total_chunks": len(chunks),
        "empty_chunks": empty_chunks,
        "tables_found": tables_found,
        "table_rows_created": table_rows_created,
        "assets_found": assets_found,
        "assets_resolved": assets_resolved,
        "products_created": products_created,
        "services_created": services_created,
        "embedding_success": embedding_success,
        "embedding_failed": embedding_failed,
        "embedding_backend": embedding_backend,
        "table_classifications": table_classifications,
        "unknown_tables": unknown_tables,
        "low_confidence_tables": low_confidence_tables,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "review_only_reasons": reason_split.review_only,
        "approval_blocking_reasons": reason_split.integrity_blockers,
        "asset_extraction_errors": asset_extraction_errors,
        "warnings": warnings,
        "stage_traces": stage_traces,
    }
