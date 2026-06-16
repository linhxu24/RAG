from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.ingestion.table_normalizer import TableClassification, canonicalize_row
from app.ingestion.table_processor import parse_decimal
from app.taxonomy import normalize_label, resolve_category


@dataclass(frozen=True)
class BusinessValidationReport:
    blocking_reasons: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
        }


def validate_business_rows(
    session: Session,
    rows: list[dict[str, Any]],
    classification: TableClassification,
    *,
    table_index: int,
) -> BusinessValidationReport:
    blocking: list[str] = []
    warnings: list[str] = []
    entity_type = classification.entity_type
    for row_index, raw_row in enumerate(rows, start=1):
        canonical = canonicalize_row(raw_row)
        prefix = f"table_{table_index}_row_{row_index}"
        name = str(
            canonical.get("service_name") or canonical.get("name") or ""
        ).strip()
        category = str(canonical.get("category") or "").strip()

        if entity_type in {"product", "service"}:
            if not name:
                blocking.append(f"{prefix}_missing_name")
            if not category:
                blocking.append(f"{prefix}_missing_category")
            else:
                resolution = resolve_category(session, entity_type, category)
                if resolution.code is None:
                    blocking.append(f"{prefix}_category_unrecognized")
                if name and normalize_label(name) == normalize_label(category):
                    blocking.append(f"{prefix}_category_equals_name")
                if len(category) > 80:
                    blocking.append(f"{prefix}_category_too_long")
            raw_price = canonical.get("price")
            if raw_price not in (None, "") and parse_decimal(raw_price) is None:
                blocking.append(f"{prefix}_invalid_price")

        if entity_type == "faq":
            if not canonical.get("question") or not canonical.get("answer"):
                blocking.append(f"{prefix}_faq_missing_question_or_answer")
            if category and resolve_category(session, "faq", category).code is None:
                blocking.append(f"{prefix}_category_unrecognized")
            if not category:
                warnings.append(f"{prefix}_faq_category_missing")

    return BusinessValidationReport(
        list(dict.fromkeys(blocking)),
        list(dict.fromkeys(warnings)),
    )
