from dataclasses import dataclass
from typing import Any

from app.retrieval.normalization import normalize_vietnamese


@dataclass(frozen=True)
class DocumentClassification:
    document_type: str
    confidence: float
    reasons: list[str]
    requires_review: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_type": self.document_type,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "requires_review": self.requires_review,
        }


TYPE_MAP = {
    "product": "product_catalog",
    "service": "service_catalog",
    "faq": "faq",
    "clinic_info": "clinic_info",
    "policy": "policy",
    "unknown": "unknown",
}


def classify_document(
    *,
    requested_type: str,
    table_classifications: list[dict[str, Any]],
    text: str,
) -> DocumentClassification:
    inferred_types = {
        item.get("inferred_entity_type") or item.get("entity_type")
        for item in table_classifications
        if item.get("inferred_entity_type") or item.get("entity_type")
    }
    requested = requested_type.strip().lower()
    if requested != "auto":
        mapped = TYPE_MAP.get(requested, requested)
        conflict = bool(
            inferred_types
            and requested in TYPE_MAP
            and requested not in inferred_types
            and mapped not in inferred_types
        )
        return DocumentClassification(
            document_type=mapped,
            confidence=0.99 if not conflict else 0.6,
            reasons=[
                "explicit_document_type",
                *(["explicit_type_conflicts_with_detected_tables"] if conflict else []),
            ],
            requires_review=conflict or mapped == "unknown",
        )

    if len(inferred_types) == 1:
        entity_type = next(iter(inferred_types))
        confidence = min(
            (
                float(item.get("inferred_confidence") or item.get("confidence") or 0)
                for item in table_classifications
                if (item.get("inferred_entity_type") or item.get("entity_type"))
                == entity_type
            ),
            default=0.0,
        )
        return DocumentClassification(
            TYPE_MAP.get(str(entity_type), str(entity_type)),
            confidence,
            [f"all_detected_tables_are_{entity_type}"],
            confidence < 0.85,
        )
    if len(inferred_types) > 1:
        return DocumentClassification(
            "mixed_business_data",
            0.95,
            ["multiple_supported_table_types_detected"],
        )

    normalized = normalize_vietnamese(text)
    signals = {
        "policy": ("chinh sach", "bao hanh", "thanh toan", "bao mat", "dieu khoan"),
        "clinic_info": ("dia chi", "gio lam viec", "hotline", "phong kham"),
        "faq": ("cau hoi", "thuong gap", "tai sao", "lam the nao"),
    }
    matches = [
        document_type
        for document_type, phrases in signals.items()
        if any(phrase in normalized for phrase in phrases)
    ]
    if len(matches) == 1:
        return DocumentClassification(
            matches[0],
            0.72,
            [f"text_signal_{matches[0]}"],
            True,
        )
    return DocumentClassification(
        "general_document" if normalized else "unknown",
        0.5 if normalized else 0.0,
        ["no_supported_business_table_detected"],
        True,
    )
