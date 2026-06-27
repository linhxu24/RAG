"""Smoke-check generation validator rejects IDs outside current context."""

import uuid

from app.constants import Intent
from app.generation.validator import ResponseValidationError, ResponseValidator


def main() -> None:
    allowed_id = str(uuid.uuid4())
    outside_id = str(uuid.uuid4())
    context = {
        "items": [
            {
                "source_type": "product",
                "source_id": allowed_id,
                "text": "Sản phẩm: AquaJet",
                "raw_json": {"name": "AquaJet"},
                "source": {},
            }
        ],
        "total_chars": 18,
    }
    payload = {
        "intent": Intent.PRODUCT_DETAIL.value,
        "confidence": 1.0,
        "answer_type": "rag",
        "entities": [],
        "result": {
            "text": "AquaJet",
            "items": [{"type": "product", "id": outside_id}],
            "sources": [{"source_type": "product", "source_id": outside_id}],
        },
        "safety": {
            "medical_disclaimer_required": False,
            "needs_human_support": False,
        },
    }
    try:
        ResponseValidator().validate(payload, context=context, session=None)
    except ResponseValidationError as exc:
        print({"rejected": True, "error": str(exc)})
        return
    raise SystemExit("Validator accepted an ID outside retrieved context")


if __name__ == "__main__":
    main()
