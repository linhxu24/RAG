import pytest

from app.constants import Intent
from app.generation.schemas import GeneratedResponse, ResultBody
from app.generation.validator import ResponseValidationError, ResponseValidator


def payload(text: str) -> dict:
    return GeneratedResponse(
        intent=Intent.PRODUCT_DETAIL,
        confidence=0.9,
        answer_type="direct_data",
        result=ResultBody(text=text),
    ).model_dump(mode="json")


def test_valid_schema_and_grounded_price():
    context = {
        "items": [
            {
                "source_id": "product-1",
                "text": "Sản phẩm Oral-B. Giá: 850000.00",
                "raw_json": {"price": 850000},
                "source": {},
            }
        ]
    }
    response = ResponseValidator().validate(
        payload("Giá sản phẩm là 850.000đ"),
        context=context,
    )
    assert response.intent == Intent.PRODUCT_DETAIL


def test_rejects_unsupported_price():
    context = {"items": [{"source_id": "x", "text": "Giá: 850000", "source": {}}]}
    with pytest.raises(ResponseValidationError, match="Unsupported price"):
        ResponseValidator().validate(payload("Giá là 999.000đ"), context=context)


def test_rejects_invalid_json():
    with pytest.raises(ResponseValidationError, match="Invalid JSON"):
        ResponseValidator().validate("{not-json", context={"items": []})


def test_normalizes_model_asset_objects_to_context_asset_uuid():
    data = payload("Ảnh: [asset:test_product_02]")
    data["result"]["items"] = [
        {
            "type": "product",
            "id": "product-1",
            "asset_ids": [{"id": "test_product_02", "token": "[asset:test_product_02]"}],
        }
    ]
    data["result"]["assets"] = [{"id": "test_product_02"}]
    data["result"]["missing_assets"] = [
        {"id": "test_product_02", "token": "[asset:test_product_02]"}
    ]
    context = {
        "items": [
            {
                "source_id": "product-1",
                "text": "Ảnh: [asset:test_product_02]",
                "raw_json": {"asset_id": "06f55b25-d810-4b6e-9be8-e2b04ab45d0e"},
                "source": {},
            }
        ]
    }

    response = ResponseValidator().validate(data, context=context)

    assert response.result.assets == []
    assert response.result.missing_assets == []
    assert response.result.items[0].asset_ids == [
        "06f55b25-d810-4b6e-9be8-e2b04ab45d0e"
    ]
