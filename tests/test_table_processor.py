from app.ingestion.table_normalizer import normalize_table
from app.ingestion.table_processor import (
    canonicalize_row,
    detect_entity_type,
    parse_decimal,
    serialize_row,
)


def test_product_table_detection_and_row_serialization():
    rows = [{"Tên sản phẩm": "Oral-B", "Giá": 850000, "Số lượng": 3}]
    assert detect_entity_type(rows, "Bảng sản phẩm") == "product"
    canonical = canonicalize_row(rows[0])
    assert canonical["name"] == "Oral-B"
    assert canonical["price"] == 850000
    assert "Tên sản phẩm: Oral-B" in serialize_row(rows[0])


def test_service_table_detection():
    rows = [{"Tên dịch vụ": "Tẩy trắng răng", "Thời gian": 60, "Giá": 1200000}]
    assert detect_entity_type(rows, "Dịch vụ") == "service"
    canonical = canonicalize_row(rows[0])
    assert canonical["service_name"] == "Tẩy trắng răng"
    assert canonical["duration"] == 60


def test_vietnamese_faq_table_detection():
    rows = [{"Câu hỏi": "Có đau không?", "Câu trả lời": "Tùy tình trạng."}]
    assert detect_entity_type(rows, "FAQ") == "faq"
    canonical = canonicalize_row(rows[0])
    assert canonical["question"] == "Có đau không?"
    assert canonical["answer"] == "Tùy tình trạng."


def test_positional_clinic_table_is_normalized_to_key_value():
    rows = [
        {"0": "Tên nha khoa", "1": "SimplyDent"},
        {"0": "Số điện thoại", "1": "0909123456"},
        {"0": "Địa chỉ", "1": "TP.HCM"},
    ]
    normalized = normalize_table(rows, "table_1")
    assert normalized.classification.entity_type == "clinic_info"
    assert normalized.classification.confidence >= 0.9
    assert normalized.rows[0] == {"key": "Tên nha khoa", "value": "SimplyDent"}


def test_opening_hours_table_is_normalized_to_clinic_info():
    normalized = normalize_table(
        [
            {"Khung giờ": "Sáng", "Mô tả": "8:00 - 12:00"},
            {"Khung giờ": "Chiều", "Mô tả": "13:00 - 17:00"},
        ],
        "table_2",
    )
    assert normalized.classification.entity_type == "clinic_info"
    assert normalized.rows[0] == {
        "key": "Giờ làm việc - Sáng",
        "value": "8:00 - 12:00",
    }


def test_unknown_table_requires_review():
    normalized = normalize_table([{"Alpha": "A", "Beta": "B"}], "table_1")
    assert normalized.classification.entity_type is None
    assert normalized.classification.requires_review is True


def test_vietnamese_prices_are_parsed_as_thousands():
    assert parse_decimal("850.000đ") == 850000
    assert parse_decimal("1.200.000 VNĐ") == 1200000
    assert parse_decimal("1.200,50") == parse_decimal("1200.50")


def test_extended_product_columns_are_canonicalized():
    normalized = normalize_table(
        [
            {
                "Tên sản phẩm": "Oral-B Pro 500",
                "Thương hiệu": "Oral-B",
                "Loại": "Bàn chải điện",
                "Giá": "850000",
                "Tên file ảnh": "oral_b_pro_500.png",
            }
        ],
        "products",
    )

    canonical = canonicalize_row(normalized.rows[0])
    assert canonical["brand"] == "Oral-B"
    assert canonical["image_reference"] == "oral_b_pro_500.png"


def test_generic_name_with_service_columns_is_classified_as_service():
    normalized = normalize_table(
        [
            {
                "name": "Nhổ răng khôn",
                "category": "ORAL_SURGERY",
                "duration_minutes": 60,
                "price": 2500000,
                "symptoms": "đau vùng răng khôn|sưng nướu",
            }
        ],
        "random-upload-name",
    )

    assert normalized.classification.entity_type == "service"
    assert normalized.classification.confidence >= 0.9
    assert normalized.classification.requires_review is False


def test_generic_name_with_product_columns_is_classified_as_product():
    normalized = normalize_table(
        [
            {
                "name": "AquaJet Mini",
                "category": "WATER_FLOSSER",
                "brand": "AquaJet",
                "model": "Mini",
                "price": 1250000,
                "quantity": 10,
            }
        ],
        "random-upload-name",
    )

    assert normalized.classification.entity_type == "product"
    assert normalized.classification.confidence >= 0.9


def test_generic_name_with_only_shared_columns_is_not_forced_to_product():
    normalized = normalize_table(
        [
            {
                "name": "Dữ liệu chưa rõ loại",
                "category": "OTHER",
                "price": 100000,
            }
        ],
        "random-upload-name",
    )

    assert normalized.classification.entity_type is None
    assert normalized.classification.requires_review is True
