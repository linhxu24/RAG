import uuid

from app.db.models import Product, Service
from app.retrieval.structured_query import ServiceQuerySpec
from app.retrieval.structured_retriever import StructuredRetriever


def test_product_fuzzy_retrieval(monkeypatch):
    product = Product(
        product_id=uuid.uuid4(),
        name="Bàn chải điện Oral-B Pro 500",
        source_doc_id=uuid.uuid4(),
        status="active",
    )
    retriever = StructuredRetriever()
    monkeypatch.setattr(retriever, "list_products", lambda _session: [product])
    match = retriever.get_product(object(), "Oral-B Pro 500")
    assert match is not None
    assert match[0].name == product.name


def test_service_fuzzy_retrieval(monkeypatch):
    service = Service(
        service_id=uuid.uuid4(),
        name="Tẩy trắng răng",
        source_doc_id=uuid.uuid4(),
        status="active",
    )
    retriever = StructuredRetriever()
    monkeypatch.setattr(retriever, "list_services", lambda _session: [service])
    match = retriever.get_service(object(), "Tôi muốn tẩy trắng")
    assert match is not None
    assert match[0].name == service.name


def test_service_match_prefers_full_treatment_terms_over_shared_rang(monkeypatch):
    services = [
        Service(
            service_id=uuid.uuid4(),
            name=name,
            source_doc_id=uuid.uuid4(),
            status="active",
        )
        for name in (
            "Tẩy trắng răng tại phòng khám",
            "Trám răng thẩm mỹ",
            "Nhổ răng khôn",
            "Điều trị tủy răng",
        )
    ]
    retriever = StructuredRetriever()
    monkeypatch.setattr(retriever, "list_services", lambda _session: services)

    match = retriever.get_service(object(), "Dịch vụ tẩy trắng răng giá bao nhiêu?")

    assert match is not None
    assert match[0].name == "Tẩy trắng răng tại phòng khám"
    assert match[1] >= 0.9


def test_faq_match_score_prefers_shared_treatment_terms():
    retriever = StructuredRetriever()
    query = "Sau nhổ răng tôi nên ăn uống và vệ sinh như thế nào?"

    extraction_score = retriever._faq_match_score(
        query,
        "Sau khi nhổ răng cần kiêng gì?",
    )
    sensitivity_score = retriever._faq_match_score(
        query,
        "Tại sao răng bị ê buốt?",
    )

    assert extraction_score > sensitivity_score


def test_service_list_filters_by_category_code():
    implant = Service(
        service_id=uuid.uuid4(),
        name="Cấy ghép Implant đơn lẻ",
        category_code="IMPLANT",
        source_category="IMPLANT",
        source_doc_id=uuid.uuid4(),
        status="active",
    )
    captured = {}

    class _Scalars:
        def all(self):
            return [implant]

    class _Session:
        def scalars(self, statement):
            captured["statement"] = statement
            return _Scalars()

    results = StructuredRetriever().list_services(
        _Session(),
        ServiceQuerySpec(category_codes=("IMPLANT",), category_terms=("implant",)),
    )

    assert [service.name for service in results] == ["Cấy ghép Implant đơn lẻ"]
    statement_text = str(captured["statement"])
    assert "services.category_code" in statement_text
    assert "ORDER BY" in statement_text


def test_service_list_builds_sql_filter_for_source_category_term():
    implant = Service(
        service_id=uuid.uuid4(),
        name="Phục hình răng mất",
        category_code=None,
        source_category="Cấy ghép Implant",
        source_doc_id=uuid.uuid4(),
        status="active",
    )
    captured = {}

    class _Scalars:
        def all(self):
            return [implant]

    class _Session:
        def scalars(self, statement):
            captured["statement"] = statement
            return _Scalars()

    results = StructuredRetriever().list_services(
        _Session(),
        ServiceQuerySpec(category_codes=("IMPLANT",), category_terms=("implant",)),
    )

    assert [service.name for service in results] == ["Phục hình răng mất"]
    statement_text = str(captured["statement"])
    assert "simplydent_unaccent" in statement_text
    assert "services.source_category" in statement_text


def test_service_list_pushes_price_duration_and_sort_to_sql():
    captured = {}

    class _Scalars:
        def all(self):
            return []

    class _Session:
        def scalars(self, statement):
            captured["statement"] = statement
            return _Scalars()

    StructuredRetriever().list_services(
        _Session(),
        ServiceQuerySpec(
            price_max=1_000_000,
            duration_max=60,
            sort_by="duration",
            sort_direction="desc",
            limit=7,
        ),
    )

    statement_text = str(captured["statement"])
    assert "services.price <=" in statement_text
    assert "services.duration_minutes <=" in statement_text
    assert "ORDER BY services.duration_minutes DESC" in statement_text
    assert "LIMIT" in statement_text
