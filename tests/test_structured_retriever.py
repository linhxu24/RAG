import uuid

from app.db.models import Product, Service
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
