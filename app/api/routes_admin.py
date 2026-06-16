import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.admin.data_reset import delete_document, reset_data
from app.config import get_settings
from app.db.models import FAQ, Product, RagTrace, RagTraceStep, Service
from app.db.session import get_db
from app.retrieval.structured_retriever import StructuredRetriever

router = APIRouter(tags=["business", "admin"])
retriever = StructuredRetriever()


class DataResetRequest(BaseModel):
    scope: Literal["content", "runtime"] = "content"
    confirmation: str


def _business_payload(result) -> dict[str, Any]:
    return result.as_dict()


@router.get("/products")
def list_products(session: Session = Depends(get_db)) -> dict:
    items = [
        retriever._product_result(session, product, 1.0).as_dict()
        for product in retriever.list_products(session)
    ]
    return {"items": items, "count": len(items)}


@router.get("/products/{product_id_or_name}")
def product_detail(product_id_or_name: str, session: Session = Depends(get_db)) -> dict:
    product = None
    try:
        product = session.get(Product, uuid.UUID(product_id_or_name))
        if product and product.status != "active":
            product = None
    except ValueError:
        match = retriever.get_product(session, product_id_or_name)
        product = match[0] if match else None
    if product is None:
        raise HTTPException(status_code=404, detail="Active product not found")
    return _business_payload(retriever._product_result(session, product, 1.0))


@router.get("/services")
def list_services(session: Session = Depends(get_db)) -> dict:
    items = [
        retriever._service_result(session, service, 1.0).as_dict()
        for service in retriever.list_services(session)
    ]
    return {"items": items, "count": len(items)}


@router.get("/services/{service_id_or_name}")
def service_detail(service_id_or_name: str, session: Session = Depends(get_db)) -> dict:
    service = None
    try:
        service = session.get(Service, uuid.UUID(service_id_or_name))
        if service and service.status != "active":
            service = None
    except ValueError:
        match = retriever.get_service(session, service_id_or_name)
        service = match[0] if match else None
    if service is None:
        raise HTTPException(status_code=404, detail="Active service not found")
    return _business_payload(retriever._service_result(session, service, 1.0))


@router.get("/faqs")
def public_faqs(
    search: str | None = Query(default=None),
    category: str | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    statement = select(FAQ).where(FAQ.is_active.is_(True))
    if category:
        statement = statement.where(FAQ.category_code == category)
    items = session.scalars(statement.order_by(FAQ.category, FAQ.question)).all()
    if search:
        normalized = search.lower().strip()
        items = [
            item
            for item in items
            if normalized in item.question.lower()
            or normalized in item.answer.lower()
            or any(normalized in keyword.lower() for keyword in (item.keywords or []))
        ]
    return {
        "items": [
            {
                "faq_id": str(item.faq_id),
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "category_code": item.category_code,
                "keywords": item.keywords or [],
            }
            for item in items
        ],
        "count": len(items),
    }


@router.delete("/documents/{doc_id}")
@router.delete("/api/documents/{doc_id}")
def hard_delete_document(
    doc_id: uuid.UUID,
    confirm: bool = Query(default=False),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true is required")
    try:
        return delete_document(session, get_settings(), doc_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/admin/data-reset")
def reset_application_data(
    request: DataResetRequest,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    if request.confirmation != "DELETE SIMPLYDENT CONTENT":
        raise HTTPException(status_code=400, detail="Invalid reset confirmation phrase")
    return reset_data(session, get_settings(), request.scope)


@router.get("/traces/{trace_id}")
def get_trace(
    trace_id: uuid.UUID,
    include_payloads: bool = Query(default=False),
    session: Session = Depends(get_db),
) -> dict:
    trace = session.get(RagTrace, trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    steps = session.scalars(
        select(RagTraceStep)
        .where(RagTraceStep.trace_id == trace_id)
        .order_by(RagTraceStep.created_at)
    ).all()
    return {
        "trace_id": str(trace.trace_id),
        "session_id": trace.session_id,
        "user_query": trace.user_query if include_payloads else "[hidden]",
        "intent": trace.detected_intent,
        "confidence": trace.confidence,
        "total_latency_ms": trace.total_latency_ms,
        "status": trace.status,
        "created_at": trace.created_at,
        "final_answer": trace.final_answer if include_payloads else None,
        "steps": [
            {
                "step_name": step.step_name,
                "status": step.status,
                "latency_ms": step.latency_ms,
                "error_message": step.error_message,
                "input": step.input_json if include_payloads else None,
                "output": step.output_json if include_payloads else None,
            }
            for step in steps
        ],
    }


@router.get("/traces")
def trace_summary(
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_db),
) -> dict:
    traces = session.scalars(
        select(RagTrace).order_by(RagTrace.created_at.desc()).limit(limit)
    ).all()
    status_counts = dict(
        session.execute(select(RagTrace.status, func.count()).group_by(RagTrace.status)).all()
    )
    return {
        "status_counts": status_counts,
        "items": [
            {
                "trace_id": str(trace.trace_id),
                "intent": trace.detected_intent,
                "confidence": trace.confidence,
                "latency_ms": trace.total_latency_ms,
                "status": trace.status,
                "created_at": trace.created_at,
            }
            for trace in traces
        ],
    }
