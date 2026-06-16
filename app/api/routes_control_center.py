import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import (
    FAQ,
    Asset,
    Chunk,
    ClinicInfo,
    Document,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationRun,
    IngestionRun,
    ParsedTable,
    Product,
    RagTrace,
    RagTraceStep,
    Service,
    TableRow,
)
from app.db.session import check_database, get_db
from app.evaluation.diagnostics import build_diagnostics
from app.generation.ollama_client import OllamaClient
from app.ingestion.embedder import EmbeddingService
from app.ingestion.pipeline import IngestionOptions, IngestionPipeline
from app.ingestion.review import (
    ApprovalValidationError,
    approve_document_records,
    set_document_status,
)
from app.observability.metrics import percentile
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.entity_resolver import DatabaseEntityResolver
from app.retrieval.planner import RetrievalPlanner
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import OptionalReranker
from app.retrieval.router import IntentRouter
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.structured_retriever import StructuredRetriever
from app.retrieval.types import RetrievalResult

router = APIRouter(prefix="/api", tags=["control-center"])


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _retrieval_item(item: RetrievalResult, rank: int) -> dict[str, Any]:
    return {
        "id": item.source_id,
        "type": item.source_type,
        "content": item.text,
        "score": item.score,
        "rank": rank,
        "source": item.source,
        "metadata": item.raw_json,
        "ranks": item.ranks,
        "canonical_key": item.canonical_key,
    }


def _asset_item(asset: Asset) -> dict[str, Any]:
    return {
        "asset_id": str(asset.asset_id),
        "stable_asset_key": asset.stable_asset_key,
        "asset_token": asset.asset_token,
        "asset_type": asset.asset_type,
        "doc_id": str(asset.doc_id),
        "chunk_id": str(asset.chunk_id) if asset.chunk_id else None,
        "local_path": asset.local_path,
        "public_url": asset.public_url,
        "page_number": asset.page_number,
        "bbox": asset.bbox,
        "status": asset.status,
        "metadata": asset.metadata_json,
        "local_file_exists": bool(asset.local_path and Path(asset.local_path).is_file()),
    }


def _counts(session: Session, doc_id: uuid.UUID) -> dict[str, int]:
    models = {"chunks": Chunk, "tables": ParsedTable, "assets": Asset}
    return {
        name: session.scalar(select(func.count()).select_from(model).where(model.doc_id == doc_id))
        or 0
        for name, model in models.items()
    }


def _document_item(session: Session, document: Document) -> dict[str, Any]:
    counts = _counts(session, document.doc_id)
    return {
        "doc_id": str(document.doc_id),
        "file_name": document.file_name,
        "file_type": document.file_type,
        "status": document.status,
        "version": document.version,
        "detected_document_type": document.detected_document_type,
        "document_type_confidence": document.document_type_confidence,
        **counts,
        "created_at": _iso(document.uploaded_at),
        "updated_at": _iso(document.uploaded_at),
        "source_path": document.source_path,
        "checksum": document.checksum,
        "metadata": document.metadata_json,
    }


def _run_item(run: IngestionRun, file_name: str | None = None) -> dict[str, Any]:
    quality = run.quality_report or {}
    timeline = [
        {
            "step_name": stage.get("stage", "unknown"),
            "status": stage.get("status", "unknown"),
            "latency_ms": stage.get("latency_ms", 0),
            "error_message": stage.get("error"),
        }
        for stage in quality.get("stage_traces", [])
    ]
    return {
        "run_id": str(run.run_id),
        "doc_id": str(run.doc_id),
        "file_name": file_name,
        "status": run.status,
        "started_at": _iso(run.started_at),
        "ended_at": _iso(run.ended_at),
        "total_latency_ms": sum(int(item["latency_ms"] or 0) for item in timeline),
        "parser_name": run.parser_name,
        "parser_version": run.parser_version,
        "total_chunks": run.total_chunks,
        "total_tables": run.total_tables,
        "total_table_rows": run.total_table_rows,
        "total_assets": run.total_assets,
        "total_embeddings": run.total_embeddings,
        "error_message": run.error_message,
        "quality_report": quality,
        "timeline": timeline,
    }


@router.get("/documents")
def documents(
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    items = session.scalars(
        select(Document).order_by(Document.uploaded_at.desc()).limit(limit)
    ).all()
    return {"items": [_document_item(session, item) for item in items]}


@router.get("/documents/{doc_id}")
def document_detail(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, Any]:
    document = session.get(Document, doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    payload = _document_item(session, document)
    chunks = session.scalars(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index).limit(100)
    ).all()
    tables = session.scalars(select(ParsedTable).where(ParsedTable.doc_id == doc_id)).all()
    assets = session.scalars(select(Asset).where(Asset.doc_id == doc_id)).all()
    latest_run = session.scalar(
        select(IngestionRun)
        .where(IngestionRun.doc_id == doc_id)
        .order_by(IngestionRun.started_at.desc())
        .limit(1)
    )
    payload.update(
        {
            "chunk_items": [
                {
                    "chunk_id": str(item.chunk_id),
                    "chunk_index": item.chunk_index,
                    "content": item.content,
                    "page_number": item.page_number,
                    "section_title": item.section_title,
                    "status": item.status,
                    "metadata": item.metadata_json,
                }
                for item in chunks
            ],
            "table_items": [
                {
                    "table_id": str(item.table_id),
                    "table_name": item.table_name,
                    "page_number": item.page_number,
                    "table_markdown": item.table_markdown,
                    "table_json": item.table_json,
                    "status": item.status,
                    "metadata": item.metadata_json,
                }
                for item in tables
            ],
            "asset_items": [_asset_item(item) for item in assets],
            "latest_ingestion_run": (
                _run_item(latest_run, document.file_name) if latest_run else None
            ),
        }
    )
    return payload


@router.post("/documents/{doc_id}/approve")
@router.post("/documents/{doc_id}/activate")
def activate_document(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, str]:
    if session.get(Document, doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        approve_document_records(session, doc_id)
    except ApprovalValidationError as exc:
        raise HTTPException(status_code=409, detail=exc.report.as_dict()) from exc
    return {"doc_id": str(doc_id), "status": "active"}


@router.post("/documents/{doc_id}/archive")
def archive_document(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, str]:
    if session.get(Document, doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    set_document_status(session, doc_id, "archived")
    return {"doc_id": str(doc_id), "status": "archived"}


@router.post("/documents/{doc_id}/reingest")
def reingest_document(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, Any]:
    document = session.get(Document, doc_id)
    if document is None or not document.source_path:
        raise HTTPException(status_code=404, detail="Document source is unavailable")
    path = Path(document.source_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document source file no longer exists")
    new_document, run = IngestionPipeline(get_settings()).ingest(
        session,
        path,
        IngestionOptions(
            duplicate_policy="replace",
            original_file_name=document.file_name,
        ),
    )
    return {
        "doc_id": str(new_document.doc_id),
        "run_id": str(run.run_id),
        "status": new_document.status,
    }


@router.get("/ingestion/runs")
def ingestion_runs(
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = session.execute(
        select(IngestionRun, Document.file_name)
        .join(Document, Document.doc_id == IngestionRun.doc_id)
        .order_by(IngestionRun.started_at.desc())
        .limit(limit)
    ).all()
    return {"items": [_run_item(run, file_name) for run, file_name in rows]}


@router.get("/ingestion/runs/{run_id}")
def ingestion_run_detail(run_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, Any]:
    row = session.execute(
        select(IngestionRun, Document.file_name)
        .join(Document, Document.doc_id == IngestionRun.doc_id)
        .where(IngestionRun.run_id == run_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    return _run_item(row[0], row[1])


@router.get("/ingestion/summary")
def ingestion_summary(session: Session = Depends(get_db)) -> dict[str, Any]:
    runs = session.scalars(select(IngestionRun)).all()
    reports = [run.quality_report or {} for run in runs]
    latencies = [
        sum(int(stage.get("latency_ms") or 0) for stage in report.get("stage_traces", []))
        for report in reports
    ]
    total_embeddings = sum(run.total_embeddings for run in runs)
    embedding_failed = sum(int(item.get("embedding_failed") or 0) for item in reports)
    return {
        "total_documents": session.scalar(select(func.count()).select_from(Document)) or 0,
        "parse_success_rate": (
            sum(run.status == "completed" for run in runs) / len(runs) if runs else 0
        ),
        "parse_failed_count": sum(run.status == "failed" for run in runs),
        "chunks_created": sum(run.total_chunks for run in runs),
        "empty_chunk_rate": (
            sum(int(item.get("empty_chunks") or 0) for item in reports)
            / max(1, sum(run.total_chunks for run in runs))
        ),
        "tables_detected": sum(run.total_tables for run in runs),
        "table_rows_created": sum(run.total_table_rows for run in runs),
        "assets_extracted": sum(run.total_assets for run in runs),
        "broken_assets": sum(
            bool(asset.local_path) and not Path(asset.local_path).is_file()
            for asset in session.scalars(select(Asset)).all()
        ),
        "embedding_success_rate": total_embeddings / max(1, total_embeddings + embedding_failed),
        "average_ingestion_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


class RetrievalDebugRequest(BaseModel):
    query: str
    use_structured: bool = True
    use_dense: bool = True
    use_sparse: bool = True
    use_rrf: bool = True
    use_reranker: bool = False
    use_hyde: bool = False


@router.post("/retrieval/debug")
async def retrieval_debug(
    request: RetrievalDebugRequest,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    structured_retriever = StructuredRetriever()
    product_names, service_names = structured_retriever.active_names(session)
    effective = settings.model_copy(
        update={
            "enable_hyde": request.use_hyde,
            "enable_reranker": request.use_reranker,
        }
    )
    routed = await IntentRouter().route_with_optional_llm(
        request.query,
        effective,
        OllamaClient(effective),
        known_products=product_names,
        known_services=service_names,
    )
    resolution = DatabaseEntityResolver(effective).resolve(
        session,
        request.query,
        routed.intent,
    )
    entities = resolution.names or routed.entities

    rewrite = await QueryRewriter().rewrite(
        request.query, routed.intent, effective, OllamaClient(effective)
    )
    structured = (
        structured_retriever.retrieve(session, routed.intent, request.query, entities)
        if request.use_structured
        else []
    )
    plan = RetrievalPlanner(effective).plan(
        query=request.query,
        routed=routed,
        entities=resolution,
        structured=structured,
    )
    embedder = EmbeddingService(effective)
    dense_sets: dict[str, list[RetrievalResult]] = {}
    if request.use_dense:
        dense_retriever = DenseRetriever(
            embedder,
            effective.dense_top_k,
            effective.dense_min_score,
        )
        dense_sets = {
            f"dense_original_{name}": values
            for name, values in dense_retriever.retrieve_by_source(
                session,
                request.query,
                routed.intent,
            ).items()
        }
        if rewrite.hyde_used and rewrite.hyde_query:
            dense_sets.update(
                {
                    f"dense_hyde_{name}": values
                    for name, values in dense_retriever.retrieve_by_source(
                        session,
                        rewrite.hyde_query,
                        routed.intent,
                    ).items()
                }
            )
    sparse_sets = (
        {
            f"sparse_{name}": values
            for name, values in SparseRetriever(
                effective.sparse_top_k,
                effective.sparse_trigram_threshold,
                min_fts_rank=effective.sparse_min_fts_rank,
                max_per_source=effective.sparse_max_per_source,
            )
            .retrieve_by_source(session, request.query, routed.intent)
            .items()
        }
        if request.use_sparse
        else {}
    )
    dense = [item for values in dense_sets.values() for item in values]
    sparse = [item for values in sparse_sets.values() for item in values]
    result_sets = {
        "structured": structured,
        **dense_sets,
        **sparse_sets,
    }
    weights = {
        name: (
            effective.structured_rrf_weight
            if name == "structured"
            else effective.sparse_rrf_weight
            if name.startswith("sparse_")
            else effective.dense_rrf_weight
        )
        for name in result_sets
    }
    fused = (
        reciprocal_rank_fusion(
            result_sets,
            effective.rrf_k,
            weights,
            max_per_source=effective.rrf_max_per_source,
        )
        if request.use_rrf
        else [*structured, *dense, *sparse]
    )
    reranked, reranker_used, rerank_meta = OptionalReranker(effective).rerank(request.query, fused)
    final_results = reranked if request.use_reranker else fused[: effective.final_top_k]
    context = ContextBuilder(
        effective.max_context_chars,
        effective.max_context_items_per_source,
    ).build(final_results)

    def items(values: list[RetrievalResult]) -> list[dict[str, Any]]:
        return [_retrieval_item(item, rank) for rank, item in enumerate(values, start=1)]

    return {
        "router": routed.as_dict(),
        "entities": resolution.as_dict(),
        "plan": plan.as_dict(),
        "rewrite": rewrite.as_dict(),
        "structured": items(structured),
        "dense": items(dense),
        "sparse": items(sparse),
        "rrf": items(fused),
        "reranker": items(reranked),
        "reranker_used": reranker_used,
        "reranker_meta": rerank_meta,
        "final_context": context,
    }


@router.get("/evaluation/datasets")
def evaluation_datasets(session: Session = Depends(get_db)) -> dict[str, Any]:
    items = session.scalars(
        select(EvaluationDataset).order_by(EvaluationDataset.created_at.desc())
    ).all()
    return {
        "items": [
            {
                "dataset_id": str(item.dataset_id),
                "name": item.name,
                "version": item.version,
                "description": item.description,
                "content_hash": item.content_hash,
                "metadata": item.metadata_json,
                "created_at": _iso(item.created_at),
            }
            for item in items
        ]
    }


@router.get("/evaluation/runs")
def evaluation_runs(
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = session.execute(
        select(EvaluationRun, EvaluationDataset.name)
        .outerjoin(EvaluationDataset, EvaluationDataset.dataset_id == EvaluationRun.dataset_id)
        .order_by(EvaluationRun.started_at.desc())
        .limit(limit)
    ).all()
    return {
        "items": [
            {
                "eval_run_id": str(run.eval_run_id),
                "dataset_id": str(run.dataset_id) if run.dataset_id else None,
                "dataset_name": name,
                "pipeline_version": run.pipeline_version,
                "data_version": run.data_version,
                "started_at": _iso(run.started_at),
                "ended_at": _iso(run.ended_at),
                "status": run.status,
                "metrics": run.metrics,
                "config_snapshot": run.config_snapshot,
            }
            for run, name in rows
        ]
    }


@router.get("/evaluation/cases")
def evaluation_cases(
    dataset_id: uuid.UUID | None = None,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    statement = select(EvaluationCase).order_by(EvaluationCase.case_id)
    if dataset_id:
        statement = statement.where(EvaluationCase.dataset_id == dataset_id)
    items = session.scalars(statement.limit(1000)).all()
    return {
        "items": [
            {
                "case_id": str(item.case_id),
                "dataset_id": str(item.dataset_id),
                "query": item.query,
                "case_key": item.case_key,
                "expected_intent": item.expected_intent,
                "expected_answer_type": item.expected_answer_type,
                "expected_doc_ids": [str(value) for value in item.expected_doc_ids or []],
                "expected_chunk_ids": [str(value) for value in item.expected_chunk_ids or []],
                "expected_row_ids": [str(value) for value in item.expected_row_ids or []],
                "expected_asset_ids": [str(value) for value in item.expected_asset_ids or []],
                "expected_entities": item.expected_entities or [],
                "expected_source_keys": item.expected_source_keys or [],
                "expected_answer_contains": item.expected_answer_contains or [],
                "forbidden_answer_contains": item.forbidden_answer_contains or [],
                "expected_answer": item.expected_answer,
                "metadata": item.metadata_json,
            }
            for item in items
        ]
    }


@router.get("/evaluation/results")
def evaluation_results(
    eval_run_id: uuid.UUID | None = None,
    limit: int = Query(default=1000, ge=1, le=5000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    if eval_run_id is None:
        latest = session.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.status == "completed")
            .order_by(EvaluationRun.started_at.desc())
        )
        eval_run_id = latest.eval_run_id if latest else None
    if eval_run_id is None:
        return {"items": []}
    items = session.scalars(
        select(EvaluationCaseResult)
        .where(EvaluationCaseResult.eval_run_id == eval_run_id)
        .order_by(EvaluationCaseResult.created_at, EvaluationCaseResult.result_id)
        .limit(limit)
    ).all()
    return {"items": [_evaluation_result_item(item) for item in items]}


@router.get("/evaluation/summary")
def evaluation_summary(session: Session = Depends(get_db)) -> dict[str, Any]:
    runs = session.scalars(
        select(EvaluationRun)
        .where(EvaluationRun.status == "completed")
        .order_by(EvaluationRun.started_at.desc())
        .limit(100)
    ).all()
    latest = next((run for run in runs if (run.metrics or {}).get("e2e")), None)
    latest = latest or (runs[0] if runs else None)
    metrics = latest.metrics if latest else {}
    router_metrics = metrics.get("router", {})
    e2e = metrics.get("e2e", {})
    retrieval = metrics.get("retrieval", {})
    generation = metrics.get("generation", {})
    assets = metrics.get("assets", {})
    return {
        "e2e_success_rate": e2e.get("success_rate"),
        "router_accuracy": router_metrics.get("accuracy"),
        "retrieval_recall_at_5": retrieval.get("recall_at_5"),
        "retrieval_mrr_at_10": retrieval.get("mrr_at_10"),
        "ndcg_at_10": retrieval.get("ndcg_at_10"),
        "retrieval_ground_truth_coverage": retrieval.get("ground_truth_coverage"),
        "json_validity_rate": generation.get("json_validity_rate"),
        "schema_pass_rate": generation.get("schema_pass_rate"),
        "answer_correctness": generation.get("answer_correctness"),
        "faithfulness_rate": generation.get("faithfulness_rate"),
        "unsupported_claim_rate": generation.get("unsupported_claim_rate"),
        "safety_pass_rate": generation.get("safety_pass_rate"),
        "asset_resolve_rate": assets.get("asset_resolve_success_rate"),
        "asset_ground_truth_coverage": assets.get("ground_truth_coverage"),
        "e2e_pass_rate": e2e.get("pass_rate"),
        "no_result_rate": e2e.get("no_result_rate"),
        "p50_latency_ms": e2e.get("p50_latency_ms"),
        "p95_latency_ms": e2e.get("p95_latency_ms"),
        "p99_latency_ms": e2e.get("p99_latency_ms"),
        "fallback_rate": e2e.get("fallback_rate"),
        "clarification_rate": e2e.get("clarification_rate"),
        "latest_run_id": str(latest.eval_run_id) if latest else None,
        "coverage": metrics.get("coverage", {}),
        "per_intent": metrics.get("per_intent", {}),
        "diagnostics": metrics.get("diagnostics", {}),
        "history": [
            {
                "eval_run_id": str(run.eval_run_id),
                "started_at": _iso(run.started_at),
                "router_accuracy": (run.metrics or {}).get("router", {}).get("accuracy"),
                "success_rate": (run.metrics or {}).get("e2e", {}).get("success_rate"),
                "pass_rate": (run.metrics or {}).get("e2e", {}).get("pass_rate"),
                "retrieval_recall_at_5": (run.metrics or {})
                .get("retrieval", {})
                .get("recall_at_5"),
                "faithfulness_rate": (run.metrics or {})
                .get("generation", {})
                .get("faithfulness_rate"),
                "p95_latency_ms": (run.metrics or {}).get("e2e", {}).get("p95_latency_ms"),
            }
            for run in reversed(runs)
            if (run.metrics or {}).get("e2e")
        ],
        "confusion_matrix": router_metrics.get("confusion_matrix", {}),
    }


def _ollama(settings: Settings) -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{settings.ollama_base_url.rstrip('/')}/api/tags",
            timeout=2,
        )
        response.raise_for_status()
        return {
            "status": "connected",
            "models": [item.get("name") for item in response.json().get("models", [])],
        }
    except Exception as exc:
        return {"status": "disconnected", "error": str(exc)}


@router.get("/observability/health")
def health(
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    database = check_database(session)
    return {
        "status": "ok" if database["connected"] else "degraded",
        "postgresql": {
            "status": "connected" if database["connected"] else "disconnected",
            "version": database.get("version"),
        },
        "pgvector": {"status": "enabled" if database["pgvector_enabled"] else "disabled"},
        "ollama": _ollama(settings),
        "embedding_model": {
            "status": "configured",
            "model": settings.embedding_model,
            "dimension": settings.embedding_dim,
        },
        "reranker": {
            "status": "enabled" if settings.enable_reranker else "disabled",
            "model": settings.reranker_model,
        },
        "assets": {
            "status": "available" if settings.asset_storage_dir.is_dir() else "missing",
            "directory": str(settings.asset_storage_dir),
            "public_base_url": settings.asset_public_base_url,
        },
    }


@router.get("/observability/metrics")
def metrics(session: Session = Depends(get_db)) -> dict[str, Any]:
    today = datetime.now(UTC).date()
    traces = session.scalars(select(RagTrace).where(func.date(RagTrace.created_at) == today)).all()
    total = len(traces)
    latencies = [float(item.total_latency_ms or 0) for item in traces]
    answers = [item.final_answer or {} for item in traces]
    return {
        "total_requests": total,
        "success_rate": sum(item.status == "success" for item in traces) / total if total else 0,
        "error_rate": sum(item.status == "failed" for item in traces) / total if total else 0,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "no_result_rate": sum(
            "chưa có đủ thông tin" in str(answer.get("answer", {}).get("text", "")).lower()
            for answer in answers
        )
        / total
        if total
        else 0,
        "fallback_rate": sum(
            answer.get("answer_type") == "fallback" for answer in answers
        )
        / total
        if total
        else 0,
        "clarification_rate": sum(item.detected_intent == "UNKNOWN" for item in traces) / total
        if total
        else 0,
    }


@router.get("/observability/diagnostics")
def observability_diagnostics(
    limit: int = Query(default=500, ge=1, le=5000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    steps = session.scalars(
        select(RagTraceStep).order_by(RagTraceStep.created_at.desc()).limit(limit)
    ).all()
    return build_diagnostics(
        case_results=[],
        trace_steps=[
            {
                "step_name": item.step_name,
                "latency_ms": item.latency_ms,
                "status": item.status,
            }
            for item in steps
        ],
        retrieval_coverage=1.0,
    )


@router.get("/observability/errors")
def errors(
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    items = session.scalars(
        select(RagTraceStep)
        .where(RagTraceStep.status == "failed")
        .order_by(RagTraceStep.created_at.desc())
        .limit(limit)
    ).all()
    return {
        "items": [
            {
                "time": _iso(item.created_at),
                "trace_id": str(item.trace_id),
                "step": item.step_name,
                "error_type": "PIPELINE_STEP_ERROR",
                "message": item.error_message,
            }
            for item in items
        ]
    }


@router.get("/traces")
def traces(
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    items = session.scalars(
        select(RagTrace).order_by(RagTrace.created_at.desc()).limit(limit)
    ).all()
    return {
        "items": [
            {
                "trace_id": str(item.trace_id),
                "session_id": item.session_id,
                "query": item.user_query,
                "intent": item.detected_intent,
                "confidence": item.confidence,
                "total_latency_ms": item.total_latency_ms,
                "status": item.status,
                "created_at": _iso(item.created_at),
                "final_answer": item.final_answer,
            }
            for item in items
        ]
    }


@router.get("/traces/{trace_id}")
def trace_detail(trace_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, Any]:
    trace = session.get(RagTrace, trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    steps = session.scalars(
        select(RagTraceStep)
        .where(RagTraceStep.trace_id == trace_id)
        .order_by(RagTraceStep.created_at)
    ).all()
    failed = next((item for item in steps if item.status == "failed"), None)
    return {
        "trace_id": str(trace.trace_id),
        "session_id": trace.session_id,
        "query": trace.user_query,
        "intent": trace.detected_intent,
        "confidence": trace.confidence,
        "total_latency_ms": trace.total_latency_ms,
        "status": trace.status,
        "created_at": _iso(trace.created_at),
        "final_answer": trace.final_answer,
        "error": failed.error_message if failed else None,
        "failed_step": failed.step_name if failed else None,
        "steps": [
            {
                "step_id": str(item.step_id),
                "step_name": item.step_name,
                "input": item.input_json,
                "output": item.output_json,
                "latency_ms": item.latency_ms,
                "status": item.status,
                "error_message": item.error_message,
                "created_at": _iso(item.created_at),
            }
            for item in steps
        ],
    }


@router.get("/assets")
def assets(
    limit: int = Query(default=500, ge=1, le=2000),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    items = session.scalars(select(Asset).order_by(Asset.asset_id.desc()).limit(limit)).all()
    return {"items": [_asset_item(item) for item in items]}


@router.get("/assets/{asset_id}")
def asset_detail(asset_id: uuid.UUID, session: Session = Depends(get_db)) -> dict[str, Any]:
    asset = session.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    payload = _asset_item(asset)
    payload["used_in"] = {
        "products": [
            {"product_id": str(item.product_id), "name": item.name}
            for item in session.scalars(select(Product).where(Product.asset_id == asset_id)).all()
        ],
        "services": [
            {"service_id": str(item.service_id), "name": item.name}
            for item in session.scalars(select(Service).where(Service.asset_id == asset_id)).all()
        ],
        "chunk_id": str(asset.chunk_id) if asset.chunk_id else None,
        "document_id": str(asset.doc_id),
    }
    return payload


def _product(item: Product) -> dict[str, Any]:
    return {
        "product_id": str(item.product_id),
        "name": item.name,
        "category": item.category,
        "category_code": item.category_code,
        "source_category": item.source_category,
        "brand": item.brand,
        "model": item.model,
        "description": item.description,
        "price": float(item.price) if item.price is not None else None,
        "currency": item.currency,
        "quantity": item.quantity,
        "link": item.link,
        "image_reference": item.image_reference,
        "asset_id": str(item.asset_id) if item.asset_id else None,
        "source_doc_id": str(item.source_doc_id),
        "source_row_id": str(item.source_row_id) if item.source_row_id else None,
        "status": item.status,
        "version": item.version,
    }


def _service(item: Service) -> dict[str, Any]:
    return {
        "service_id": str(item.service_id),
        "name": item.name,
        "category_code": item.category_code,
        "source_category": item.source_category,
        "description": item.description,
        "duration_minutes": item.duration_minutes,
        "price": float(item.price) if item.price is not None else None,
        "currency": item.currency,
        "symptoms": item.symptoms,
        "indications": item.indications,
        "contraindications": item.contraindications,
        "image_reference": item.image_reference,
        "asset_id": str(item.asset_id) if item.asset_id else None,
        "source_doc_id": str(item.source_doc_id),
        "source_row_id": str(item.source_row_id) if item.source_row_id else None,
        "status": item.status,
        "version": item.version,
    }


def _evaluation_result_item(item: EvaluationCaseResult) -> dict[str, Any]:
    return {
        "result_id": str(item.result_id),
        "eval_run_id": str(item.eval_run_id),
        "case_id": str(item.case_id) if item.case_id else None,
        "trace_id": str(item.trace_id) if item.trace_id else None,
        "query": item.query,
        "expected_intent": item.expected_intent,
        "actual_intent": item.actual_intent,
        "status": item.status,
        "passed": item.passed,
        "latency_ms": item.latency_ms,
        "expected_ids": item.expected_ids or [],
        "retrieved_ids": item.retrieved_ids or [],
        "answer_text": item.answer_text,
        "scores": item.scores or {},
        "violations": item.violations or [],
        "details": item.details or {},
        "error_message": item.error_message,
        "created_at": _iso(item.created_at),
    }


@router.get("/products")
def products(session: Session = Depends(get_db)) -> dict[str, Any]:
    return {
        "items": [
            _product(item) for item in session.scalars(select(Product).order_by(Product.name)).all()
        ]
    }


@router.get("/services")
def services(session: Session = Depends(get_db)) -> dict[str, Any]:
    return {
        "items": [
            _service(item) for item in session.scalars(select(Service).order_by(Service.name)).all()
        ]
    }


@router.get("/faqs")
def faqs(session: Session = Depends(get_db)) -> dict[str, Any]:
    items = session.scalars(select(FAQ).order_by(FAQ.question)).all()
    return {
        "items": [
            {
                "faq_id": str(item.faq_id),
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "category_code": item.category_code,
                "keywords": item.keywords or [],
                "is_active": item.is_active,
                "source_doc_id": str(item.source_doc_id) if item.source_doc_id else None,
                "source_row_id": str(item.source_row_id) if item.source_row_id else None,
                "embedding_status": "ready" if item.embedding is not None else "missing",
                "metadata": item.metadata_json,
            }
            for item in items
        ]
    }


@router.get("/clinic-info")
def clinic_info(session: Session = Depends(get_db)) -> dict[str, Any]:
    items = session.scalars(select(ClinicInfo).order_by(ClinicInfo.key)).all()
    return {
        "items": [
            {
                "id": str(item.id),
                "key": item.key,
                "value": item.value,
                "status": item.status,
                "source_doc_id": str(item.source_doc_id) if item.source_doc_id else None,
                "metadata": item.metadata_json,
            }
            for item in items
        ]
    }


@router.get("/tables")
def tables(session: Session = Depends(get_db)) -> dict[str, Any]:
    items = session.scalars(select(ParsedTable).limit(1000)).all()
    return {
        "items": [
            {
                "table_id": str(item.table_id),
                "doc_id": str(item.doc_id),
                "page_number": item.page_number,
                "table_name": item.table_name,
                "table_markdown": item.table_markdown,
                "table_json": item.table_json,
                "status": item.status,
                "metadata": item.metadata_json,
            }
            for item in items
        ]
    }


@router.get("/table-rows")
def table_rows(session: Session = Depends(get_db)) -> dict[str, Any]:
    items = session.scalars(
        select(TableRow).order_by(TableRow.table_id, TableRow.row_index).limit(2000)
    ).all()
    return {
        "items": [
            {
                "row_id": str(item.row_id),
                "table_id": str(item.table_id),
                "doc_id": str(item.doc_id),
                "row_index": item.row_index,
                "entity_type": item.entity_type,
                "entity_name": item.entity_name,
                "row_text": item.row_text,
                "row_json": item.row_json,
                "embedding_status": "ready" if item.embedding is not None else "missing",
                "status": item.status,
                "metadata": item.metadata_json,
            }
            for item in items
        ]
    }


@router.get("/chunks")
def chunks(session: Session = Depends(get_db)) -> dict[str, Any]:
    items = session.scalars(
        select(Chunk).order_by(Chunk.doc_id, Chunk.chunk_index).limit(2000)
    ).all()
    return {
        "items": [
            {
                "chunk_id": str(item.chunk_id),
                "doc_id": str(item.doc_id),
                "chunk_index": item.chunk_index,
                "content": item.content,
                "content_type": item.content_type,
                "page_number": item.page_number,
                "section_title": item.section_title,
                "embedding_status": "ready" if item.embedding is not None else "missing",
                "status": item.status,
                "metadata": item.metadata_json,
            }
            for item in items
        ]
    }


@router.get("/settings")
def settings_payload(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "api_environment": settings.app_env,
        "router_model": settings.ollama_router_model,
        "generation_model": settings.ollama_generation_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "top_k_dense": settings.dense_top_k,
        "top_k_sparse": settings.sparse_top_k,
        "top_k_final": settings.final_top_k,
        "rrf_k": settings.rrf_k,
        "reranker_enabled": settings.enable_reranker,
        "hyde_enabled": settings.enable_hyde,
        "confidence_threshold": settings.confidence_threshold,
        "assets_dir": str(settings.asset_storage_dir),
        "public_assets_base_url": settings.asset_public_base_url,
        "max_context_tokens": settings.max_context_chars,
        "json_retry_count": settings.json_retry_count,
        "read_only": True,
    }
