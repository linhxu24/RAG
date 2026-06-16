import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Document, IngestionRun, ParsedTable, TableRow
from app.db.session import get_db
from app.ingestion.business_dedup import business_key
from app.ingestion.embedder import EmbeddingService
from app.ingestion.pipeline import (
    DuplicateDocumentError,
    IngestionOptions,
    IngestionPipeline,
)
from app.ingestion.review import (
    ApprovalValidationError,
    approve_document_records,
    set_document_status,
)
from app.ingestion.table_normalizer import canonicalize_row, normalize_table
from app.ingestion.table_processor import TableProcessor, serialize_row

router = APIRouter(tags=["ingestion"])


class IngestPathRequest(BaseModel):
    source_path: str
    document_type: str = "auto"
    extract_tables: bool = True
    extract_assets: bool = True
    create_embeddings: bool = True
    require_review: bool = False
    duplicate_policy: str | None = None


class TableClassificationRequest(BaseModel):
    entity_type: Literal["product", "service", "faq", "clinic_info"]
    column_mapping: dict[str, str] = Field(default_factory=dict)


def _run_ingestion(
    session: Session,
    path: Path,
    options: IngestionOptions | None = None,
) -> dict:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        document, run = IngestionPipeline(get_settings()).ingest(session, path, options)
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Document content already exists. "
                f"existing_doc_id={exc.document.doc_id}; "
                "use duplicate_policy=replace or force explicitly."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {exc}") from exc
    return {
        "doc_id": str(document.doc_id),
        "run_id": str(run.run_id),
        "document_status": document.status,
        "run_status": run.status,
        "detected_document_type": document.detected_document_type,
        "document_type_confidence": document.document_type_confidence,
        "quality_report": run.quality_report,
    }


@router.post("/ingest/upload")
def upload_and_ingest(
    file: UploadFile = File(...),
    asset_files: list[UploadFile] = File(default=[]),
    document_type: str = Form("auto"),
    extract_tables: bool = Form(True),
    extract_assets: bool = Form(True),
    create_embeddings: bool = Form(True),
    require_review: bool = Form(False),
    duplicate_policy: str | None = Form(None),
    session: Session = Depends(get_db),
) -> dict:
    settings = get_settings()
    settings.ensure_directories()
    suffix = Path(file.filename or "upload.bin").suffix
    destination = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    companion_dir = settings.upload_dir / f"{destination.stem}_assets"
    companion_paths: list[Path] = []
    if asset_files:
        companion_dir.mkdir(parents=True, exist_ok=True)
        for asset_file in asset_files:
            safe_name = Path(asset_file.filename or f"{uuid.uuid4().hex}.bin").name
            asset_destination = companion_dir / safe_name
            with asset_destination.open("wb") as output:
                shutil.copyfileobj(asset_file.file, output)
            companion_paths.append(asset_destination)
    options = IngestionOptions(
        document_type=document_type,
        extract_tables=extract_tables,
        extract_assets=extract_assets,
        create_embeddings=create_embeddings,
        require_review=require_review,
        duplicate_policy=duplicate_policy,
        original_file_name=file.filename,
        asset_paths=tuple(companion_paths),
    )
    try:
        return _run_ingestion(session, destination, options)
    except HTTPException as exc:
        if exc.status_code == 409:
            destination.unlink(missing_ok=True)
            shutil.rmtree(companion_dir, ignore_errors=True)
        raise


@router.post("/ingest/run")
def ingest_existing_path(request: IngestPathRequest, session: Session = Depends(get_db)) -> dict:
    return _run_ingestion(
        session,
        Path(request.source_path),
        IngestionOptions(
            document_type=request.document_type,
            extract_tables=request.extract_tables,
            extract_assets=request.extract_assets,
            create_embeddings=request.create_embeddings,
            require_review=request.require_review,
            duplicate_policy=request.duplicate_policy,
        ),
    )


@router.get("/ingest/runs/{run_id}")
def get_ingestion_run(run_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    run = session.get(IngestionRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Ingestion run not found")
    return {
        "run_id": str(run.run_id),
        "doc_id": str(run.doc_id),
        "status": run.status,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "parser_name": run.parser_name,
        "parser_version": run.parser_version,
        "totals": {
            "chunks": run.total_chunks,
            "tables": run.total_tables,
            "table_rows": run.total_table_rows,
            "assets": run.total_assets,
            "embeddings": run.total_embeddings,
        },
        "quality_report": run.quality_report,
        "error_message": run.error_message,
    }


@router.get("/documents/{doc_id}")
def get_document(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    document = session.get(Document, doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    runs = session.scalars(
        select(IngestionRun)
        .where(IngestionRun.doc_id == doc_id)
        .order_by(IngestionRun.started_at.desc())
    ).all()
    return {
        "doc_id": str(document.doc_id),
        "file_name": document.file_name,
        "file_type": document.file_type,
        "source_path": document.source_path,
        "uploaded_at": document.uploaded_at,
        "checksum": document.checksum,
        "status": document.status,
        "version": document.version,
        "detected_document_type": document.detected_document_type,
        "document_type_confidence": document.document_type_confidence,
        "metadata": document.metadata_json,
        "run_ids": [str(run.run_id) for run in runs],
    }


@router.post("/documents/{doc_id}/approve")
def approve_document(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    if session.get(Document, doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        report = approve_document_records(session, doc_id)
    except ApprovalValidationError as exc:
        raise HTTPException(status_code=409, detail=exc.report.as_dict()) from exc
    return {
        "doc_id": str(doc_id),
        "status": "active",
        "approval_validation": report.as_dict(),
    }


@router.post("/documents/{doc_id}/archive")
def archive_document(doc_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    if session.get(Document, doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    set_document_status(session, doc_id, "archived")
    return {"doc_id": str(doc_id), "status": "archived"}


@router.post("/documents/{doc_id}/tables/{table_id}/classify")
@router.post("/api/documents/{doc_id}/tables/{table_id}/classify")
def classify_table_for_review(
    doc_id: uuid.UUID,
    table_id: uuid.UUID,
    request: TableClassificationRequest,
    session: Session = Depends(get_db),
) -> dict:
    table = session.get(ParsedTable, table_id)
    if table is None or table.doc_id != doc_id:
        raise HTTPException(status_code=404, detail="Table not found for document")
    table_rows = session.scalars(
        select(TableRow)
        .where(TableRow.table_id == table_id)
        .order_by(TableRow.row_index)
    ).all()
    if any(row.entity_type for row in table_rows):
        raise HTTPException(
            status_code=409,
            detail="Table is already classified; archive/reingest to replace its business records.",
        )

    mapped_rows = [
        {
            request.column_mapping.get(str(key), str(key)): value
            for key, value in row.row_json.items()
        }
        for row in table_rows
    ]
    normalized = normalize_table(mapped_rows, table.table_name, request.entity_type)
    if len(normalized.rows) != len(table_rows):
        raise HTTPException(status_code=422, detail="Column mapping removed one or more table rows")

    embedder = EmbeddingService(get_settings())
    row_texts = [serialize_row(row) for row in normalized.rows]
    embeddings = embedder.embed_documents(row_texts) if row_texts else []
    processor = TableProcessor()
    seen_business_keys: set[tuple[str, str]] = set()
    rows_synced = 0
    duplicates_skipped = 0
    for table_row, row_json, embedding in zip(
        table_rows,
        normalized.rows,
        embeddings,
        strict=True,
    ):
        canonical = canonicalize_row(row_json)
        table_row.row_json = row_json
        table_row.row_text = serialize_row(row_json)
        table_row.entity_type = request.entity_type
        table_row.entity_name = (
            str(canonical.get("service_name") or canonical.get("name") or "").strip()
            or None
        )
        table_row.embedding = embedding
        table_row.metadata_json = {
            **(table_row.metadata_json or {}),
            "classification": normalized.classification.as_metadata(),
            "manually_classified": True,
        }
        dedup_key = business_key(request.entity_type, canonical)
        if dedup_key and dedup_key in seen_business_keys:
            table_row.metadata_json = {
                **(table_row.metadata_json or {}),
                "business_dedup": {
                    "key": f"{dedup_key[0]}:{dedup_key[1]}",
                    "action": "business_sync_skipped",
                },
            }
            duplicates_skipped += 1
        else:
            if dedup_key:
                seen_business_keys.add(dedup_key)
            if processor.sync_business_record(
                session,
                table_row=table_row,
                entity_type=request.entity_type,
                status="review_required",
                canonical=canonical,
                embedding=embedding,
            ):
                rows_synced += 1

    table.table_json = normalized.rows
    table.metadata_json = {
        **(table.metadata_json or {}),
        **normalized.classification.as_metadata(),
        "manually_classified": True,
    }
    session.commit()
    return {
        "doc_id": str(doc_id),
        "table_id": str(table_id),
        "entity_type": request.entity_type,
        "rows_synced": rows_synced,
        "business_duplicates_skipped": duplicates_skipped,
        "status": "review_required",
    }
