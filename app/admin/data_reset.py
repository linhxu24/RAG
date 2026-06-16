import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.assets.storage import AssetStorage
from app.config import Settings
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
    Service,
    TableRow,
)

ResetScope = Literal["content", "runtime"]


def data_counts(session: Session) -> dict[str, int]:
    models = {
        "documents": Document,
        "chunks": Chunk,
        "tables": ParsedTable,
        "table_rows": TableRow,
        "assets": Asset,
        "products": Product,
        "services": Service,
        "faqs": FAQ,
        "clinic_info": ClinicInfo,
        "ingestion_runs": IngestionRun,
        "rag_traces": RagTrace,
        "evaluation_datasets": EvaluationDataset,
        "evaluation_cases": EvaluationCase,
        "evaluation_runs": EvaluationRun,
        "evaluation_case_results": EvaluationCaseResult,
    }
    return {
        name: int(session.scalar(select(func.count()).select_from(model)) or 0)
        for name, model in models.items()
    }


def delete_document(
    session: Session,
    settings: Settings,
    doc_id: uuid.UUID,
) -> dict[str, Any]:
    document = session.get(Document, doc_id)
    if document is None:
        raise LookupError("Document not found")
    before = _document_counts(session, doc_id)
    source_paths = _document_source_paths(document)
    session.execute(delete(ClinicInfo).where(ClinicInfo.source_doc_id == doc_id))
    session.delete(document)
    session.commit()
    AssetStorage(settings).rollback(doc_id)
    _remove_upload_paths(settings, source_paths)
    return {"doc_id": str(doc_id), "deleted": before}


def reset_data(
    session: Session,
    settings: Settings,
    scope: ResetScope,
) -> dict[str, Any]:
    before = data_counts(session)
    documents = session.scalars(select(Document)).all()
    source_paths = [
        path
        for document in documents
        for path in _document_source_paths(document)
    ]

    session.execute(delete(FAQ))
    session.execute(delete(ClinicInfo))
    session.execute(delete(Product))
    session.execute(delete(Service))
    session.execute(delete(Document))
    if scope == "runtime":
        session.execute(delete(EvaluationCaseResult))
        session.execute(delete(EvaluationRun))
        session.execute(delete(EvaluationCase))
        session.execute(delete(EvaluationDataset))
        session.execute(delete(RagTrace))
    session.commit()

    storage = AssetStorage(settings)
    removed_asset_files = storage.purge_all_content()
    _remove_upload_paths(settings, source_paths)
    return {
        "scope": scope,
        "deleted": before,
        "remaining": data_counts(session),
        "taxonomy_preserved": True,
        "removed_asset_files": removed_asset_files,
    }


def _document_counts(session: Session, doc_id: uuid.UUID) -> dict[str, int]:
    models = {
        "chunks": (Chunk, Chunk.doc_id),
        "tables": (ParsedTable, ParsedTable.doc_id),
        "table_rows": (TableRow, TableRow.doc_id),
        "assets": (Asset, Asset.doc_id),
        "products": (Product, Product.source_doc_id),
        "services": (Service, Service.source_doc_id),
        "faqs": (FAQ, FAQ.source_doc_id),
        "clinic_info": (ClinicInfo, ClinicInfo.source_doc_id),
        "ingestion_runs": (IngestionRun, IngestionRun.doc_id),
    }
    return {
        name: int(
            session.scalar(
                select(func.count()).select_from(model).where(column == doc_id)
            )
            or 0
        )
        for name, (model, column) in models.items()
    }


def _document_source_paths(document: Document) -> list[Path]:
    paths = [Path(document.source_path)] if document.source_path else []
    paths.extend(
        Path(value)
        for value in (document.metadata_json or {}).get("companion_asset_paths", [])
        if value
    )
    return paths


def _remove_upload_paths(settings: Settings, paths: list[Path]) -> None:
    upload_root = settings.upload_dir.resolve()
    directories: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(upload_root):
            continue
        resolved.unlink(missing_ok=True)
        directories.add(resolved.parent)
    for directory in sorted(directories, reverse=True):
        if directory == upload_root or not directory.is_relative_to(upload_root):
            continue
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    staging = upload_root / ".companion"
    if staging.exists() and not any(staging.iterdir()):
        shutil.rmtree(staging, ignore_errors=True)
