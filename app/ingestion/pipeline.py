import hashlib
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.resolver import detect_asset_tokens
from app.assets.storage import AssetStorage
from app.config import Settings
from app.db.models import (
    Asset,
    Chunk,
    ChunkAsset,
    Document,
    IngestionRun,
    ParsedTable,
    Product,
    Service,
)
from app.ingestion.asset_masker import IMAGE_PLACEHOLDER_PATTERN, mask_asset_positions
from app.ingestion.business_validation import validate_business_rows
from app.ingestion.chunker import DocumentChunker
from app.ingestion.docling_parser import (
    DocumentParser,
    ParsedAssetBlock,
    ParsedDocument,
    ParsedTextBlock,
)
from app.ingestion.document_classifier import classify_document
from app.ingestion.embedder import EmbeddingService
from app.ingestion.normalizer import normalize_document
from app.ingestion.quality_checks import build_quality_report
from app.ingestion.review import apply_document_status
from app.ingestion.smoke_checks import run_ingestion_smoke_checks
from app.ingestion.table_normalizer import NormalizedTable, normalize_table
from app.ingestion.table_processor import TableProcessor, serialize_row
from app.ner.entity_span_extractor import EntitySpanExtractor

DuplicatePolicy = Literal["reject", "reuse", "replace", "force"]


@dataclass(frozen=True)
class IngestionOptions:
    document_type: str = "auto"
    extract_tables: bool = True
    extract_assets: bool = True
    create_embeddings: bool = True
    require_review: bool = False
    duplicate_policy: DuplicatePolicy | None = None
    original_file_name: str | None = None
    asset_paths: tuple[Path, ...] = ()


class DuplicateDocumentError(ValueError):
    def __init__(self, document: Document):
        self.document = document
        super().__init__(
            f"Document with checksum already exists: {document.doc_id} ({document.file_name})"
        )


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.parser = DocumentParser()
        self.storage = AssetStorage(settings)
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingService(settings)
        self.table_processor = TableProcessor()
        self.entity_span_extractor = EntitySpanExtractor(settings)

    def ingest(
        self,
        session: Session,
        file_path: Path,
        options: IngestionOptions | None = None,
    ) -> tuple[Document, IngestionRun]:
        options = options or IngestionOptions()
        file_path = file_path.resolve()
        file_bytes = file_path.read_bytes()
        checksum = hashlib.sha256(file_bytes).hexdigest()
        policy = options.duplicate_policy or self.settings.duplicate_ingestion_policy
        if policy not in {"reject", "reuse", "replace", "force"}:
            raise ValueError(f"Unsupported duplicate ingestion policy: {policy}")

        duplicates = session.scalars(
            select(Document)
            .where(Document.checksum == checksum, Document.status != "archived")
            .order_by(Document.uploaded_at.desc())
        ).all()
        if duplicates and policy == "reject":
            raise DuplicateDocumentError(duplicates[0])
        if duplicates and policy == "reuse":
            latest_run = session.scalar(
                select(IngestionRun)
                .where(IngestionRun.doc_id == duplicates[0].doc_id)
                .order_by(IngestionRun.started_at.desc())
                .limit(1)
            )
            if latest_run is None:
                raise DuplicateDocumentError(duplicates[0])
            return duplicates[0], latest_run

        version = max((document.version for document in duplicates), default=0) + 1
        doc = Document(
            file_name=options.original_file_name or file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            source_path=str(file_path),
            checksum=checksum,
            status="draft",
            version=version,
        )
        session.add(doc)
        session.flush()
        run = IngestionRun(doc_id=doc.doc_id, status="running")
        session.add(run)
        session.commit()

        stage_traces: list[dict[str, Any]] = []
        try:
            with self._stage(stage_traces, "deduplication"):
                duplicate_ids = [str(item.doc_id) for item in duplicates]

            with self._stage(stage_traces, "parse"):
                parsed = self.parser.parse(file_path)
                for asset_path in options.asset_paths:
                    if not asset_path.is_file():
                        raise ValueError(f"Companion asset does not exist: {asset_path}")
                    parsed.assets.append(
                        ParsedAssetBlock(
                            data=asset_path.read_bytes(),
                            extension=asset_path.suffix.lower() or ".bin",
                            asset_type="companion_image",
                            metadata={
                                "source_ref": asset_path.name,
                                "original_file_name": asset_path.name,
                                "companion_upload": True,
                            },
                        )
                    )
                parsed = normalize_document(parsed)
                run.parser_name = parsed.parser_name
                run.parser_version = parsed.parser_version

            normalized_tables: list[tuple[Any, NormalizedTable]] = []
            table_classifications: list[dict[str, Any]] = []
            business_validation_reports: list[dict[str, Any]] = []
            review_reasons: list[str] = []
            if options.require_review:
                review_reasons.append("review_required_by_upload_option")

            with self._stage(stage_traces, "table_normalization_classification"):
                source_tables = parsed.tables if options.extract_tables else []
                for index, parsed_table in enumerate(source_tables, start=1):
                    inferred = normalize_table(
                        parsed_table.rows,
                        parsed_table.name,
                        "auto",
                    )
                    normalized = normalize_table(
                        parsed_table.rows,
                        parsed_table.name,
                        options.document_type,
                    )
                    classification = normalized.classification
                    summary = {
                        "table_index": index,
                        "table_name": parsed_table.name,
                        "page_number": parsed_table.page_number,
                        "entity_type": classification.entity_type,
                        "confidence": classification.confidence,
                        "requires_review": classification.requires_review,
                        "reasons": classification.reasons,
                        "column_mapping": classification.column_mapping,
                        "warnings": normalized.warnings,
                        "inferred_entity_type": inferred.classification.entity_type,
                        "inferred_confidence": inferred.classification.confidence,
                    }
                    table_classifications.append(summary)
                    normalized_tables.append((parsed_table, normalized))
                    if classification.entity_type is None:
                        review_reasons.append(f"table_{index}_schema_unknown")
                    elif (
                        classification.requires_review
                        or classification.confidence
                        < self.settings.table_classification_threshold
                    ):
                        review_reasons.append(f"table_{index}_classification_low_confidence")
                    validation = validate_business_rows(
                        session,
                        normalized.rows,
                        normalized.classification,
                        table_index=index,
                    )
                    business_validation_reports.append(
                        {
                            "table_index": index,
                            **validation.as_dict(),
                        }
                    )
                    review_reasons.extend(validation.blocking_reasons)

            document_classification = classify_document(
                requested_type=options.document_type,
                table_classifications=table_classifications,
                text="\n".join(block.text for block in parsed.text_blocks),
            )
            doc.detected_document_type = document_classification.document_type
            doc.document_type_confidence = document_classification.confidence
            if document_classification.requires_review:
                review_reasons.append("document_type_requires_review")

            asset_records: list[Asset] = []
            stored_assets = []
            asset_tokens_by_page: dict[int | None, list[str]] = defaultdict(list)
            with self._stage(stage_traces, "asset_storage"):
                source_assets = parsed.assets if options.extract_assets else []
                authored_tokens = self._collect_authored_asset_tokens(parsed)
                authored_index = 0
                for index, parsed_asset in enumerate(source_assets, start=1):
                    is_companion = bool(parsed_asset.metadata.get("companion_upload"))
                    token_override = None
                    if not is_companion and authored_index < len(authored_tokens):
                        token_override = authored_tokens[authored_index]
                        authored_index += 1
                    stored = self.storage.stage_bytes(
                        doc_id=doc.doc_id,
                        document_checksum=checksum,
                        data=parsed_asset.data,
                        extension=parsed_asset.extension,
                        index=index,
                        source_key=parsed_asset.metadata.get("source_ref"),
                        token_override=token_override,
                    )
                    asset = Asset(
                        asset_id=stored.asset_id,
                        doc_id=doc.doc_id,
                        asset_token=stored.token,
                        stable_asset_key=stored.stable_asset_key,
                        asset_type=parsed_asset.asset_type,
                        local_path=stored.local_path,
                        public_url=stored.public_url,
                        page_number=parsed_asset.page_number,
                        bbox=parsed_asset.bbox,
                        status="review_required",
                        metadata_json={
                            **parsed_asset.metadata,
                            "checksum": stored.checksum,
                        },
                    )
                    session.add(asset)
                    asset_records.append(asset)
                    stored_assets.append(stored)
                    if not is_companion:
                        asset_tokens_by_page[parsed_asset.page_number].append(stored.token)
                session.flush()

            with self._stage(stage_traces, "asset_masking"):
                self._mask_assets(parsed.text_blocks, asset_tokens_by_page)
                placeholder_count = sum(
                    len(IMAGE_PLACEHOLDER_PATTERN.findall(block.text))
                    for block in parsed.text_blocks
                )
                extraction_errors = list(parsed.metadata.get("asset_extraction_errors", []))
                if options.extract_assets and (placeholder_count or extraction_errors):
                    review_reasons.append("one_or_more_document_images_were_not_extracted")

            ingestion_entity_report: dict[str, Any] = {}
            with self._stage(stage_traces, "entity_extraction"):
                entity_rows = self._entity_extraction_rows(normalized_tables)
                ingestion_entities = self.entity_span_extractor.extract_for_ingestion(
                    text_blocks=[
                        {
                            "text": block.text,
                            "page_number": block.page_number,
                        }
                        for block in parsed.text_blocks
                    ],
                    table_rows=entity_rows,
                    known_products=self._known_ingestion_names(entity_rows, "product"),
                    known_services=self._known_ingestion_names(entity_rows, "service"),
                )
                ingestion_entity_report = ingestion_entities.as_dict()
                if ingestion_entities.degraded:
                    review_reasons.append("entity_extraction_degraded")

            table_rows_created = products_created = services_created = 0
            faqs_created = clinic_info_created = 0
            business_duplicates_skipped = 0
            seen_business_keys: set[tuple[str, str]] = set()
            embedding_success = embedding_failed = 0
            with self._stage(stage_traces, "table_processing"):
                for parsed_table, normalized in normalized_tables:
                    classification_metadata = normalized.classification.as_metadata()
                    table = ParsedTable(
                        doc_id=doc.doc_id,
                        page_number=parsed_table.page_number,
                        table_name=parsed_table.name,
                        table_markdown=parsed_table.markdown,
                        table_json=normalized.rows,
                        status="review_required",
                        metadata_json={
                            **parsed_table.metadata,
                            **classification_metadata,
                            "normalization_warnings": normalized.warnings,
                        },
                    )
                    session.add(table)
                    session.flush()
                    texts = [serialize_row(row) for row in normalized.rows]
                    embeddings: list[list[float] | None]
                    if options.create_embeddings and texts:
                        try:
                            embeddings = self.embedder.embed_documents(texts)
                            embedding_success += len(embeddings)
                        except Exception:
                            if self.settings.strict_embedding:
                                raise
                            embeddings = [None for _ in texts]
                            embedding_failed += len(texts)
                    else:
                        embeddings = [None for _ in texts]
                    counts = self.table_processor.process_rows(
                        session,
                        table_id=table.table_id,
                        doc_id=doc.doc_id,
                        rows=normalized.rows,
                        table_name=parsed_table.name,
                        status="review_required",
                        embeddings=embeddings,
                        classification=normalized.classification,
                        seen_business_keys=seen_business_keys,
                    )
                    table_rows_created += counts.rows
                    products_created += counts.products
                    services_created += counts.services
                    faqs_created += counts.faqs
                    clinic_info_created += counts.clinic_info
                    business_duplicates_skipped += counts.duplicates_skipped

            unresolved_image_references = [
                {
                    "entity_type": "product",
                    "entity_id": str(record.product_id),
                    "image_reference": record.image_reference,
                }
                for record in session.scalars(
                    select(Product).where(
                        Product.source_doc_id == doc.doc_id,
                        Product.image_reference.is_not(None),
                        Product.asset_id.is_(None),
                    )
                ).all()
            ]
            unresolved_image_references.extend(
                {
                    "entity_type": "service",
                    "entity_id": str(record.service_id),
                    "image_reference": record.image_reference,
                }
                for record in session.scalars(
                    select(Service).where(
                        Service.source_doc_id == doc.doc_id,
                        Service.image_reference.is_not(None),
                        Service.asset_id.is_(None),
                    )
                ).all()
            )
            if unresolved_image_references:
                review_reasons.append("one_or_more_image_references_were_not_resolved")

            with self._stage(stage_traces, "chunking_embedding"):
                text_chunks = self.chunker.split(parsed.text_blocks)
                texts = [chunk.content for chunk in text_chunks]
                if options.create_embeddings and texts:
                    try:
                        vectors: list[list[float] | None] = self.embedder.embed_documents(texts)
                        embedding_success += len(vectors)
                    except Exception:
                        if self.settings.strict_embedding:
                            raise
                        vectors = [None for _ in texts]
                        embedding_failed += len(texts)
                else:
                    vectors = [None for _ in texts]
                chunk_records: list[Chunk] = []
                for index, (text_chunk, vector) in enumerate(
                    zip(text_chunks, vectors, strict=True)
                ):
                    chunk = Chunk(
                        doc_id=doc.doc_id,
                        chunk_index=index,
                        content=text_chunk.content,
                        content_type=text_chunk.content_type,
                        page_number=text_chunk.page_number,
                        section_title=text_chunk.section_title,
                        embedding=vector,
                        status="review_required",
                        metadata_json=text_chunk.metadata or {},
                    )
                    session.add(chunk)
                    chunk_records.append(chunk)
                session.flush()
                for asset in asset_records:
                    matching_chunks = [
                        chunk for chunk in chunk_records if asset.asset_token in chunk.content
                    ]
                    asset.chunk_id = matching_chunks[0].chunk_id if matching_chunks else None
                    for chunk in matching_chunks:
                        session.add(
                            ChunkAsset(
                                chunk_id=chunk.chunk_id,
                                asset_id=asset.asset_id,
                                occurrence_count=chunk.content.count(asset.asset_token),
                                metadata_json={"asset_token": asset.asset_token},
                            )
                        )
                session.flush()

            embedding_backend = (
                self.embedder.backend_name
                if options.create_embeddings and (text_chunks or table_rows_created)
                else "disabled"
            )
            if options.create_embeddings and self.embedder.using_fallback:
                review_reasons.append("embedding_model_unavailable_hash_fallback_used")
            if embedding_failed:
                review_reasons.append("one_or_more_embeddings_failed")

            with self._stage(stage_traces, "ingestion_smoke_test"):
                smoke_report = run_ingestion_smoke_checks(
                    session,
                    doc.doc_id,
                    require_embeddings=options.create_embeddings,
                    staged_asset_paths={
                        stored.asset_id: stored.staged_path for stored in stored_assets
                    },
                    ignored_duplicate_doc_ids={item.doc_id for item in duplicates}
                    if policy == "replace"
                    else None,
                )
                review_reasons.extend(smoke_report.blocking_reasons)

            review_reasons = list(dict.fromkeys(review_reasons))
            final_status = "review_required"
            if self.settings.auto_approve_ingestion and not review_reasons:
                final_status = "active"
            with self._stage(stage_traces, "quality_checks"):
                report = build_quality_report(
                    chunks=[chunk.content for chunk in text_chunks],
                    tables_found=len(normalized_tables),
                    table_rows_created=table_rows_created,
                    assets_found=len(parsed.assets) if options.extract_assets else 0,
                    assets_resolved=len(asset_records),
                    products_created=products_created,
                    services_created=services_created,
                    embedding_success=embedding_success,
                    embedding_failed=embedding_failed,
                    embedding_backend=embedding_backend,
                    table_classifications=table_classifications,
                    classification_threshold=self.settings.table_classification_threshold,
                    review_reasons=review_reasons,
                    asset_extraction_errors=list(
                        parsed.metadata.get("asset_extraction_errors", [])
                    ),
                    stage_traces=stage_traces,
                )
                report["faqs_created"] = faqs_created
                report["clinic_info_created"] = clinic_info_created
                report["document_classification"] = document_classification.as_dict()
                report["business_validation"] = business_validation_reports
                report["ingestion_entity_extraction"] = ingestion_entity_report
                report["unresolved_image_references"] = unresolved_image_references
                report["smoke_test"] = smoke_report.as_dict()

            activation_report = apply_document_status(session, doc.doc_id, final_status)
            if policy == "replace" and final_status == "active":
                for duplicate in duplicates:
                    apply_document_status(session, duplicate.doc_id, "archived")
            report["business_dedup"] = {
                "intra_document_duplicates_skipped": business_duplicates_skipped,
                **activation_report,
            }

            doc.status = final_status
            doc.metadata_json = {
                **parsed.metadata,
                "ingestion_options": {
                    "document_type": options.document_type,
                    "extract_tables": options.extract_tables,
                    "extract_assets": options.extract_assets,
                    "create_embeddings": options.create_embeddings,
                    "require_review": options.require_review,
                    "duplicate_policy": policy,
                },
                "duplicate_document_ids": duplicate_ids,
                "table_classifications": table_classifications,
                "document_classification": document_classification.as_dict(),
                "business_validation": business_validation_reports,
                "ingestion_entity_extraction": ingestion_entity_report,
                "companion_asset_paths": [str(path) for path in options.asset_paths],
                "unresolved_image_references": unresolved_image_references,
                "review_reasons": review_reasons,
                "embedding_backend": embedding_backend,
                "smoke_test": smoke_report.as_dict(),
                "business_dedup": report["business_dedup"],
            }
            run.status = "completed"
            run.ended_at = datetime.now(UTC)
            run.total_chunks = len(text_chunks)
            run.total_tables = len(normalized_tables)
            run.total_table_rows = table_rows_created
            run.total_assets = len(asset_records)
            run.total_embeddings = embedding_success
            run.quality_report = report
            session.add_all([doc, run])
            self.storage.promote(doc.doc_id)
            session.commit()
            session.refresh(doc)
            session.refresh(run)
            return doc, run
        except Exception as exc:
            session.rollback()
            self.storage.rollback(doc.doc_id)
            failed_doc = session.get(Document, doc.doc_id)
            failed_run = session.get(IngestionRun, run.run_id)
            if failed_doc:
                failed_doc.status = "failed"
            if failed_run:
                failed_run.status = "failed"
                failed_run.ended_at = datetime.now(UTC)
                failed_run.error_message = str(exc)
                failed_run.quality_report = {"stage_traces": stage_traces}
            session.commit()
            raise

    @staticmethod
    def _collect_authored_asset_tokens(parsed: ParsedDocument) -> list[str]:
        # Table tokens are the strongest source for linking normalized business records.
        candidates = [
            token
            for table in parsed.tables
            for row in table.rows
            for value in row.values()
            for token in detect_asset_tokens(str(value))
        ]
        candidates.extend(
            token
            for block in parsed.text_blocks
            for token in detect_asset_tokens(block.text)
        )
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _mask_assets(
        text_blocks: list[ParsedTextBlock],
        tokens_by_page: dict[int | None, list[str]],
    ) -> None:
        remaining = {page: list(tokens) for page, tokens in tokens_by_page.items()}
        last_block_by_page: dict[int | None, ParsedTextBlock] = {}
        for block in text_blocks:
            last_block_by_page[block.page_number] = block
            page_tokens = remaining.get(block.page_number, [])
            placeholder_count = len(IMAGE_PLACEHOLDER_PATTERN.findall(block.text))
            if page_tokens and placeholder_count:
                selected = page_tokens[:placeholder_count]
                block.text = mask_asset_positions(block.text, selected)
                remaining[block.page_number] = page_tokens[len(selected) :]
        for page_number, page_tokens in list(remaining.items()):
            if not page_tokens:
                continue
            target = last_block_by_page.get(page_number)
            if target is not None:
                target.text = mask_asset_positions(target.text, page_tokens)
                remaining[page_number] = []
        unplaced = [token for tokens in remaining.values() for token in tokens if token]
        if not unplaced:
            return
        if text_blocks:
            text_blocks[0].text = mask_asset_positions(text_blocks[0].text, unplaced)
        else:
            text_blocks.append(ParsedTextBlock("\n".join(unplaced)))

    @staticmethod
    def _entity_extraction_rows(
        normalized_tables: list[tuple[Any, NormalizedTable]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _parsed_table, normalized in normalized_tables:
            entity_type = normalized.classification.entity_type
            for row in normalized.rows:
                rows.append({**row, "_entity_type": entity_type})
        return rows

    @staticmethod
    def _known_ingestion_names(
        rows: list[dict[str, Any]],
        entity_type: str,
    ) -> list[str]:
        names: list[str] = []
        for row in rows:
            if row.get("_entity_type") != entity_type:
                continue
            value = row.get("name")
            if value:
                names.append(str(value))
        return list(dict.fromkeys(names))

    @staticmethod
    @contextmanager
    def _stage(stage_traces: list[dict[str, Any]], name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            stage_traces.append(
                {
                    "stage": name,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise
        else:
            stage_traces.append(
                {
                    "stage": name,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "status": "success",
                }
            )
