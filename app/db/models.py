import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str | None] = mapped_column(Text)
    source_path: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    checksum: Mapped[str | None] = mapped_column(Text, index=True)
    status: Mapped[str] = mapped_column(Text, default="draft", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    detected_document_type: Mapped[str | None] = mapped_column(Text)
    document_type_confidence: Mapped[float | None] = mapped_column(Float)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunks_doc_index"),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_tsv: Mapped[Any | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', simplydent_unaccent(coalesce(content, '')))",
            persisted=True,
        ),
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(get_settings().embedding_dim))
    content_type: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class Asset(Base):
    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_asset_token", "asset_token"),
        UniqueConstraint("doc_id", "stable_asset_key", name="uq_assets_doc_stable_key"),
    )

    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.chunk_id", ondelete="SET NULL")
    )
    asset_token: Mapped[str] = mapped_column(Text, nullable=False)
    stable_asset_key: Mapped[str] = mapped_column(Text, nullable=False)
    asset_type: Mapped[str | None] = mapped_column(Text)
    local_path: Mapped[str | None] = mapped_column(Text)
    public_url: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ChunkAsset(Base):
    __tablename__ = "chunk_assets"
    __table_args__ = (
        UniqueConstraint("chunk_id", "asset_id", name="uq_chunk_assets_chunk_asset"),
    )

    chunk_asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.chunk_id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.asset_id", ondelete="CASCADE"), index=True
    )
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ParsedTable(Base):
    __tablename__ = "tables"

    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    table_name: Mapped[str | None] = mapped_column(Text)
    table_markdown: Mapped[str | None] = mapped_column(Text)
    table_json: Mapped[list[dict[str, Any]] | dict[str, Any]] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class TableRow(Base):
    __tablename__ = "table_rows"
    __table_args__ = (
        Index("ix_table_rows_row_tsv", "row_tsv", postgresql_using="gin"),
        Index(
            "ix_table_rows_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        UniqueConstraint("table_id", "row_index", name="uq_table_rows_table_index"),
    )

    row_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tables.table_id", ondelete="CASCADE"), index=True
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    row_index: Mapped[int] = mapped_column(Integer)
    entity_type: Mapped[str | None] = mapped_column(Text, index=True)
    entity_name: Mapped[str | None] = mapped_column(Text, index=True)
    row_text: Mapped[str] = mapped_column(Text, nullable=False)
    row_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    row_tsv: Mapped[Any | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', simplydent_unaccent(coalesce(row_text, '')))",
            persisted=True,
        ),
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(get_settings().embedding_dim))
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ProductCategory(Base):
    __tablename__ = "product_categories"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    parent_code: Mapped[str | None] = mapped_column(
        Text, ForeignKey("product_categories.code", ondelete="SET NULL")
    )
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    status: Mapped[str] = mapped_column(Text, default="active")
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ServiceCategory(Base):
    __tablename__ = "service_categories"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    parent_code: Mapped[str | None] = mapped_column(
        Text, ForeignKey("service_categories.code", ondelete="SET NULL")
    )
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    status: Mapped[str] = mapped_column(Text, default="active")
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class FAQCategory(Base):
    __tablename__ = "faq_categories"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    parent_code: Mapped[str | None] = mapped_column(
        Text, ForeignKey("faq_categories.code", ondelete="SET NULL")
    )
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    status: Mapped[str] = mapped_column(Text, default="active")
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class CategoryAlias(Base):
    __tablename__ = "category_aliases"
    __table_args__ = (
        UniqueConstraint(
            "entity_type",
            "normalized_alias",
            name="uq_category_alias_entity_normalized",
        ),
    )

    alias_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    category_code: Mapped[str] = mapped_column(Text, nullable=False)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(Text, nullable=False)


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    category: Mapped[str | None] = mapped_column(Text)
    category_code: Mapped[str | None] = mapped_column(
        Text, ForeignKey("product_categories.code", ondelete="SET NULL")
    )
    source_category: Mapped[str | None] = mapped_column(Text)
    brand: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(Text, default="VND")
    quantity: Mapped[int | None] = mapped_column(Integer)
    link: Mapped[str | None] = mapped_column(Text)
    image_reference: Mapped[str | None] = mapped_column(Text)
    asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.asset_id", ondelete="SET NULL")
    )
    source_doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    source_row_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("table_rows.row_id", ondelete="SET NULL")
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ProductAlias(Base):
    __tablename__ = "product_aliases"
    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "normalized_alias",
            name="uq_product_alias_product_normalized",
        ),
    )

    alias_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("products.product_id", ondelete="CASCADE"), index=True
    )
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(Text, nullable=False, index=True)


class Service(Base):
    __tablename__ = "services"

    service_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    category_code: Mapped[str | None] = mapped_column(
        Text, ForeignKey("service_categories.code", ondelete="SET NULL")
    )
    source_category: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    duration_minutes: Mapped[int | None] = mapped_column(Integer)
    price: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(Text, default="VND")
    symptoms: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    indications: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    contraindications: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    image_reference: Mapped[str | None] = mapped_column(Text)
    asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.asset_id", ondelete="SET NULL")
    )
    source_doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    source_row_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("table_rows.row_id", ondelete="SET NULL")
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ClinicInfo(Base):
    __tablename__ = "clinic_info"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, default="active", index=True)
    source_doc_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="SET NULL")
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class FAQ(Base):
    __tablename__ = "faqs"
    __table_args__ = (
        Index("ix_faqs_question_tsv", "question_tsv", postgresql_using="gin"),
        Index(
            "ix_faqs_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    faq_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(Text)
    category_code: Mapped[str | None] = mapped_column(
        Text, ForeignKey("faq_categories.code", ondelete="SET NULL")
    )
    keywords: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    question_tsv: Mapped[Any | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', simplydent_unaccent("
            "coalesce(question, '') || ' ' || coalesce(answer, '')))",
            persisted=True,
        ),
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(get_settings().embedding_dim))
    source_doc_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    source_row_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("table_rows.row_id", ondelete="SET NULL")
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class FAQAlias(Base):
    __tablename__ = "faq_aliases"
    __table_args__ = (
        UniqueConstraint("faq_id", "normalized_variant", name="uq_faq_alias_faq_normalized"),
    )

    alias_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    faq_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("faqs.faq_id", ondelete="CASCADE"), index=True
    )
    question_variant: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_variant: Mapped[str] = mapped_column(Text, nullable=False, index=True)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, default="running", index=True)
    parser_name: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(Text)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    total_tables: Mapped[int] = mapped_column(Integer, default=0)
    total_table_rows: Mapped[int] = mapped_column(Integer, default=0)
    total_assets: Mapped[int] = mapped_column(Integer, default=0)
    total_embeddings: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    quality_report: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class RagTrace(Base):
    __tablename__ = "rag_traces"

    trace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str | None] = mapped_column(Text, index=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    detected_intent: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, default="running", index=True)
    final_answer: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    steps: Mapped[list["RagTraceStep"]] = relationship(
        back_populates="trace", cascade="all, delete-orphan", order_by="RagTraceStep.created_at"
    )


class RagTraceStep(Base):
    __tablename__ = "rag_trace_steps"

    step_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rag_traces.trace_id", ondelete="CASCADE"), index=True
    )
    step_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    input_json: Mapped[dict[str, Any]] = mapped_column("input", JSONB, default=dict)
    output_json: Mapped[dict[str, Any]] = mapped_column("output", JSONB, default=dict)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(Text, default="success")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    trace: Mapped[RagTrace] = relationship(back_populates="steps")


class EvaluationDataset(Base):
    __tablename__ = "evaluation_datasets"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationCase(Base):
    __tablename__ = "evaluation_cases"

    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_datasets.dataset_id", ondelete="CASCADE"),
        index=True,
    )
    case_key: Mapped[str | None] = mapped_column(Text, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_intent: Mapped[str | None] = mapped_column(Text)
    expected_answer_type: Mapped[str | None] = mapped_column(Text)
    expected_doc_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    expected_chunk_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    expected_row_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    expected_asset_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    expected_entities: Mapped[list[str]] = mapped_column(JSONB, default=list)
    expected_source_keys: Mapped[list[str]] = mapped_column(JSONB, default=list)
    expected_answer_contains: Mapped[list[str]] = mapped_column(JSONB, default=list)
    forbidden_answer_contains: Mapped[list[str]] = mapped_column(JSONB, default=list)
    expected_answer: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("evaluation_datasets.dataset_id", ondelete="SET NULL")
    )
    pipeline_version: Mapped[str] = mapped_column(Text, default="0.1.0")
    data_version: Mapped[str] = mapped_column(Text, default="unknown")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(Text, default="running", index=True)


class EvaluationCaseResult(Base):
    __tablename__ = "evaluation_case_results"
    __table_args__ = (
        UniqueConstraint("eval_run_id", "case_id", name="uq_eval_case_results_run_case"),
    )

    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.eval_run_id", ondelete="CASCADE"),
        index=True,
    )
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_cases.case_id", ondelete="SET NULL"),
        index=True,
    )
    trace_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rag_traces.trace_id", ondelete="SET NULL"),
        index=True,
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_intent: Mapped[str | None] = mapped_column(Text)
    actual_intent: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="pending", index=True)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    expected_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    retrieved_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    answer_text: Mapped[str | None] = mapped_column(Text)
    scores: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    violations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
