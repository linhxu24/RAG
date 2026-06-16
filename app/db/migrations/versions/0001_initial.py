"""Initial PostgreSQL, pgvector, tracing, and evaluation schema."""

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    schema_sql = (
        """
        CREATE TABLE documents (
          doc_id UUID PRIMARY KEY,
          file_name TEXT NOT NULL,
          file_type TEXT,
          source_path TEXT,
          uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          checksum TEXT,
          status TEXT NOT NULL DEFAULT 'draft',
          version INTEGER NOT NULL DEFAULT 1,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_documents_status ON documents(status);

        CREATE TABLE chunks (
          chunk_id UUID PRIMARY KEY,
          doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          chunk_index INTEGER NOT NULL,
          content TEXT NOT NULL,
          content_tsv TSVECTOR GENERATED ALWAYS AS
            (to_tsvector('simple', coalesce(content, ''))) STORED,
          embedding VECTOR(%(embedding_dim)s),
          content_type TEXT,
          page_number INTEGER,
          section_title TEXT,
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
          CONSTRAINT uq_chunks_doc_index UNIQUE(doc_id, chunk_index)
        );
        CREATE INDEX ix_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX ix_chunks_status ON chunks(status);
        CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN(content_tsv);
        CREATE INDEX ix_chunks_embedding_hnsw ON chunks
          USING hnsw (embedding vector_cosine_ops);

        CREATE TABLE assets (
          asset_id UUID PRIMARY KEY,
          doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
          asset_token TEXT UNIQUE NOT NULL,
          asset_type TEXT,
          local_path TEXT,
          public_url TEXT,
          page_number INTEGER,
          bbox JSONB,
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_assets_doc_id ON assets(doc_id);
        CREATE INDEX ix_assets_status ON assets(status);

        CREATE TABLE tables (
          table_id UUID PRIMARY KEY,
          doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          page_number INTEGER,
          table_name TEXT,
          table_markdown TEXT,
          table_json JSONB NOT NULL DEFAULT '[]'::jsonb,
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_tables_doc_id ON tables(doc_id);
        CREATE INDEX ix_tables_status ON tables(status);

        CREATE TABLE table_rows (
          row_id UUID PRIMARY KEY,
          table_id UUID NOT NULL REFERENCES tables(table_id) ON DELETE CASCADE,
          doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          row_index INTEGER NOT NULL,
          entity_type TEXT,
          entity_name TEXT,
          row_text TEXT NOT NULL,
          row_json JSONB NOT NULL DEFAULT '{}'::jsonb,
          row_tsv TSVECTOR GENERATED ALWAYS AS
            (to_tsvector('simple', coalesce(row_text, ''))) STORED,
          embedding VECTOR(%(embedding_dim)s),
          status TEXT NOT NULL DEFAULT 'active',
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
          CONSTRAINT uq_table_rows_table_index UNIQUE(table_id, row_index)
        );
        CREATE INDEX ix_table_rows_table_id ON table_rows(table_id);
        CREATE INDEX ix_table_rows_doc_id ON table_rows(doc_id);
        CREATE INDEX ix_table_rows_entity_type ON table_rows(entity_type);
        CREATE INDEX ix_table_rows_entity_name ON table_rows(entity_name);
        CREATE INDEX ix_table_rows_status ON table_rows(status);
        CREATE INDEX ix_table_rows_row_tsv ON table_rows USING GIN(row_tsv);
        CREATE INDEX ix_table_rows_embedding_hnsw ON table_rows
          USING hnsw (embedding vector_cosine_ops);

        CREATE TABLE products (
          product_id UUID PRIMARY KEY,
          name TEXT NOT NULL,
          category TEXT,
          description TEXT,
          price NUMERIC(14,2),
          quantity INTEGER,
          link TEXT,
          asset_id UUID REFERENCES assets(asset_id) ON DELETE SET NULL,
          source_doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          source_row_id UUID REFERENCES table_rows(row_id) ON DELETE SET NULL,
          version INTEGER NOT NULL DEFAULT 1,
          status TEXT NOT NULL DEFAULT 'active',
          valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
          valid_to TIMESTAMPTZ,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_products_name ON products(name);
        CREATE INDEX ix_products_name_trgm ON products USING GIN(name gin_trgm_ops);
        CREATE INDEX ix_products_status ON products(status);
        CREATE INDEX ix_products_source_doc_id ON products(source_doc_id);

        CREATE TABLE services (
          service_id UUID PRIMARY KEY,
          name TEXT NOT NULL,
          description TEXT,
          duration_minutes INTEGER,
          price NUMERIC(14,2),
          symptoms TEXT[],
          asset_id UUID REFERENCES assets(asset_id) ON DELETE SET NULL,
          source_doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          source_row_id UUID REFERENCES table_rows(row_id) ON DELETE SET NULL,
          version INTEGER NOT NULL DEFAULT 1,
          status TEXT NOT NULL DEFAULT 'active',
          valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
          valid_to TIMESTAMPTZ,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_services_name ON services(name);
        CREATE INDEX ix_services_name_trgm ON services USING GIN(name gin_trgm_ops);
        CREATE INDEX ix_services_status ON services(status);
        CREATE INDEX ix_services_source_doc_id ON services(source_doc_id);

        CREATE TABLE clinic_info (
          id UUID PRIMARY KEY,
          key TEXT NOT NULL,
          value TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active',
          source_doc_id UUID REFERENCES documents(doc_id) ON DELETE SET NULL,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_clinic_info_key ON clinic_info(key);
        CREATE INDEX ix_clinic_info_status ON clinic_info(status);

        CREATE TABLE faqs (
          faq_id UUID PRIMARY KEY,
          question TEXT NOT NULL,
          answer TEXT NOT NULL,
          category TEXT,
          is_active BOOLEAN NOT NULL DEFAULT true,
          question_tsv TSVECTOR GENERATED ALWAYS AS
            (to_tsvector('simple', coalesce(question, ''))) STORED,
          embedding VECTOR(%(embedding_dim)s),
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_faqs_is_active ON faqs(is_active);
        CREATE INDEX ix_faqs_question_tsv ON faqs USING GIN(question_tsv);
        CREATE INDEX ix_faqs_embedding_hnsw ON faqs USING hnsw (embedding vector_cosine_ops);

        CREATE TABLE ingestion_runs (
          run_id UUID PRIMARY KEY,
          doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          ended_at TIMESTAMPTZ,
          status TEXT NOT NULL DEFAULT 'running',
          parser_name TEXT,
          parser_version TEXT,
          total_chunks INTEGER NOT NULL DEFAULT 0,
          total_tables INTEGER NOT NULL DEFAULT 0,
          total_table_rows INTEGER NOT NULL DEFAULT 0,
          total_assets INTEGER NOT NULL DEFAULT 0,
          total_embeddings INTEGER NOT NULL DEFAULT 0,
          error_message TEXT,
          quality_report JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_ingestion_runs_doc_id ON ingestion_runs(doc_id);
        CREATE INDEX ix_ingestion_runs_status ON ingestion_runs(status);

        CREATE TABLE rag_traces (
          trace_id UUID PRIMARY KEY,
          session_id TEXT,
          user_query TEXT NOT NULL,
          detected_intent TEXT,
          confidence DOUBLE PRECISION,
          total_latency_ms INTEGER,
          status TEXT NOT NULL DEFAULT 'running',
          final_answer JSONB,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_rag_traces_session_id ON rag_traces(session_id);
        CREATE INDEX ix_rag_traces_status ON rag_traces(status);
        CREATE INDEX ix_rag_traces_created_at ON rag_traces(created_at);

        CREATE TABLE rag_trace_steps (
          step_id UUID PRIMARY KEY,
          trace_id UUID NOT NULL REFERENCES rag_traces(trace_id) ON DELETE CASCADE,
          step_name TEXT NOT NULL,
          input JSONB NOT NULL DEFAULT '{}'::jsonb,
          output JSONB NOT NULL DEFAULT '{}'::jsonb,
          latency_ms INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'success',
          error_message TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_rag_trace_steps_trace_id ON rag_trace_steps(trace_id);
        CREATE INDEX ix_rag_trace_steps_step_name ON rag_trace_steps(step_name);

        CREATE TABLE evaluation_datasets (
          dataset_id UUID PRIMARY KEY,
          name TEXT NOT NULL,
          version TEXT NOT NULL,
          description TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE evaluation_cases (
          case_id UUID PRIMARY KEY,
          dataset_id UUID NOT NULL REFERENCES evaluation_datasets(dataset_id) ON DELETE CASCADE,
          query TEXT NOT NULL,
          expected_intent TEXT,
          expected_doc_ids UUID[],
          expected_chunk_ids UUID[],
          expected_row_ids UUID[],
          expected_asset_ids UUID[],
          expected_answer JSONB,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );
        CREATE INDEX ix_evaluation_cases_dataset_id ON evaluation_cases(dataset_id);
        CREATE TABLE evaluation_runs (
          eval_run_id UUID PRIMARY KEY,
          dataset_id UUID REFERENCES evaluation_datasets(dataset_id) ON DELETE SET NULL,
          pipeline_version TEXT NOT NULL DEFAULT '0.1.0',
          data_version TEXT NOT NULL DEFAULT 'unknown',
          started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          ended_at TIMESTAMPTZ,
          metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
          status TEXT NOT NULL DEFAULT 'running'
        );
        CREATE INDEX ix_evaluation_runs_status ON evaluation_runs(status);
        """
    ).replace("%(embedding_dim)s", str(_embedding_dim()))
    for statement in schema_sql.split(";"):
        if statement.strip():
            op.execute(statement.strip())


def _embedding_dim() -> int:
    from app.config import get_settings

    return get_settings().embedding_dim


def downgrade() -> None:
    schema_sql = """
        DROP TABLE IF EXISTS evaluation_runs;
        DROP TABLE IF EXISTS evaluation_cases;
        DROP TABLE IF EXISTS evaluation_datasets;
        DROP TABLE IF EXISTS rag_trace_steps;
        DROP TABLE IF EXISTS rag_traces;
        DROP TABLE IF EXISTS ingestion_runs;
        DROP TABLE IF EXISTS faqs;
        DROP TABLE IF EXISTS clinic_info;
        DROP TABLE IF EXISTS services;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS table_rows;
        DROP TABLE IF EXISTS tables;
        DROP TABLE IF EXISTS assets;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS documents;
        """
    for statement in schema_sql.split(";"):
        if statement.strip():
            op.execute(statement.strip())
