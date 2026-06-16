"""Add Vietnamese-normalized full-text and trigram search."""

from alembic import op

revision = "0005_vietnamese_retrieval"
down_revision = "0004_eval_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION simplydent_unaccent(value text)
        RETURNS text
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        STRICT
        AS $$ SELECT public.unaccent(value) $$
        """
    )

    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN content_tsv")
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv TSVECTOR GENERATED ALWAYS AS (
          to_tsvector('simple', simplydent_unaccent(coalesce(content, '')))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN(content_tsv)")
    op.execute(
        """
        CREATE INDEX ix_chunks_content_trgm
        ON chunks USING GIN (
          (lower(simplydent_unaccent(content))) gin_trgm_ops
        )
        """
    )

    op.execute("DROP INDEX IF EXISTS ix_table_rows_row_tsv")
    op.execute("ALTER TABLE table_rows DROP COLUMN row_tsv")
    op.execute(
        """
        ALTER TABLE table_rows
        ADD COLUMN row_tsv TSVECTOR GENERATED ALWAYS AS (
          to_tsvector('simple', simplydent_unaccent(coalesce(row_text, '')))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_table_rows_row_tsv ON table_rows USING GIN(row_tsv)")
    op.execute(
        """
        CREATE INDEX ix_table_rows_text_trgm
        ON table_rows USING GIN (
          (lower(simplydent_unaccent(row_text))) gin_trgm_ops
        )
        """
    )

    op.execute("DROP INDEX IF EXISTS ix_faqs_question_tsv")
    op.execute("ALTER TABLE faqs DROP COLUMN question_tsv")
    op.execute(
        """
        ALTER TABLE faqs
        ADD COLUMN question_tsv TSVECTOR GENERATED ALWAYS AS (
          to_tsvector(
            'simple',
            simplydent_unaccent(
              coalesce(question, '') || ' ' || coalesce(answer, '')
            )
          )
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_faqs_question_tsv ON faqs USING GIN(question_tsv)")
    op.execute(
        """
        CREATE INDEX ix_faqs_text_trgm
        ON faqs USING GIN (
          (
            lower(
              simplydent_unaccent(
                coalesce(question, '') || ' ' || coalesce(answer, '')
              )
            )
          ) gin_trgm_ops
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_products_name_unaccent_trgm
        ON products USING GIN (
          (lower(simplydent_unaccent(name))) gin_trgm_ops
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_services_name_unaccent_trgm
        ON services USING GIN (
          (lower(simplydent_unaccent(name))) gin_trgm_ops
        )
        """
    )


def downgrade() -> None:
    for index_name in (
        "ix_services_name_unaccent_trgm",
        "ix_products_name_unaccent_trgm",
        "ix_faqs_text_trgm",
        "ix_table_rows_text_trgm",
        "ix_chunks_content_trgm",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index_name}")

    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN content_tsv")
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv TSVECTOR GENERATED ALWAYS AS (
          to_tsvector('simple', coalesce(content, ''))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN(content_tsv)")

    op.execute("DROP INDEX IF EXISTS ix_table_rows_row_tsv")
    op.execute("ALTER TABLE table_rows DROP COLUMN row_tsv")
    op.execute(
        """
        ALTER TABLE table_rows
        ADD COLUMN row_tsv TSVECTOR GENERATED ALWAYS AS (
          to_tsvector('simple', coalesce(row_text, ''))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_table_rows_row_tsv ON table_rows USING GIN(row_tsv)")

    op.execute("DROP INDEX IF EXISTS ix_faqs_question_tsv")
    op.execute("ALTER TABLE faqs DROP COLUMN question_tsv")
    op.execute(
        """
        ALTER TABLE faqs
        ADD COLUMN question_tsv TSVECTOR GENERATED ALWAYS AS (
          to_tsvector('simple', coalesce(question, ''))
        ) STORED
        """
    )
    op.execute("CREATE INDEX ix_faqs_question_tsv ON faqs USING GIN(question_tsv)")
    op.execute("DROP FUNCTION IF EXISTS simplydent_unaccent(text)")
