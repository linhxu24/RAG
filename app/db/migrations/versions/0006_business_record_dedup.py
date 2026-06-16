"""Enforce one active business record per normalized identity."""

from alembic import op

revision = "0006_business_record_dedup"
down_revision = "0005_vietnamese_retrieval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        WITH ranked AS (
          SELECT
            p.product_id,
            row_number() OVER (
              PARTITION BY lower(simplydent_unaccent(p.name))
              ORDER BY d.uploaded_at DESC NULLS LAST,
                       p.valid_from DESC NULLS LAST,
                       p.product_id DESC
            ) AS duplicate_rank
          FROM products p
          LEFT JOIN documents d ON d.doc_id = p.source_doc_id
          WHERE p.status = 'active'
        )
        UPDATE products p
        SET status = 'archived',
            valid_to = coalesce(p.valid_to, now())
        FROM ranked r
        WHERE p.product_id = r.product_id
          AND r.duplicate_rank > 1
        """
    )
    op.execute(
        """
        WITH ranked AS (
          SELECT
            s.service_id,
            row_number() OVER (
              PARTITION BY lower(simplydent_unaccent(s.name))
              ORDER BY d.uploaded_at DESC NULLS LAST,
                       s.valid_from DESC NULLS LAST,
                       s.service_id DESC
            ) AS duplicate_rank
          FROM services s
          LEFT JOIN documents d ON d.doc_id = s.source_doc_id
          WHERE s.status = 'active'
        )
        UPDATE services s
        SET status = 'archived',
            valid_to = coalesce(s.valid_to, now())
        FROM ranked r
        WHERE s.service_id = r.service_id
          AND r.duplicate_rank > 1
        """
    )
    op.execute(
        """
        WITH ranked AS (
          SELECT
            f.faq_id,
            row_number() OVER (
              PARTITION BY lower(simplydent_unaccent(f.question))
              ORDER BY d.uploaded_at DESC NULLS LAST, f.faq_id DESC
            ) AS duplicate_rank
          FROM faqs f
          LEFT JOIN documents d
            ON d.doc_id::text = f.metadata ->> 'source_doc_id'
          WHERE f.is_active = true
        )
        UPDATE faqs f
        SET is_active = false
        FROM ranked r
        WHERE f.faq_id = r.faq_id
          AND r.duplicate_rank > 1
        """
    )
    op.execute(
        """
        WITH ranked AS (
          SELECT
            c.id,
            row_number() OVER (
              PARTITION BY lower(simplydent_unaccent(c.key))
              ORDER BY d.uploaded_at DESC NULLS LAST, c.id DESC
            ) AS duplicate_rank
          FROM clinic_info c
          LEFT JOIN documents d ON d.doc_id = c.source_doc_id
          WHERE c.status = 'active'
        )
        UPDATE clinic_info c
        SET status = 'archived'
        FROM ranked r
        WHERE c.id = r.id
          AND r.duplicate_rank > 1
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX uq_products_active_normalized_name
        ON products ((lower(simplydent_unaccent(name))))
        WHERE status = 'active'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_services_active_normalized_name
        ON services ((lower(simplydent_unaccent(name))))
        WHERE status = 'active'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_faqs_active_normalized_question
        ON faqs ((lower(simplydent_unaccent(question))))
        WHERE is_active = true
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_clinic_info_active_normalized_key
        ON clinic_info ((lower(simplydent_unaccent(key))))
        WHERE status = 'active'
        """
    )


def downgrade() -> None:
    for index_name in (
        "uq_clinic_info_active_normalized_key",
        "uq_faqs_active_normalized_question",
        "uq_services_active_normalized_name",
        "uq_products_active_normalized_name",
    ):
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
