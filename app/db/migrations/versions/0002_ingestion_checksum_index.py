"""Index document checksums used by ingestion deduplication."""

from alembic import op

revision = "0002_ingestion_checksum_index"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_documents_checksum", "documents", ["checksum"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_documents_checksum", table_name="documents")
