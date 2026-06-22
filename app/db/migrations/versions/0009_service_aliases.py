"""Add service aliases for DB-backed service entity resolution."""

from alembic import op

revision = "0009_service_aliases"
down_revision = "0008_conversation_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE service_aliases (
          alias_id BIGSERIAL PRIMARY KEY,
          service_id UUID NOT NULL REFERENCES services(service_id) ON DELETE CASCADE,
          alias TEXT NOT NULL,
          normalized_alias TEXT NOT NULL,
          CONSTRAINT uq_service_alias_service_normalized
            UNIQUE(service_id, normalized_alias)
        );
        CREATE INDEX ix_service_aliases_service_id ON service_aliases(service_id);
        CREATE INDEX ix_service_aliases_normalized ON service_aliases(normalized_alias);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS service_aliases;")
