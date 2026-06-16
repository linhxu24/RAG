"""Add stable asset identity and many-to-many chunk links."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_asset_links"
down_revision = "0002_ingestion_checksum_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("assets", sa.Column("stable_asset_key", sa.Text(), nullable=True))
    op.execute(
        """
        UPDATE assets
        SET stable_asset_key = doc_id::text || ':' || asset_id::text
        """
    )
    op.alter_column("assets", "stable_asset_key", nullable=False)
    op.drop_constraint("assets_asset_token_key", "assets", type_="unique")
    op.create_index("ix_assets_asset_token", "assets", ["asset_token"], unique=False)
    op.create_unique_constraint(
        "uq_assets_doc_stable_key",
        "assets",
        ["doc_id", "stable_asset_key"],
    )
    op.create_table(
        "chunk_assets",
        sa.Column("chunk_asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("asset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.chunk_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_asset_id"),
        sa.UniqueConstraint("chunk_id", "asset_id", name="uq_chunk_assets_chunk_asset"),
    )
    op.create_index("ix_chunk_assets_asset_id", "chunk_assets", ["asset_id"])
    op.create_index("ix_chunk_assets_chunk_id", "chunk_assets", ["chunk_id"])


def downgrade() -> None:
    op.drop_table("chunk_assets")
    op.drop_constraint("uq_assets_doc_stable_key", "assets", type_="unique")
    op.drop_index("ix_assets_asset_token", table_name="assets")
    op.create_unique_constraint("assets_asset_token_key", "assets", ["asset_token"])
    op.drop_column("assets", "stable_asset_key")
