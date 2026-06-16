"""Add grounded evaluation metadata and case-level results."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_eval_results"
down_revision = "0003_asset_links"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("evaluation_datasets", sa.Column("content_hash", sa.Text(), nullable=True))
    op.add_column(
        "evaluation_datasets",
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index(
        "ix_evaluation_datasets_content_hash",
        "evaluation_datasets",
        ["content_hash"],
    )

    op.add_column("evaluation_cases", sa.Column("case_key", sa.Text(), nullable=True))
    op.add_column(
        "evaluation_cases",
        sa.Column("expected_answer_type", sa.Text(), nullable=True),
    )
    for column in (
        "expected_entities",
        "expected_source_keys",
        "expected_answer_contains",
        "forbidden_answer_contains",
    ):
        op.add_column(
            "evaluation_cases",
            sa.Column(
                column,
                postgresql.JSONB(),
                nullable=False,
                server_default=sa.text("'[]'::jsonb"),
            ),
        )
    op.execute("UPDATE evaluation_cases SET case_key = case_id::text WHERE case_key IS NULL")
    op.create_index("ix_evaluation_cases_case_key", "evaluation_cases", ["case_key"])

    op.add_column(
        "evaluation_runs",
        sa.Column(
            "config_snapshot",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    op.create_table(
        "evaluation_case_results",
        sa.Column("result_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("eval_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("expected_intent", sa.Text(), nullable=True),
        sa.Column("actual_intent", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "expected_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "retrieved_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column(
            "scores",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "violations",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "details",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["case_id"],
            ["evaluation_cases.case_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["eval_run_id"],
            ["evaluation_runs.eval_run_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["trace_id"],
            ["rag_traces.trace_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("result_id"),
        sa.UniqueConstraint(
            "eval_run_id",
            "case_id",
            name="uq_eval_case_results_run_case",
        ),
    )
    op.create_index(
        "ix_evaluation_case_results_eval_run_id",
        "evaluation_case_results",
        ["eval_run_id"],
    )
    op.create_index(
        "ix_evaluation_case_results_case_id",
        "evaluation_case_results",
        ["case_id"],
    )
    op.create_index(
        "ix_evaluation_case_results_trace_id",
        "evaluation_case_results",
        ["trace_id"],
    )
    op.create_index(
        "ix_evaluation_case_results_status",
        "evaluation_case_results",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("evaluation_case_results")
    op.drop_column("evaluation_runs", "config_snapshot")
    op.drop_index("ix_evaluation_cases_case_key", table_name="evaluation_cases")
    for column in (
        "forbidden_answer_contains",
        "expected_answer_contains",
        "expected_source_keys",
        "expected_entities",
        "expected_answer_type",
        "case_key",
    ):
        op.drop_column("evaluation_cases", column)
    op.drop_index(
        "ix_evaluation_datasets_content_hash",
        table_name="evaluation_datasets",
    )
    op.drop_column("evaluation_datasets", "metadata")
    op.drop_column("evaluation_datasets", "content_hash")
