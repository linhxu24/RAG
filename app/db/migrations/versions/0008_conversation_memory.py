"""Add conversation memory tables."""

from alembic import op

revision = "0008_conversation_memory"
down_revision = "0007_catalog_taxonomy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE conversation_sessions (
          session_id TEXT PRIMARY KEY,
          created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
          updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        );

        CREATE TABLE conversation_turns (
          turn_id UUID PRIMARY KEY,
          session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id)
            ON DELETE CASCADE,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          detected_intents TEXT[] NOT NULL DEFAULT '{}',
          entities JSONB NOT NULL DEFAULT '{}'::jsonb,
          resolved_ids JSONB NOT NULL DEFAULT '{}'::jsonb,
          trace_id UUID REFERENCES rag_traces(trace_id) ON DELETE SET NULL,
          created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_conversation_turns_session_id
          ON conversation_turns(session_id, created_at);
        CREATE INDEX ix_conversation_turns_trace_id
          ON conversation_turns(trace_id);

        CREATE TABLE conversation_summaries (
          summary_id UUID PRIMARY KEY,
          session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id)
            ON DELETE CASCADE,
          summary TEXT NOT NULL,
          last_turn_id UUID REFERENCES conversation_turns(turn_id) ON DELETE SET NULL,
          metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
          updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        );
        CREATE INDEX ix_conversation_summaries_session_id
          ON conversation_summaries(session_id);
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS conversation_summaries;
        DROP TABLE IF EXISTS conversation_turns;
        DROP TABLE IF EXISTS conversation_sessions;
        """
    )
