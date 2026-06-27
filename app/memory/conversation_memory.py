from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import ConversationSession, ConversationSummary, ConversationTurn


class ConversationMemory:
    def __init__(self, history_turns: int = 8):
        self.history_turns = history_turns

    def load(self, session: Session, session_id: str | None) -> dict[str, Any]:
        if not session_id:
            return {
                "session_id": None,
                "turns": [],
                "summary": None,
                "state": _empty_state(),
            }
        conversation = self._ensure_session(session, session_id)
        # history_turns is configured as conversation exchanges. The table stores
        # both user and assistant rows, so load twice as many rows.
        row_limit = max(1, self.history_turns * 2)
        turns = list(
            session.scalars(
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(desc(ConversationTurn.created_at))
                .limit(row_limit)
            ).all()
        )
        turns.reverse()
        summary = session.scalar(
            select(ConversationSummary)
            .where(ConversationSummary.session_id == session_id)
            .order_by(desc(ConversationSummary.updated_at))
            .limit(1)
        )
        return {
            "session_id": session_id,
            "summary": summary.summary if summary else None,
            "state": _normalize_state(
                (conversation.metadata_json or {}).get("state")
            ),
            "turns": [
                {
                    "turn_id": str(turn.turn_id),
                    "role": turn.role,
                    "content": turn.content,
                    "detected_intents": turn.detected_intents,
                    "entities": turn.entities,
                    "resolved_ids": turn.resolved_ids,
                    "trace_id": str(turn.trace_id) if turn.trace_id else None,
                    "created_at": turn.created_at.isoformat(),
                }
                for turn in turns
            ],
        }

    def save_exchange(
        self,
        session: Session,
        *,
        session_id: str | None,
        user_content: str,
        assistant_content: str,
        detected_intents: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        resolved_ids: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        trace_id=None,
    ) -> dict[str, Any]:
        if not session_id:
            return {"saved": False, "state": _normalize_state(state)}
        conversation = self._ensure_session(session, session_id)
        normalized_state = _normalize_state(state)
        metadata = dict(conversation.metadata_json or {})
        metadata["state"] = normalized_state
        conversation.metadata_json = metadata
        conversation.updated_at = datetime.now(UTC)

        user_turn = ConversationTurn(
            session_id=session_id,
            role="user",
            content=user_content,
            detected_intents=detected_intents or [],
            entities=entities or {},
            resolved_ids=resolved_ids or {},
            trace_id=trace_id,
        )
        assistant_turn = ConversationTurn(
            session_id=session_id,
            role="assistant",
            content=assistant_content,
            detected_intents=detected_intents or [],
            entities=entities or {},
            resolved_ids=resolved_ids or {},
            trace_id=trace_id,
        )
        session.add(conversation)
        session.add_all([user_turn, assistant_turn])
        session.flush()
        summary_text = _build_summary(normalized_state)
        summary = session.scalar(
            select(ConversationSummary)
            .where(ConversationSummary.session_id == session_id)
            .order_by(desc(ConversationSummary.updated_at))
            .limit(1)
        )
        if summary is None:
            summary = ConversationSummary(session_id=session_id, summary=summary_text)
        else:
            summary.summary = summary_text
        summary.last_turn_id = assistant_turn.turn_id
        summary.metadata_json = {"state": normalized_state}
        summary.updated_at = datetime.now(UTC)
        session.add(summary)
        session.commit()
        session.refresh(user_turn)
        session.refresh(assistant_turn)
        return {
            "saved": True,
            "state": normalized_state,
            "user_turn_id": str(user_turn.turn_id),
            "assistant_turn_id": str(assistant_turn.turn_id),
            "summary": summary_text,
        }

    def save_turn(
        self,
        session: Session,
        *,
        session_id: str | None,
        role: str,
        content: str,
        detected_intents: list[str] | None = None,
        entities: dict[str, Any] | None = None,
        resolved_ids: dict[str, Any] | None = None,
        trace_id=None,
    ) -> ConversationTurn | None:
        if not session_id:
            return None
        conversation = self._ensure_session(session, session_id)
        conversation.updated_at = datetime.now(UTC)
        turn = ConversationTurn(
            session_id=session_id,
            role=role,
            content=content,
            detected_intents=detected_intents or [],
            entities=entities or {},
            resolved_ids=resolved_ids or {},
            trace_id=trace_id,
        )
        session.add(conversation)
        session.add(turn)
        session.commit()
        session.refresh(turn)
        return turn

    @staticmethod
    def _ensure_session(session: Session, session_id: str) -> ConversationSession:
        conversation = session.get(ConversationSession, session_id)
        if conversation is not None:
            return conversation
        conversation = ConversationSession(session_id=session_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return conversation


def _empty_state() -> dict[str, Any]:
    return {
        "active_product_ids": [],
        "active_product_names": [],
        "active_service_ids": [],
        "active_service_names": [],
        "active_domain": None,
        "active_topic": None,
        "last_intents": [],
        "last_filters": {},
        "pending_clarification": None,
        "interest_state": {},
        "suggestion_state": {
            "recent_impressions": [],
            "accepted_suggestion_ids": [],
            "dismissed_suggestion_ids": [],
        },
    }


def _normalize_state(value: object) -> dict[str, Any]:
    state = _empty_state()
    if not isinstance(value, dict):
        return state
    state["active_product_ids"] = _string_list(value.get("active_product_ids"))
    state["active_product_names"] = _string_list(value.get("active_product_names"))
    state["active_service_ids"] = _string_list(value.get("active_service_ids"))
    state["active_service_names"] = _string_list(value.get("active_service_names"))
    state["last_intents"] = _string_list(value.get("last_intents"))[:8]
    last_filters = value.get("last_filters")
    state["last_filters"] = last_filters if isinstance(last_filters, dict) else {}
    active_topic = value.get("active_topic")
    state["active_topic"] = str(active_topic).strip() if active_topic else None
    active_domain = str(value.get("active_domain") or "").strip()
    state["active_domain"] = (
        active_domain if active_domain in {"product", "service", "faq", "clinic_info"} else None
    )
    pending = value.get("pending_clarification")
    state["pending_clarification"] = (
        pending if isinstance(pending, dict) or pending is None else {"message": str(pending)}
    )
    interest_state = value.get("interest_state")
    state["interest_state"] = (
        dict(interest_state) if isinstance(interest_state, dict) else {}
    )
    suggestion_state = value.get("suggestion_state")
    if isinstance(suggestion_state, dict):
        state["suggestion_state"] = {
            "recent_impressions": _string_list(
                suggestion_state.get("recent_impressions")
            )[:24],
            "accepted_suggestion_ids": _string_list(
                suggestion_state.get("accepted_suggestion_ids")
            )[:24],
            "dismissed_suggestion_ids": _string_list(
                suggestion_state.get("dismissed_suggestion_ids")
            )[:24],
        }
    return state


def _string_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value if item not in (None, "")]
    else:
        values = [str(value)]
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def _build_summary(state: dict[str, Any]) -> str:
    parts: list[str] = []
    if state.get("active_topic"):
        parts.append(f"Chủ đề hiện tại: {state['active_topic']}.")
    if state.get("active_domain"):
        parts.append(f"Miền hội thoại hiện tại: {state['active_domain']}.")
    if state.get("active_product_names"):
        parts.append(
            "Sản phẩm đang nhắc: "
            + ", ".join(state["active_product_names"][:5])
            + "."
        )
    if state.get("active_service_names"):
        parts.append(
            "Dịch vụ đang nhắc: "
            + ", ".join(state["active_service_names"][:5])
            + "."
        )
    if state.get("last_intents"):
        parts.append("Intent gần nhất: " + ", ".join(state["last_intents"][:5]) + ".")
    interest = state.get("interest_state")
    if isinstance(interest, dict):
        journey_stage = interest.get("journey_stage")
        goals = _string_list(interest.get("goals"))
        if journey_stage:
            parts.append(f"Giai đoạn quan tâm: {journey_stage}.")
        if goals:
            parts.append("Mục tiêu đang quan tâm: " + ", ".join(goals[:4]) + ".")
    pending = state.get("pending_clarification")
    if isinstance(pending, dict) and pending.get("message"):
        parts.append(f"Đang chờ làm rõ: {pending['message']}")
    return " ".join(parts) or "Chưa có chủ đề hội thoại ổn định."
