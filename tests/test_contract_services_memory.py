from inspect import Parameter, signature
from typing import Any, get_type_hints

from sqlalchemy.orm import Session

from app.memory.conversation_memory import ConversationMemory, _normalize_state


def test_services_memory_method_signatures_are_stable():
    load_signature = signature(ConversationMemory.load)
    load_hints = get_type_hints(ConversationMemory.load)
    assert list(load_signature.parameters) == ["self", "session", "session_id"]
    assert load_hints["session"] is Session
    assert load_hints["session_id"] == str | None
    assert load_hints["return"] == dict[str, Any]

    save_signature = signature(ConversationMemory.save_exchange)
    assert list(save_signature.parameters) == [
        "self",
        "session",
        "session_id",
        "user_content",
        "assistant_content",
        "detected_intents",
        "entities",
        "resolved_ids",
        "state",
        "trace_id",
    ]
    assert save_signature.parameters["session"].annotation is Session
    for name in (
        "session_id",
        "user_content",
        "assistant_content",
        "detected_intents",
        "entities",
        "resolved_ids",
        "state",
    ):
        assert save_signature.parameters[name].kind is Parameter.KEYWORD_ONLY
    assert save_signature.parameters["trace_id"].default is None


def test_conversation_state_contract_exact_keys_and_types():
    state = _normalize_state({})
    assert list(state) == [
        "active_product_ids",
        "active_product_names",
        "active_service_ids",
        "active_service_names",
        "active_domain",
        "active_topic",
        "last_intents",
        "last_filters",
        "pending_clarification",
        "interest_state",
        "suggestion_state",
    ]
    assert isinstance(state["active_product_ids"], list)
    assert isinstance(state["active_product_names"], list)
    assert isinstance(state["active_service_ids"], list)
    assert isinstance(state["active_service_names"], list)
    assert state["active_domain"] is None
    assert state["active_topic"] is None
    assert isinstance(state["last_intents"], list)
    assert isinstance(state["last_filters"], dict)
    assert state["pending_clarification"] is None
    assert isinstance(state["interest_state"], dict)
    assert state["suggestion_state"] == {
        "recent_impressions": [],
        "accepted_suggestion_ids": [],
        "dismissed_suggestion_ids": [],
    }
