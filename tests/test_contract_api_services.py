from inspect import signature
from typing import Any

from sqlalchemy.orm import Session

from app.api.routes_chat import chat
from app.constants import Intent
from app.generation.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSuggestion,
    ResultBody,
    SafetyInfo,
)
from app.services.chat import ChatService


def _assert_model_fields(model, expected):
    fields = model.model_fields
    assert list(fields) == list(expected)
    for name, contract in expected.items():
        assert fields[name].annotation == contract["type"]
        assert fields[name].is_required() is contract["required"]
        assert (fields[name].default_factory is not None) is contract.get(
            "default_factory",
            False,
        )
        if "default" in contract:
            assert fields[name].default == contract["default"]


def test_api_chat_route_hands_exact_request_contract_to_service():
    route_signature = signature(chat)
    service_signature = signature(ChatService.chat)

    assert route_signature.parameters["request"].annotation is ChatRequest
    assert route_signature.parameters["session"].annotation is Session
    assert route_signature.return_annotation is ChatResponse

    assert service_signature.parameters["session"].annotation is Session
    assert service_signature.parameters["request"].annotation is ChatRequest
    assert service_signature.return_annotation is ChatResponse


def test_chat_request_response_schema_contract():
    _assert_model_fields(
        ChatRequest,
        {
            "message": {"type": str, "required": True},
            "session_id": {"type": str | None, "required": False, "default": None},
            "selected_suggestion_id": {
                "type": str | None,
                "required": False,
                "default": None,
            },
            "history": {
                "type": list[dict[str, Any]],
                "required": False,
                "default_factory": True,
            },
            "debug": {"type": bool, "required": False, "default": False},
        },
    )
    _assert_model_fields(
        ChatResponse,
        {
            "trace_id": {"type": str, "required": True},
            "intent": {"type": Intent, "required": True},
            "answer_type": {
                "type": str | None,
                "required": False,
                "default": None,
            },
            "answer": {"type": ResultBody, "required": True},
            "safety": {
                "type": SafetyInfo,
                "required": False,
                "default_factory": True,
            },
            "degraded": {"type": bool, "required": False, "default": False},
            "suggestions": {
                "type": list[ChatSuggestion],
                "required": False,
                "default_factory": True,
            },
            "debug": {
                "type": dict[str, Any],
                "required": False,
                "default_factory": True,
            },
        },
    )
