from inspect import signature
from typing import Any, get_type_hints

from app.config import Settings
from app.constants import Intent
from app.generation.llm_client import LLMClient
from app.orchestration.schemas import (
    BindingSource,
    BoundTask,
    BoundTaskPlan,
    CanonicalFilters,
    EvidenceItem,
    EvidencePack,
    PlannedTask,
    ReferenceMode,
    TaskPlan,
    TrustLevel,
)
from app.orchestration.task_planner import TaskPlanner


def _assert_fields(model, expected):
    fields = model.model_fields
    assert list(fields) == list(expected)
    for name, contract in expected.items():
        assert fields[name].annotation == contract["type"]
        assert fields[name].is_required() is contract["required"]


def test_task_planner_dispatch_signature_contract():
    hints = get_type_hints(TaskPlanner.plan)
    params = signature(TaskPlanner.plan).parameters
    assert list(params) == [
        "self",
        "query",
        "history",
        "settings",
        "llm",
        "ollama",
        "known_products",
        "known_services",
        "known_product_categories",
    ]
    assert hints["query"] is str
    assert hints["history"] == dict[str, Any]
    assert hints["settings"] is Settings
    assert hints["llm"] == LLMClient | None
    assert hints["ollama"] == LLMClient | None
    assert hints["return"] is TaskPlan


def test_planned_and_bound_task_schema_contracts():
    _assert_fields(
        PlannedTask,
        {
            "task_id": {"type": str, "required": True},
            "intent": {"type": Intent, "required": True},
            "planner_query": {"type": str, "required": True},
            "planner_entities": {"type": tuple[str, ...], "required": False},
            "planner_filters": {"type": dict[str, Any], "required": False},
            "planner_sort": {"type": dict[str, Any] | None, "required": False},
            "planner_limit": {"type": int | None, "required": False},
            "planner_entity_type": {"type": str | None, "required": False},
            "priority": {"type": int, "required": False},
            "planner_needs_clarification": {"type": bool, "required": False},
            "planner_clarification_question": {
                "type": str | None,
                "required": False,
            },
        },
    )
    _assert_fields(
        TaskPlan,
        {
            "tasks": {"type": tuple[PlannedTask, ...], "required": True},
            "planner_global_entities": {
                "type": tuple[str, ...],
                "required": False,
            },
            "clarification_question": {
                "type": str | None,
                "required": False,
            },
            "source": {"type": str, "required": False},
            "metadata": {"type": dict[str, Any], "required": False},
        },
    )
    _assert_fields(
        BoundTask,
        {
            "task_id": {"type": str, "required": True},
            "intent": {"type": Intent, "required": True},
            "priority": {"type": int, "required": False},
            "planner_query": {"type": str, "required": True},
            "effective_query": {"type": str, "required": True},
            "entity_type": {"type": str | None, "required": False},
            "entity_names": {"type": tuple[str, ...], "required": False},
            "resolved_ids": {"type": tuple[str, ...], "required": False},
            "filters": {"type": CanonicalFilters, "required": False},
            "binding_source": {"type": BindingSource, "required": False},
            "reference_mode": {"type": ReferenceMode, "required": False},
            "inherited_from_task_id": {
                "type": str | None,
                "required": False,
            },
            "resolution_status": {"type": str, "required": False},
            "clarification_required": {"type": bool, "required": False},
            "clarification_question": {
                "type": str | None,
                "required": False,
            },
            "operation": {"type": str | None, "required": False},
            "capability_version": {"type": str, "required": False},
        },
    )


def test_evidence_pack_boundary_schema_contract():
    _assert_fields(
        EvidenceItem,
        {
            "task_id": {"type": str, "required": True},
            "source_type": {"type": str, "required": True},
            "source_id": {"type": str, "required": True},
            "text": {"type": str, "required": True},
            "score": {"type": float, "required": False},
            "trust_level": {"type": TrustLevel, "required": False},
            "raw_json": {"type": dict[str, Any], "required": False},
            "source": {"type": dict[str, Any], "required": False},
            "canonical_key": {"type": str | None, "required": False},
            "asset_ids": {"type": list[str], "required": False},
        },
    )
    _assert_fields(
        EvidencePack,
        {
            "query": {"type": str, "required": True},
            "tasks": {"type": tuple[BoundTask, ...], "required": True},
            "items": {"type": list[EvidenceItem], "required": True},
            "conflicts": {"type": list[dict[str, Any]], "required": False},
            "missing_info": {"type": list[str], "required": False},
        },
    )
    assert BoundTaskPlan.model_fields["tasks"].annotation == tuple[BoundTask, ...]
