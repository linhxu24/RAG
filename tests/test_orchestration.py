import asyncio
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.constants import Intent
from app.generation.llm_client import LLMResponse
from app.generation.validator import ResponseValidationError, ResponseValidator
from app.ner.entity_span_extractor import EntitySpan, SpanExtractionResult
from app.orchestration.binding_pipeline import TaskBindingPipeline
from app.orchestration.consistency_gate import ConsistencyGate
from app.orchestration.context_binder import ContextBinder
from app.orchestration.evidence_merger import EvidenceMerger
from app.orchestration.intent_registry import (
    INTENT_CAPABILITIES,
    EntityScope,
    capability_for,
    validate_intent_registry,
)
from app.orchestration.schemas import (
    BindingDecision,
    BindingSource,
    BoundTask,
    BoundTaskPlan,
    CanonicalFilters,
    EvidenceItem,
    EvidencePack,
    GateStatus,
    PlannedTask,
    ReferenceMode,
    TaskPlan,
    TaskResolution,
)
from app.orchestration.task_canonicalizer import TaskCanonicalizer
from app.orchestration.task_planner import TaskPlanner
from app.orchestration.tool_executor import ToolExecutor
from app.retrieval.types import RetrievalResult
from app.services.chat import ChatService


def planned_task(
    *,
    intent: Intent = Intent.SERVICE_DETAIL,
    query: str = "Tẩy trắng răng giá bao nhiêu?",
    entities: tuple[str, ...] = ("Tẩy trắng răng",),
    filters: dict | None = None,
    entity_type: str | None = "service",
    task_id: str = "t1",
    priority: int = 1,
) -> PlannedTask:
    return PlannedTask(
        task_id=task_id,
        intent=intent,
        planner_query=query,
        planner_entities=entities,
        planner_filters=filters or {},
        planner_entity_type=entity_type,
        priority=priority,
    )


def bound_task(
    *,
    intent: Intent = Intent.SERVICE_DETAIL,
    task_id: str = "t1",
    entity_name: str = "Tẩy trắng răng tại phòng khám",
    entity_id: str = "service-1",
    entity_type: str = "service",
    effective_query: str | None = None,
    reference_mode: ReferenceMode = ReferenceMode.EXPLICIT,
) -> BoundTask:
    product = entity_type == "product"
    return BoundTask(
        task_id=task_id,
        intent=intent,
        planner_query="planner proposal",
        effective_query=effective_query
        or f"{entity_name}. Giá bao nhiêu?",
        entity_type=entity_type,
        entity_names=(entity_name,),
        resolved_ids=(entity_id,),
        filters=CanonicalFilters(
            product_ids=(entity_id,) if product else (),
            product_names=(entity_name,) if product else (),
            service_ids=() if product else (entity_id,),
            service_names=() if product else (entity_name,),
        ),
        binding_source=BindingSource.EXPLICIT_SPAN,
        reference_mode=reference_mode,
        resolution_status="resolved",
    )


def test_intent_registry_covers_every_intent_and_is_valid():
    validate_intent_registry()
    assert set(INTENT_CAPABILITIES) == set(Intent)
    assert capability_for(Intent.PRODUCT_DETAIL).entity_scope == EntityScope.EXACTLY_ONE
    assert capability_for(Intent.PRODUCT_DETAIL).entity_domain == "product"
    assert capability_for(Intent.SERVICE_DETAIL).entity_domain == "service"
    assert capability_for(Intent.CLINIC_INFO).entity_domain is None
    assert capability_for(Intent.FAQ).entity_domain is None
    assert capability_for(Intent.FAQ).allowed_tools == (
        "faq_tool",
        "document_rag_tool",
    )


def test_core_orchestration_has_no_intent_prefix_domain_inference():
    root = Path(__file__).resolve().parents[1]
    core_files = [
        root / "app/services/chat.py",
        root / "app/orchestration/context_binder.py",
        root / "app/orchestration/task_canonicalizer.py",
        root / "app/orchestration/task_resolver.py",
        root / "app/orchestration/tool_executor.py",
        root / "app/orchestration/evidence_merger.py",
        root / "app/retrieval/entity_resolver.py",
    ]

    for path in core_files:
        text = path.read_text()
        assert "intent.name.startswith" not in text
        assert 'startswith("PRODUCT_' not in text
        assert 'startswith("SERVICE_' not in text


def test_planned_task_is_frozen_untrusted_proposal():
    task = planned_task()

    assert task.planner_query == "Tẩy trắng răng giá bao nhiêu?"
    assert not hasattr(task, "effective_query")
    assert not hasattr(task, "resolved_ids")
    with pytest.raises(ValidationError):
        task.planner_query = "mutated"


def test_heuristic_planner_decomposes_multi_intent_query_as_proposals():
    plan = asyncio.run(
        TaskPlanner().plan(
            query="Tẩy trắng răng giá bao nhiêu và có đau không?",
            history={"turns": []},
            settings=Settings(enable_llm_router=False, max_sub_queries=3),
            llm=None,
            known_products=[],
            known_services=["Tẩy trắng răng"],
            known_product_categories=[],
        )
    )

    assert [task.intent for task in plan.tasks] == [
        Intent.SERVICE_DETAIL,
        Intent.FAQ,
    ]
    assert plan.tasks[0].planner_query == "Tẩy trắng răng giá bao nhiêu"
    assert not hasattr(plan.tasks[0], "resolved_ids")


def test_heuristic_planner_uses_query_features_for_product_follow_up():
    plan = asyncio.run(
        TaskPlanner().plan(
            query="Giá bao nhiêu?",
            history={
                "state": {
                    "active_domain": "product",
                    "active_product_names": ["Máy tăm nước AquaJet"],
                    "last_intents": ["PRODUCT_DETAIL"],
                },
                "turns": [{}],
            },
            settings=Settings(enable_llm_router=False),
            llm=None,
            known_products=[],
            known_services=[],
            known_product_categories=[],
        )
    )

    assert plan.tasks[0].intent == Intent.PRODUCT_DETAIL
    assert plan.tasks[0].planner_entities == ("Máy tăm nước AquaJet",)


def test_heuristic_planner_keeps_social_follow_up_as_chitchat():
    plan = asyncio.run(
        TaskPlanner().plan(
            query="Cảm ơn",
            history={
                "state": {
                    "active_domain": "service",
                    "active_service_names": ["Tẩy trắng răng"],
                    "last_intents": ["SERVICE_DETAIL"],
                },
                "turns": [{}],
            },
            settings=Settings(enable_llm_router=False),
            llm=None,
            known_products=[],
            known_services=[],
            known_product_categories=[],
        )
    )

    assert plan.tasks[0].intent == Intent.CHITCHAT
    assert plan.tasks[0].planner_entities == ()


def test_planner_preserves_llm_usage_when_empty_plan_returns_safe_unknown():
    class EmptyPlannerLLM:
        async def generate(self, **_kwargs):
            return LLMResponse(
                text='{"tasks":[],"global_entities":[]}',
                latency_ms=123,
                model="local-router",
                prompt_eval_count=321,
                eval_count=7,
            )

    plan = asyncio.run(
        TaskPlanner().plan(
            query="Cho tôi danh sách sản phẩm đang có",
            history={"turns": []},
            settings=Settings(
                enable_llm_router=True,
                enable_plan_review=False,
            ),
            llm=EmptyPlannerLLM(),
            known_products=[],
            known_services=[],
            known_product_categories=[],
        )
    )

    assert plan.source == "safe_unknown"
    assert plan.tasks[0].intent == Intent.UNKNOWN
    assert plan.tasks[0].planner_needs_clarification is True
    assert plan.metadata["planner_fallback"] == (
        "safe_unknown_after_empty_llm_plan"
    )
    assert plan.metadata["usage"] == {
        "prompt_tokens": 321,
        "completion_tokens": 7,
        "total_tokens": 328,
    }


def test_planner_llm_timeout_falls_back_to_deterministic_valid_intents():
    class TimeoutPlannerLLM:
        async def generate(self, **_kwargs):
            raise TimeoutError("router model timed out")

    cases = {
        "Xin chào": Intent.GREETING,
        "Danh sách sản phẩm đang bán": Intent.PRODUCT_LIST,
        "Dịch vụ đang có": Intent.SERVICE_LIST,
        "Cuối tuần phòng khám mở cửa không?": Intent.CLINIC_INFO,
        "Tẩy trắng răng có đau không?": Intent.FAQ,
    }

    for query, expected_intent in cases.items():
        plan = asyncio.run(
            TaskPlanner().plan(
                query=query,
                history={"turns": [], "state": {}},
                settings=Settings(enable_llm_router=True),
                llm=TimeoutPlannerLLM(),
                known_products=[],
                known_services=["Tẩy trắng răng"],
                known_product_categories=[],
            )
        )

        assert plan.source == "heuristic"
        assert plan.metadata["planner_fallback"] == "heuristic_after_llm_error"
        assert plan.tasks[0].intent == expected_intent
        assert plan.tasks[0].intent != Intent.UNKNOWN
        assert plan.tasks[0].planner_needs_clarification is False


def test_valid_timeout_fallback_queries_do_not_emit_generic_unknown_clarification():
    class TimeoutPlannerLLM:
        async def generate(self, **_kwargs):
            raise TimeoutError("router model timed out")

    generic = "Bạn vui lòng cho biết cụ thể muốn hỏi về sản phẩm"
    cases = {
        "Xin chào": Intent.GREETING,
        "Danh sách sản phẩm đang bán": Intent.PRODUCT_LIST,
        "Dịch vụ đang có": Intent.SERVICE_LIST,
        "Cuối tuần phòng khám mở cửa không?": Intent.CLINIC_INFO,
        "Tẩy trắng răng có đau không?": Intent.FAQ,
    }

    for query, expected_intent in cases.items():
        plan = asyncio.run(
            TaskPlanner().plan(
                query=query,
                history={"turns": [], "state": {}},
                settings=Settings(enable_llm_router=True),
                llm=TimeoutPlannerLLM(),
                known_products=[],
                known_services=["Tẩy trắng răng"],
                known_product_categories=[],
            )
        )
        span = EntitySpan(
            text="Tẩy trắng răng",
            label="service_name",
            score=1.0,
            source="catalog_match",
            metadata={"catalog_name": "Tẩy trắng răng"},
        )
        pipeline = TaskBindingPipeline(
            binder=ContextBinder(Settings()),
            resolver=_DeterministicResolver(),
            canonicalizer=TaskCanonicalizer(),
            consistency_gate=ConsistencyGate(),
        )
        result = pipeline.run(
            None,
            plan=plan,
            original_query=query,
            history={"turns": [], "state": {}},
            span_result=SpanExtractionResult(
                query=query,
                spans=[span] if expected_intent == Intent.FAQ else [],
            ),
        )

        task = result.bound_plan.tasks[0]
        assert task.intent == expected_intent
        assert generic not in str(result.as_dict())
        if expected_intent in {
            Intent.PRODUCT_LIST,
            Intent.SERVICE_LIST,
            Intent.CLINIC_INFO,
            Intent.FAQ,
        }:
            assert capability_for(task.intent).allowed_tools


def test_list_intents_do_not_require_entity_ids_before_tool_execution():
    for intent, entity_type in (
        (Intent.PRODUCT_LIST, "product"),
        (Intent.SERVICE_LIST, "service"),
    ):
        task = BoundTask(
            task_id="t1",
            intent=intent,
            planner_query="catalog overview",
            effective_query="catalog overview",
            entity_type=entity_type,
            filters=CanonicalFilters(),
            reference_mode=ReferenceMode.NO_ENTITY,
            binding_source=BindingSource.TASK_FILTERS,
            resolution_status="not_applicable",
        )

        report = ConsistencyGate().check_bound_plan(BoundTaskPlan(tasks=(task,)))

        assert report.status == GateStatus.PASS
        assert report.valid_task_ids == ("t1",)


def test_unknown_generic_clarification_is_only_for_ambiguous_unknown_task():
    plan = asyncio.run(
        TaskPlanner().plan(
            query="abc",
            history={"turns": [], "state": {}},
            settings=Settings(enable_llm_router=False),
            llm=None,
            known_products=[],
            known_services=[],
            known_product_categories=[],
        )
    )
    result = TaskBindingPipeline(
        binder=ContextBinder(Settings()),
        resolver=_DeterministicResolver(),
        canonicalizer=TaskCanonicalizer(),
        consistency_gate=ConsistencyGate(),
    ).run(
        None,
        plan=plan,
        original_query="abc",
        history={"turns": [], "state": {}},
        span_result=SpanExtractionResult(query="abc"),
    )

    task = result.bound_plan.tasks[0]
    assert task.intent == Intent.UNKNOWN
    assert task.clarification_required is True
    assert "Bạn vui lòng cho biết cụ thể muốn hỏi" in task.clarification_question


def test_binder_does_not_use_conversation_entity_pointer_for_follow_up():
    task = planned_task(
        query="Mặt dán sứ Veneer mất bao lâu",
        entities=("Mặt dán sứ Veneer",),
    )
    plan = TaskPlan(
        tasks=(task,),
        planner_global_entities=("Mặt dán sứ Veneer",),
    )

    decision = ContextBinder(Settings()).bind(
        plan=plan,
        original_query="Mất bao lâu?",
        history={
            "state": {
                "active_domain": "service",
                "active_service_ids": ["service-whitening"],
                "active_service_names": ["Tẩy trắng răng tại phòng khám"],
            }
        },
        span_result=SpanExtractionResult(query="Mất bao lâu?"),
    ).decisions[0]

    assert task.planner_query == "Mặt dán sứ Veneer mất bao lâu"
    assert task.planner_entities == ("Mặt dán sứ Veneer",)
    assert decision.entity_names == ()
    assert decision.inherited_resolved_ids == ()
    assert decision.rejected_planner_entities == ("Mặt dán sứ Veneer",)
    assert decision.binding_source == BindingSource.NONE
    assert decision.clarification_required is True
    assert decision.reason_codes == ("missing_authoritative_entity_context",)


def test_binder_explicit_entity_switch_overrides_memory():
    task = planned_task(
        query="Mặt dán sứ Veneer mất bao lâu",
        entities=("Mặt dán sứ Veneer",),
    )
    span = EntitySpan(
        text="Mặt dán sứ Veneer",
        label="service_name",
        score=1.0,
        source="catalog_match",
        metadata={"catalog_name": "Mặt dán sứ Veneer"},
    )

    decision = ContextBinder(Settings()).bind_task(
        task=task,
        original_query=task.planner_query,
        history={
            "state": {
                "active_service_names": [
                    "Tẩy trắng răng tại phòng khám"
                ]
            }
        },
        span_result=SpanExtractionResult(
            query=task.planner_query,
            spans=[span],
        ),
    )

    assert decision.entity_names == ("Mặt dán sứ Veneer",)
    assert decision.binding_source == BindingSource.EXPLICIT_SPAN
    assert decision.reference_mode == ReferenceMode.EXPLICIT
    assert decision.inherited_resolved_ids == ()


def test_clinic_info_does_not_inherit_product_or_service_memory():
    task = planned_task(
        intent=Intent.CLINIC_INFO,
        query="Cuối tuần phòng khám làm việc không?",
        entities=("Tẩy trắng răng tại phòng khám",),
        entity_type="service",
    )

    decision = ContextBinder(Settings()).bind_task(
        task=task,
        original_query=task.planner_query,
        history={
            "state": {
                "active_domain": "service",
                "active_service_names": ["Tẩy trắng răng tại phòng khám"],
                "active_service_ids": ["service-whitening"],
            }
        },
        span_result=SpanExtractionResult(query=task.planner_query),
    )

    assert decision.reference_mode == ReferenceMode.NO_ENTITY
    assert decision.binding_source == BindingSource.NONE
    assert decision.entity_names == ()
    assert decision.inherited_resolved_ids == ()
    assert decision.rejected_planner_entities == ("Tẩy trắng răng tại phòng khám",)


def test_binder_compare_uses_rewritten_explicit_entities_not_memory_pointer():
    task = planned_task(
        intent=Intent.PRODUCT_COMPARE,
        query=(
            "So sánh FreshMint Total Protection Toothpaste với "
            "EnamelGuard Sensitive Toothpaste"
        ),
        entities=(
            "FreshMint Total Protection Toothpaste",
            "EnamelGuard Sensitive Toothpaste",
        ),
        entity_type="product",
    )
    spans = (
        EntitySpan(
            text="FreshMint Total Protection Toothpaste",
            label="product_name",
            score=1.0,
            source="catalog_match",
            metadata={"catalog_name": "FreshMint Total Protection Toothpaste"},
        ),
        EntitySpan(
            text="EnamelGuard Sensitive Toothpaste",
            label="product_name",
            score=1.0,
            source="catalog_match",
            metadata={"catalog_name": "EnamelGuard Sensitive Toothpaste"},
        ),
    )

    decision = ContextBinder(Settings()).bind_task(
        task=task,
        original_query=task.planner_query,
        history={
            "state": {
                "active_domain": "product",
                "active_product_names": [
                    "FreshMint Total Protection Toothpaste"
                ],
                "active_product_ids": ["product-memory"],
            }
        },
        span_result=SpanExtractionResult(
            query=task.planner_query,
            spans=list(spans),
        ),
    )

    assert decision.reference_mode == ReferenceMode.COMPARE
    assert decision.entity_names == (
        "FreshMint Total Protection Toothpaste",
        "EnamelGuard Sensitive Toothpaste",
    )
    assert decision.binding_source == BindingSource.EXPLICIT_SPAN
    assert decision.inherited_resolved_ids == ()


def test_faq_follow_up_uses_entity_from_rewritten_query_span():
    task = planned_task(
        intent=Intent.FAQ,
        query="Tẩy trắng răng tại phòng khám có đau không?",
        entities=("Tẩy trắng răng tại phòng khám",),
        entity_type=None,
    )
    span = EntitySpan(
        text="Tẩy trắng răng tại phòng khám",
        label="service_name",
        score=1.0,
        source="catalog_match",
        metadata={"catalog_name": "Tẩy trắng răng tại phòng khám"},
    )
    decision = ContextBinder(Settings()).bind_task(
        task=task,
        original_query=task.planner_query,
        history={
            "state": {
                "active_domain": "service",
                "active_service_names": [
                    "Tẩy trắng răng tại phòng khám"
                ],
                "active_service_ids": ["service-whitening"],
            }
        },
        span_result=SpanExtractionResult(query=task.planner_query, spans=[span]),
    )

    assert decision.entity_type == "service"
    assert decision.entity_names == (
        "Tẩy trắng răng tại phòng khám",
    )
    assert decision.inherited_resolved_ids == ()
    assert decision.binding_source == BindingSource.EXPLICIT_SPAN


def test_filter_refinement_reuses_memory_filters_during_canonicalization():
    task = planned_task(
        intent=Intent.PRODUCT_LIST,
        query="Trong số đó loại nào còn hàng?",
        entities=(),
        filters={"stock": "available"},
        entity_type="product",
    )
    decision = ContextBinder(Settings()).bind_task(
        task=task,
        original_query=task.planner_query,
        history={
            "state": {
                "last_filters": {"product": {"price_max": 500000}}
            }
        },
        span_result=SpanExtractionResult(query=task.planner_query),
    )
    canonical = TaskCanonicalizer().canonicalize(
        task=task,
        decision=decision,
        resolution=TaskResolution(
            task_id="t1",
            status="not_applicable",
            entity_type="product",
        ),
        history={
            "state": {
                "last_filters": {"product": {"price_max": 500000}}
            }
        },
    )

    assert decision.reference_mode == ReferenceMode.FILTER_REFINEMENT
    assert canonical.filters.price_max == 500000
    assert canonical.filters.stock == "available"
    assert canonical.entity_names == ()
    assert canonical.resolved_ids == ()


def test_canonicalizer_uses_rewritten_explicit_entity_state():
    service_id = str(uuid.uuid4())
    task = planned_task(
        query="Mặt dán sứ Veneer mất bao lâu",
        entities=("Mặt dán sứ Veneer",),
    )
    decision = BindingDecision(
        task_id="t1",
        intent=Intent.SERVICE_DETAIL,
        entity_type="service",
        reference_mode=ReferenceMode.EXPLICIT,
        binding_source=BindingSource.EXPLICIT_SPAN,
        entity_names=("Mặt dán sứ Veneer",),
    )
    resolution = TaskResolution(
        task_id="t1",
        status="resolved",
        entity_type="service",
        entity_names=("Mặt dán sứ Veneer",),
        resolved_ids=(service_id,),
        source="database",
    )

    bound = TaskCanonicalizer().canonicalize(
        task=task,
        decision=decision,
        resolution=resolution,
        history={"state": {}},
    )

    assert "Mặt dán sứ Veneer" in bound.effective_query
    assert "Tẩy trắng răng tại phòng khám" not in bound.effective_query
    assert bound.entity_names == ("Mặt dán sứ Veneer",)
    assert bound.resolved_ids == (service_id,)
    assert bound.filters.service_ids == (service_id,)
    assert bound.filters.service_names == ("Mặt dán sứ Veneer",)


class _DeterministicResolver:
    def __init__(self):
        self.ids = {
            "Tẩy trắng răng tại phòng khám": "service-whitening",
            "Mặt dán sứ Veneer": "service-veneer",
            "Máy tăm nước": "product-water-flosser",
        }

    def resolve(self, _session, *, task, decision):
        names = decision.entity_names
        resolved_ids = tuple(
            self.ids[name]
            for name in names
            if name in self.ids
        )
        return TaskResolution(
            task_id=task.task_id,
            status=(
                "resolved"
                if names and len(resolved_ids) == len(names)
                else "not_applicable"
                if not names
                else "not_found"
            ),
            entity_type=decision.entity_type,
            entity_names=names if resolved_ids else (),
            resolved_ids=resolved_ids,
        )


def test_faq_inherits_only_from_resolved_same_turn_bound_task():
    service = "Tẩy trắng răng tại phòng khám"
    span = EntitySpan(
        text="Tẩy trắng răng",
        label="service_name",
        score=1.0,
        source="catalog_match",
        metadata={"catalog_name": service},
    )
    plan = TaskPlan(
        tasks=(
            planned_task(
                query="Tẩy trắng răng giá bao nhiêu",
                entities=("Tẩy trắng răng",),
            ),
            planned_task(
                task_id="t2",
                intent=Intent.FAQ,
                query="có đau không?",
                entities=(),
                entity_type=None,
                priority=2,
            ),
        )
    )
    pipeline = TaskBindingPipeline(
        binder=ContextBinder(Settings()),
        resolver=_DeterministicResolver(),
        canonicalizer=TaskCanonicalizer(),
        consistency_gate=ConsistencyGate(),
    )

    result = pipeline.run(
        None,
        plan=plan,
        original_query="Tẩy trắng răng giá bao nhiêu và có đau không?",
        history={"state": {}},
        span_result=SpanExtractionResult(
            query="Tẩy trắng răng giá bao nhiêu và có đau không?",
            spans=[span],
        ),
    )

    faq = result.bound_plan.tasks[1]
    assert faq.entity_names == (service,)
    assert faq.resolved_ids == ("service-whitening",)
    assert faq.inherited_from_task_id == "t1"
    assert faq.binding_source == BindingSource.SAME_TURN_TASK
    assert faq.effective_query == f"{service}. có đau không?"


def test_product_detail_explicit_entity_resolves_to_authoritative_id():
    product = "Máy tăm nước"
    span = EntitySpan(
        text=product,
        label="product_name",
        score=1.0,
        source="catalog_match",
        metadata={"catalog_name": product},
    )
    plan = TaskPlan(
        tasks=(
            planned_task(
                intent=Intent.PRODUCT_DETAIL,
                query="Máy tăm nước còn hàng không?",
                entities=(product,),
                entity_type="product",
            ),
        )
    )
    pipeline = TaskBindingPipeline(
        binder=ContextBinder(Settings()),
        resolver=_DeterministicResolver(),
        canonicalizer=TaskCanonicalizer(),
        consistency_gate=ConsistencyGate(),
    )

    result = pipeline.run(
        None,
        plan=plan,
        original_query="Máy tăm nước còn hàng không?",
        history={"state": {}},
        span_result=SpanExtractionResult(
            query="Máy tăm nước còn hàng không?",
            spans=[span],
        ),
    )

    task = result.bound_plan.tasks[0]
    assert task.entity_type == "product"
    assert task.entity_names == (product,)
    assert task.resolved_ids == ("product-water-flosser",)
    assert task.filters.product_ids == ("product-water-flosser",)
    assert task.clarification_required is False


def test_multitask_service_faq_and_clinic_binding_is_task_scoped():
    service = "Tẩy trắng răng tại phòng khám"
    span = EntitySpan(
        text="Tẩy trắng răng",
        label="service_name",
        score=1.0,
        source="catalog_match",
        metadata={"catalog_name": service},
    )
    plan = TaskPlan(
        tasks=(
            planned_task(
                query="Tẩy trắng răng giá bao nhiêu",
                entities=("Tẩy trắng răng",),
                task_id="t1",
                priority=1,
            ),
            planned_task(
                intent=Intent.FAQ,
                query="có đau không",
                entities=(),
                entity_type=None,
                task_id="t2",
                priority=2,
            ),
            planned_task(
                intent=Intent.CLINIC_INFO,
                query="cuối tuần mở cửa không",
                entities=(),
                entity_type=None,
                task_id="t3",
                priority=3,
            ),
        )
    )
    pipeline = TaskBindingPipeline(
        binder=ContextBinder(Settings()),
        resolver=_DeterministicResolver(),
        canonicalizer=TaskCanonicalizer(),
        consistency_gate=ConsistencyGate(),
    )

    result = pipeline.run(
        None,
        plan=plan,
        original_query=(
            "Tẩy trắng răng giá bao nhiêu, có đau không và cuối tuần mở cửa không?"
        ),
        history={"state": {}},
        span_result=SpanExtractionResult(
            query=(
                "Tẩy trắng răng giá bao nhiêu, có đau không và cuối tuần mở cửa không?"
            ),
            spans=[span],
        ),
    )

    service_task, faq_task, clinic_task = result.bound_plan.tasks
    assert service_task.entity_names == (service,)
    assert faq_task.entity_names == (service,)
    assert faq_task.inherited_from_task_id == "t1"
    assert clinic_task.entity_names == ()
    assert clinic_task.resolved_ids == ()
    assert clinic_task.reference_mode == ReferenceMode.NO_ENTITY


def test_faq_does_not_inherit_from_failed_same_turn_task():
    plan = TaskPlan(
        tasks=(
            planned_task(
                query="Dịch vụ không tồn tại giá bao nhiêu",
                entities=("Dịch vụ không tồn tại",),
            ),
            planned_task(
                task_id="t2",
                intent=Intent.FAQ,
                query="có đau không?",
                entities=(),
                entity_type=None,
                priority=2,
            ),
        )
    )
    pipeline = TaskBindingPipeline(
        binder=ContextBinder(Settings()),
        resolver=_DeterministicResolver(),
        canonicalizer=TaskCanonicalizer(),
        consistency_gate=ConsistencyGate(),
    )

    result = pipeline.run(
        None,
        plan=plan,
        original_query="Dịch vụ không tồn tại giá bao nhiêu và có đau không?",
        history={"state": {}},
        span_result=SpanExtractionResult(
            query="Dịch vụ không tồn tại giá bao nhiêu và có đau không?"
        ),
    )

    assert result.bound_plan.tasks[0].clarification_required is True
    assert result.bound_plan.tasks[1].entity_names == ()
    assert result.bound_plan.tasks[1].inherited_from_task_id is None


def test_pre_tool_gate_blocks_internal_task_mismatch():
    task = bound_task()
    inconsistent = task.model_copy(
        update={
            "effective_query": "Mặt dán sứ Veneer mất bao lâu",
            "filters": CanonicalFilters(
                service_ids=("wrong-id",),
                service_names=task.entity_names,
            ),
        }
    )

    report = ConsistencyGate().check_bound_plan(
        BoundTaskPlan(tasks=(inconsistent,))
    )

    assert report.status == GateStatus.BLOCK
    assert {
        violation.code for violation in report.violations
    } == {"filter_id_mismatch"}


def test_pre_tool_gate_blocks_entity_on_clinic_info():
    invalid = bound_task(
        intent=Intent.CLINIC_INFO,
        entity_name="Tẩy trắng răng",
        entity_id="service-1",
    )
    report = ConsistencyGate().check_bound_plan(
        BoundTaskPlan(tasks=(invalid,))
    )

    assert report.status == GateStatus.BLOCK
    assert report.violations[0].code == "entity_blocked_for_intent"


def test_post_evidence_gate_rejects_wrong_authoritative_row():
    task = bound_task(entity_id="service-correct")
    evidence = [
        EvidenceItem(
            task_id="t1",
            source_type="service",
            source_id="service-wrong",
            text="Dịch vụ khác",
            trust_level="authoritative",
        )
    ]

    report = ConsistencyGate().check_evidence(
        plan=BoundTaskPlan(tasks=(task,)),
        evidence=evidence,
    )

    assert report.status == GateStatus.BLOCK
    assert report.violations[0].code == "evidence_resolved_id_mismatch"


def test_post_evidence_gate_passes_zero_evidence_capability_without_intent_set():
    task = BoundTask(
        task_id="t1",
        intent=Intent.CHITCHAT,
        planner_query="Cảm ơn",
        effective_query="Cảm ơn",
        reference_mode=ReferenceMode.NO_ENTITY,
        resolution_status="not_applicable",
    )

    report = ConsistencyGate().check_evidence(
        plan=BoundTaskPlan(tasks=(task,)),
        evidence=[],
    )

    assert report.status == GateStatus.PASS
    assert report.valid_task_ids == ("t1",)


def test_synthesis_payload_excludes_planner_and_global_metadata():
    task = bound_task()
    pack = EvidencePack(
        query="Mất bao lâu?",
        tasks=(task,),
        items=[
            EvidenceItem(
                task_id="t1",
                source_type="service",
                source_id="service-1",
                text="Tẩy trắng răng mất 90 phút",
                trust_level="authoritative",
                raw_json={
                    "name": "Tẩy trắng răng tại phòng khám",
                    "duration_minutes": 90,
                },
            )
        ],
    )

    payload = pack.to_prompt_payload()
    serialized = str(payload)

    assert payload["tasks"][0]["effective_query"] == task.effective_query
    assert "planner_query" not in serialized
    assert "planner_global_entities" not in serialized
    assert "resolved_ids" not in serialized


def test_tool_executor_uses_registry_policy_and_bound_resolved_ids():
    product_ids = ("product-1", "product-2")

    class _Structured:
        def list_products(self, _session, specification):
            assert specification.product_ids == product_ids
            return list(product_ids)

        def _product_result(self, _session, item, score):
            return RetrievalResult(
                source_type="product",
                source_id=item,
                text=f"Sản phẩm {item}",
                score=score,
                raw_json={"name": item},
                canonical_key=f"product:{item}",
            )

    task = BoundTask(
        task_id="t1",
        intent=Intent.PRODUCT_COMPARE,
        planner_query="untrusted",
        effective_query="So sánh Product 1 và Product 2",
        entity_type="product",
        entity_names=("Product 1", "Product 2"),
        resolved_ids=product_ids,
        filters=CanonicalFilters(
            product_ids=product_ids,
            product_names=("Product 1", "Product 2"),
        ),
        binding_source=BindingSource.EXPLICIT_SPAN,
        reference_mode=ReferenceMode.COMPARE,
        resolution_status="resolved",
    )
    executor = ToolExecutor(
        structured=_Structured(),
        dense=None,
        sparse=None,
        reranker=None,
        settings=Settings(),
    )

    result = executor.execute_many(
        None,
        BoundTaskPlan(tasks=(task,)),
    )

    assert result.tool_counts == {"product_tool": 2}
    assert [item.source_id for item in result.evidence] == list(product_ids)


def test_tool_executor_unknown_tool_raises_clear_error():
    executor = ToolExecutor(
        structured=None,
        dense=None,
        sparse=None,
        reranker=None,
        settings=Settings(),
    )

    with pytest.raises(ValueError, match="not registered in ToolExecutor"):
        executor._execute_tool(None, bound_task(), "unknown_tool")


def test_tool_executor_register_tool_dispatches_runtime_handler():
    executor = ToolExecutor(
        structured=None,
        dense=None,
        sparse=None,
        reranker=None,
        settings=Settings(),
    )
    evidence_item = EvidenceItem(
        task_id="t1",
        source_type="extension",
        source_id="extension-1",
        text="Extension evidence",
    )

    def handler(_session, task, tool_name):
        assert task.task_id == "t1"
        assert tool_name == "custom_tool"
        return [evidence_item], {"reranked": False}

    executor.register_tool("custom_tool", handler)
    evidence, meta = executor._execute_tool(None, bound_task(), "custom_tool")

    assert evidence == [evidence_item]
    assert meta == {"reranked": False}
    with pytest.raises(ValueError, match="already registered"):
        executor.register_tool("custom_tool", handler)


def test_tool_executor_receives_allowed_tools_for_core_retrieval_intents():
    called: list[tuple[str, str]] = []
    executor = ToolExecutor(
        structured=None,
        dense=None,
        sparse=None,
        reranker=None,
        settings=Settings(),
    )

    def handler(_session, task, tool_name):
        called.append((task.intent.value, tool_name))
        return [], None

    for tool in (
        "product_tool",
        "service_tool",
        "clinic_info_tool",
        "faq_tool",
        "document_rag_tool",
    ):
        executor.register_tool(tool, handler, override=True)

    tasks = (
        BoundTask(
            task_id="product-list",
            intent=Intent.PRODUCT_LIST,
            planner_query="products",
            effective_query="products",
            entity_type="product",
            reference_mode=ReferenceMode.NO_ENTITY,
            binding_source=BindingSource.TASK_FILTERS,
            resolution_status="not_applicable",
        ),
        BoundTask(
            task_id="service-list",
            intent=Intent.SERVICE_LIST,
            planner_query="services",
            effective_query="services",
            entity_type="service",
            reference_mode=ReferenceMode.NO_ENTITY,
            binding_source=BindingSource.TASK_FILTERS,
            resolution_status="not_applicable",
        ),
        BoundTask(
            task_id="clinic",
            intent=Intent.CLINIC_INFO,
            planner_query="hours",
            effective_query="hours",
            reference_mode=ReferenceMode.NO_ENTITY,
            resolution_status="not_applicable",
        ),
        BoundTask(
            task_id="faq",
            intent=Intent.FAQ,
            planner_query="pain",
            effective_query="pain",
            reference_mode=ReferenceMode.NO_ENTITY,
            resolution_status="not_applicable",
        ),
    )

    result = executor.execute_many(None, BoundTaskPlan(tasks=tasks))

    assert result.errors == []
    assert ("PRODUCT_LIST", "product_tool") in called
    assert ("SERVICE_LIST", "service_tool") in called
    assert ("CLINIC_INFO", "clinic_info_tool") in called
    assert ("FAQ", "faq_tool") in called
    assert ("FAQ", "document_rag_tool") in called


def test_faq_tool_merges_curated_sparse_and_dense_faq_evidence():
    faq_direct = object()
    sparse_result = RetrievalResult(
        source_type="faq",
        source_id="faq-sparse",
        text="Sparse FAQ",
        score=0.8,
        raw_json={"question": "Sparse?", "answer": "Sparse answer"},
        canonical_key="faq:faq-sparse",
    )
    dense_result = RetrievalResult(
        source_type="faq",
        source_id="faq-dense",
        text="Dense FAQ",
        score=0.7,
        raw_json={"question": "Dense?", "answer": "Dense answer"},
        canonical_key="faq:faq-dense",
    )

    class _Structured:
        def search_faqs(self, _session, query, *, limit):
            assert query == "Bound FAQ query"
            assert limit > 0
            return [(faq_direct, 0.95)]

        def _faq_result(self, item, score):
            assert item is faq_direct
            return RetrievalResult(
                source_type="faq",
                source_id="faq-direct",
                text="Curated FAQ",
                score=score,
                raw_json={"question": "Direct?", "answer": "Direct answer"},
                canonical_key="faq:faq-direct",
            )

    class _Sparse:
        def retrieve_faqs(self, _session, query):
            assert query == "Bound FAQ query"
            return [sparse_result]

    class _Dense:
        def retrieve_faqs(self, _session, query):
            assert query == "Bound FAQ query"
            return [dense_result]

    executor = ToolExecutor(
        structured=_Structured(),
        dense=_Dense(),
        sparse=_Sparse(),
        reranker=None,
        settings=Settings(final_top_k=5),
    )
    task = BoundTask(
        task_id="t1",
        intent=Intent.FAQ,
        planner_query="planner FAQ query",
        effective_query="Bound FAQ query",
        reference_mode=ReferenceMode.NO_ENTITY,
        resolution_status="not_applicable",
    )

    evidence, meta = executor._handle_faq_tool(None, task, "faq_tool")

    assert meta is None
    assert {item.source_id for item in evidence} >= {
        "faq-direct",
        "faq-sparse",
        "faq-dense",
    }
    assert all(item.trust_level == "curated" for item in evidence)


def test_document_rag_uses_bound_effective_query_without_hyde_authority():
    seen_queries: list[str] = []

    class _Dense:
        def retrieve_by_source(self, _session, query, intent):
            seen_queries.append(f"dense:{query}:{intent.value}")
            return {
                "chunk": [
                    RetrievalResult(
                        source_type="chunk",
                        source_id="chunk-1",
                        text="Chunk evidence",
                        score=0.8,
                        canonical_key="chunk:chunk-1",
                    )
                ]
            }

    class _Sparse:
        def retrieve_by_source(self, _session, query, intent):
            seen_queries.append(f"sparse:{query}:{intent.value}")
            return {}

    class _Reranker:
        def rerank(self, query, results):
            assert query == "Bound effective FAQ query"
            return results, False, {"reranked": False}

    executor = ToolExecutor(
        structured=None,
        dense=_Dense(),
        sparse=_Sparse(),
        reranker=_Reranker(),
        settings=Settings(enable_hyde=True),
    )
    task = BoundTask(
        task_id="t1",
        intent=Intent.FAQ,
        planner_query="Untrusted planner query",
        effective_query="Bound effective FAQ query",
        reference_mode=ReferenceMode.NO_ENTITY,
        resolution_status="not_applicable",
    )

    evidence, meta = executor._document_rag_evidence(None, task)

    assert [item.source_id for item in evidence] == ["chunk-1"]
    assert meta["reranked"] is False
    assert seen_queries == [
        "dense:Bound effective FAQ query:FAQ",
        "sparse:Bound effective FAQ query:FAQ",
    ]
    assert task.effective_query == "Bound effective FAQ query"
    assert task.planner_query == "Untrusted planner query"


def test_evidence_merger_uses_bound_tasks_and_authoritative_evidence():
    task = bound_task()
    pack = EvidenceMerger(Settings(max_evidence_items=5)).merge(
        query=task.effective_query,
        plan=BoundTaskPlan(tasks=(task,)),
        evidence=[
            EvidenceItem(
                task_id="t1",
                source_type="chunk",
                source_id="chunk-1",
                text="Giá cũ",
                score=0.9,
                trust_level="retrieved",
                canonical_key="service:service-1",
            ),
            EvidenceItem(
                task_id="t1",
                source_type="service",
                source_id="service-1",
                text="Dịch vụ tẩy trắng",
                score=1.0,
                trust_level="authoritative",
                canonical_key="service:service-1",
                raw_json={
                    "name": "Tẩy trắng răng tại phòng khám"
                },
            ),
        ],
    )

    assert len(pack.items) == 1
    assert pack.items[0].source_type == "service"
    assert pack.missing_info == []


def test_memory_persists_only_passed_resolved_memory_eligible_tasks():
    detail = bound_task(
        intent=Intent.PRODUCT_DETAIL,
        entity_name="AquaJet Mini Water Flosser",
        entity_id="product-1",
        entity_type="product",
        effective_query="AquaJet Mini Water Flosser. Còn hàng không?",
    )
    product_list = BoundTask(
        task_id="t2",
        intent=Intent.PRODUCT_LIST,
        planner_query="Sản phẩm dưới 2 triệu",
        effective_query="Sản phẩm dưới 2 triệu",
        entity_type="product",
        filters=CanonicalFilters(price_max=2_000_000),
        reference_mode=ReferenceMode.NO_ENTITY,
        resolution_status="not_applicable",
    )
    plan = BoundTaskPlan(tasks=(detail, product_list))
    pack = EvidencePack(query="query", tasks=plan.tasks, items=[])

    state = ChatService._build_conversation_state(
        {"state": {}},
        plan,
        pack,
        passed_task_ids=("t1", "t2"),
    )

    assert state["active_product_ids"] == ["product-1"]
    assert state["active_product_names"] == [
        "AquaJet Mini Water Flosser"
    ]
    assert state["last_filters"]["product"]["price_max"] == 2_000_000


def test_semantic_validator_rejects_answer_about_wrong_entity():
    context = {
        "tasks": [
            {
                "task_id": "t1",
                "intent": "SERVICE_DETAIL",
                "entity_names": ["Tẩy trắng răng tại phòng khám"],
            }
        ],
        "items": [
            {
                "source_type": "service",
                "source_id": "service-1",
                "text": "Tẩy trắng răng tại phòng khám mất 90 phút.",
                "raw_json": {
                    "name": "Tẩy trắng răng tại phòng khám",
                    "duration_minutes": 90,
                },
                "source": {"task_id": "t1"},
            }
        ],
    }
    payload = {
        "intent": "SERVICE_DETAIL",
        "confidence": 0.9,
        "answer_type": "rag",
        "entities": [],
        "result": {
            "text": "Mặt dán sứ Veneer mất 90 phút.",
            "items": [],
            "assets": [],
            "sources": [],
            "missing_assets": [],
        },
        "safety": {},
    }

    with pytest.raises(
        ResponseValidationError,
        match="canonical task entity",
    ):
        ResponseValidator().validate(payload, context=context)


def test_planner_object_entities_remain_untrusted_diagnostic_values():
    payload = TaskPlanner._parse_llm_output(
        """
        {
          "tasks": [{
            "task_id": "t1",
            "intent": "PRODUCT_LIST",
            "query": "Sản phẩm nào dưới 2 triệu",
            "entities": {"price_range": ["dưới 2 triệu"]}
          }],
          "global_entities": {"price_range": ["dưới 2 triệu"]}
        }
        """
    )

    assert payload.tasks[0].planner_entities == ("dưới 2 triệu",)
    assert payload.global_entities == ["dưới 2 triệu"]
