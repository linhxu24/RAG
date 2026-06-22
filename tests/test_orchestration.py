import asyncio
import uuid

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
    assert capability_for(Intent.FAQ).allowed_tools == (
        "faq_tool",
        "document_rag_tool",
    )


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


def test_planner_preserves_llm_usage_when_empty_plan_falls_back():
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

    assert plan.source == "heuristic"
    assert plan.tasks[0].intent == Intent.PRODUCT_LIST
    assert plan.metadata["planner_fallback"] == (
        "heuristic_after_empty_llm_plan"
    )
    assert plan.metadata["usage"] == {
        "prompt_tokens": 321,
        "completion_tokens": 7,
        "total_tokens": 328,
    }


def test_binder_does_not_mutate_planner_and_rejects_stale_follow_up_entity():
    service_id = str(uuid.uuid4())
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
                "active_service_ids": [service_id],
                "active_service_names": ["Tẩy trắng răng tại phòng khám"],
            }
        },
        span_result=SpanExtractionResult(query="Mất bao lâu?"),
    ).decisions[0]

    assert task.planner_query == "Mặt dán sứ Veneer mất bao lâu"
    assert task.planner_entities == ("Mặt dán sứ Veneer",)
    assert decision.entity_names == ("Tẩy trắng răng tại phòng khám",)
    assert decision.inherited_resolved_ids == (service_id,)
    assert decision.rejected_planner_entities == ("Mặt dán sứ Veneer",)
    assert decision.binding_source == BindingSource.CONVERSATION_STATE


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


def test_binder_mixed_compare_combines_memory_and_explicit_entity():
    task = planned_task(
        intent=Intent.PRODUCT_COMPARE,
        query="So sánh nó với EnamelGuard Sensitive Toothpaste",
        entities=("EnamelGuard Sensitive Toothpaste",),
        entity_type="product",
    )
    span = EntitySpan(
        text="EnamelGuard Sensitive Toothpaste",
        label="product_name",
        score=1.0,
        source="catalog_match",
        metadata={"catalog_name": "EnamelGuard Sensitive Toothpaste"},
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
            spans=[span],
        ),
    )

    assert decision.reference_mode == ReferenceMode.MIXED
    assert decision.entity_names == (
        "FreshMint Total Protection Toothpaste",
        "EnamelGuard Sensitive Toothpaste",
    )
    assert decision.inherited_resolved_ids == ("product-memory",)


def test_faq_follow_up_uses_active_memory_domain_and_verified_id():
    task = planned_task(
        intent=Intent.FAQ,
        query="Có đau không?",
        entities=(),
        entity_type=None,
    )
    decision = ContextBinder(Settings()).bind_task(
        task=task,
        original_query="Có đau không?",
        history={
            "state": {
                "active_domain": "service",
                "active_service_names": [
                    "Tẩy trắng răng tại phòng khám"
                ],
                "active_service_ids": ["service-whitening"],
            }
        },
        span_result=SpanExtractionResult(query="Có đau không?"),
    )

    assert decision.entity_type == "service"
    assert decision.entity_names == (
        "Tẩy trắng răng tại phòng khám",
    )
    assert decision.inherited_resolved_ids == ("service-whitening",)
    assert decision.binding_source == BindingSource.CONVERSATION_STATE


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


def test_canonicalizer_atomically_removes_stale_entity_from_effective_state():
    service_id = str(uuid.uuid4())
    task = planned_task(
        query="Mặt dán sứ Veneer mất bao lâu",
        entities=("Mặt dán sứ Veneer",),
    )
    decision = BindingDecision(
        task_id="t1",
        intent=Intent.SERVICE_DETAIL,
        entity_type="service",
        reference_mode=ReferenceMode.IMPLICIT,
        binding_source=BindingSource.CONVERSATION_STATE,
        entity_names=("Tẩy trắng răng tại phòng khám",),
        inherited_resolved_ids=(service_id,),
        rejected_planner_entities=("Mặt dán sứ Veneer",),
    )
    resolution = TaskResolution(
        task_id="t1",
        status="resolved",
        entity_type="service",
        entity_names=("Tẩy trắng răng tại phòng khám",),
        resolved_ids=(service_id,),
        source="conversation_state",
    )

    bound = TaskCanonicalizer().canonicalize(
        task=task,
        decision=decision,
        resolution=resolution,
        history={"state": {}},
    )

    assert "Mặt dán sứ Veneer" not in bound.effective_query
    assert "Tẩy trắng răng tại phòng khám" in bound.effective_query
    assert bound.entity_names == ("Tẩy trắng răng tại phòng khám",)
    assert bound.resolved_ids == (service_id,)
    assert bound.filters.service_ids == (service_id,)
    assert bound.filters.service_names == (
        "Tẩy trắng răng tại phòng khám",
    )


class _DeterministicResolver:
    def __init__(self):
        self.ids = {
            "Tẩy trắng răng tại phòng khám": "service-whitening",
            "Mặt dán sứ Veneer": "service-veneer",
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
    } >= {
        "effective_query_missing_entity",
        "filter_id_mismatch",
    }


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
