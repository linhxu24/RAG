from app.constants import Intent
from app.orchestration.schemas import (
    BindingSource,
    BoundTask,
    BoundTaskPlan,
    CanonicalFilters,
    EvidenceItem,
    EvidencePack,
    ReferenceMode,
)
from app.orchestration.suggestions import ContextualSuggestionEngine


def test_product_list_generates_evidence_backed_next_questions():
    task = BoundTask(
        task_id="t1",
        intent=Intent.PRODUCT_LIST,
        planner_query="Cho tôi danh sách sản phẩm đang có",
        effective_query="Cho tôi danh sách sản phẩm đang có",
        entity_type="product",
        filters=CanonicalFilters(limit=10),
        binding_source=BindingSource.TASK_FILTERS,
        reference_mode=ReferenceMode.NO_ENTITY,
        resolution_status="not_applicable",
        operation="list",
    )
    plan = BoundTaskPlan(tasks=(task,))
    evidence = EvidencePack(
        query=task.effective_query,
        tasks=plan.tasks,
        items=[
            EvidenceItem(
                task_id="t1",
                source_type="product",
                source_id="product-1",
                text="Sản phẩm OralWave Pro S2",
                trust_level="authoritative",
                raw_json={
                    "name": "OralWave Pro S2",
                    "price": 1_350_000,
                    "quantity": 12,
                },
            )
        ],
    )
    engine = ContextualSuggestionEngine(max_suggestions=3)
    conversation_state = {
        "last_intents": ["PRODUCT_LIST"],
        "active_product_ids": [],
        "active_service_ids": [],
        "suggestion_state": {},
    }

    interest = engine.build_interest_state(
        original_query=task.effective_query,
        history={"state": {}},
        conversation_state=conversation_state,
        plan=plan,
        evidence_pack=evidence,
        valid_task_ids=("t1",),
    )
    candidates = engine.generate_candidates(
        original_query=task.effective_query,
        plan=plan,
        evidence_pack=evidence,
        interest_state=interest,
        valid_task_ids=("t1",),
    )
    selection = engine.rank_and_gate(
        candidates=candidates,
        plan=plan,
        evidence_pack=evidence,
        conversation_state=conversation_state,
        interest_state=interest,
    )

    assert len(selection.accepted) == 3
    assert {item.target_intent for item in selection.accepted} <= {
        Intent.PRODUCT_LIST,
        Intent.PRODUCT_DETAIL,
    }
    assert any(
        item.type.value == "recommendation"
        and item.resolved_ids == ("product-1",)
        for item in selection.accepted
    )


def test_service_detail_uses_canonical_entity_and_skips_answered_fields():
    task = BoundTask(
        task_id="t1",
        intent=Intent.SERVICE_DETAIL,
        planner_query="Tẩy trắng răng giá bao nhiêu?",
        effective_query="Tẩy trắng răng tại phòng khám giá bao nhiêu?",
        entity_type="service",
        entity_names=("Tẩy trắng răng tại phòng khám",),
        resolved_ids=("service-1",),
        filters=CanonicalFilters(
            service_ids=("service-1",),
            service_names=("Tẩy trắng răng tại phòng khám",),
        ),
        binding_source=BindingSource.EXPLICIT_SPAN,
        reference_mode=ReferenceMode.EXPLICIT,
        resolution_status="resolved",
        operation="price_lookup",
    )
    plan = BoundTaskPlan(tasks=(task,))
    evidence = EvidencePack(
        query=task.effective_query,
        tasks=plan.tasks,
        items=[
            EvidenceItem(
                task_id="t1",
                source_type="service",
                source_id="service-1",
                text="Tẩy trắng răng, giá và thời lượng",
                trust_level="authoritative",
                raw_json={
                    "name": "Tẩy trắng răng tại phòng khám",
                    "price": 2_000_000,
                    "duration_minutes": 60,
                },
            )
        ],
    )
    engine = ContextualSuggestionEngine(max_suggestions=3)
    conversation_state = {
        "last_intents": ["SERVICE_DETAIL"],
        "active_service_ids": ["service-1"],
        "active_product_ids": [],
        "suggestion_state": {},
    }

    interest = engine.build_interest_state(
        original_query=task.effective_query,
        history={"state": {}},
        conversation_state=conversation_state,
        plan=plan,
        evidence_pack=evidence,
        valid_task_ids=("t1",),
    )
    selection = engine.rank_and_gate(
        candidates=engine.generate_candidates(
            original_query=task.effective_query,
            plan=plan,
            evidence_pack=evidence,
            interest_state=interest,
            valid_task_ids=("t1",),
        ),
        plan=plan,
        evidence_pack=evidence,
        conversation_state=conversation_state,
        interest_state=interest,
    )

    assert {"price", "duration"} <= set(interest.answered_topics)
    assert [item.topic for item in selection.accepted] == [
        "pain",
        "safety",
        "clinic_hours",
    ]
    entity_questions = [
        item for item in selection.accepted if item.entity_names
    ]
    assert all(item.resolved_ids == ("service-1",) for item in entity_questions)
    assert all(
        "Tẩy trắng răng tại phòng khám" in item.query
        for item in entity_questions
    )


def test_recent_suggestions_are_not_repeated_and_click_is_persisted():
    task = BoundTask(
        task_id="t1",
        intent=Intent.GREETING,
        planner_query="Xin chào",
        effective_query="Xin chào",
        binding_source=BindingSource.NONE,
        reference_mode=ReferenceMode.NO_ENTITY,
    )
    plan = BoundTaskPlan(tasks=(task,))
    evidence = EvidencePack(query="Xin chào", tasks=plan.tasks, items=[])
    engine = ContextualSuggestionEngine(max_suggestions=3, history_limit=6)
    state = {
        "last_intents": ["GREETING"],
        "active_product_ids": [],
        "active_service_ids": [],
        "suggestion_state": {},
    }
    interest = engine.build_interest_state(
        original_query="Xin chào",
        history={"state": {}},
        conversation_state=state,
        plan=plan,
        evidence_pack=evidence,
        valid_task_ids=("t1",),
    )
    candidates = engine.generate_candidates(
        original_query="Xin chào",
        plan=plan,
        evidence_pack=evidence,
        interest_state=interest,
        valid_task_ids=("t1",),
    )
    first = engine.rank_and_gate(
        candidates=candidates,
        plan=plan,
        evidence_pack=evidence,
        conversation_state=state,
        interest_state=interest,
    )
    state["suggestion_state"] = engine.update_suggestion_state(
        conversation_state=state,
        selection=first,
        selected_suggestion_id=None,
    )
    state["suggestion_state"] = engine.update_suggestion_state(
        conversation_state=state,
        selection=first,
        selected_suggestion_id=first.accepted[0].suggestion_id,
    )
    second = engine.rank_and_gate(
        candidates=candidates,
        plan=plan,
        evidence_pack=evidence,
        conversation_state=state,
        interest_state=interest,
    )

    assert first.accepted
    assert second.accepted == ()
    assert first.accepted[0].suggestion_id in state[
        "suggestion_state"
    ]["accepted_suggestion_ids"]
    assert {
        rejection.reason for rejection in second.rejected
    } == {"recently_shown"}


def test_interest_topics_follow_same_entity_across_intent_changes():
    task = BoundTask(
        task_id="t1",
        intent=Intent.FAQ,
        planner_query="Có đau không?",
        effective_query="Tẩy trắng răng tại phòng khám có đau không?",
        entity_type="service",
        entity_names=("Tẩy trắng răng tại phòng khám",),
        resolved_ids=("service-1",),
        binding_source=BindingSource.CONVERSATION_STATE,
        reference_mode=ReferenceMode.IMPLICIT,
        resolution_status="resolved",
    )
    plan = BoundTaskPlan(tasks=(task,))
    evidence = EvidencePack(
        query=task.effective_query,
        tasks=plan.tasks,
        items=[
            EvidenceItem(
                task_id="t1",
                source_type="faq",
                source_id="faq-1",
                text="Thông tin đau sau tẩy trắng",
                trust_level="curated",
            )
        ],
    )
    engine = ContextualSuggestionEngine()
    previous_interest = {
        "active_domain": "service",
        "context_signature": "service:service-1",
        "answered_topics": ["price", "duration"],
    }

    interest = engine.build_interest_state(
        original_query="Có đau không?",
        history={"state": {"interest_state": previous_interest}},
        conversation_state={
            "last_intents": ["FAQ", "SERVICE_DETAIL"],
            "active_service_ids": ["service-1"],
        },
        plan=plan,
        evidence_pack=evidence,
        valid_task_ids=("t1",),
    )

    assert interest.active_domain == "service"
    assert interest.context_signature == "service:service-1"
    assert {"price", "duration", "pain"} <= set(
        interest.answered_topics
    )
