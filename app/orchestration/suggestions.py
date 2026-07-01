from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.constants import Intent
from app.generation.schemas import ChatSuggestion
from app.orchestration.intent_registry import EntityScope, capability_for, domain_for_intent
from app.orchestration.schemas import BoundTaskPlan, EvidencePack
from app.retrieval.normalization import normalize_vietnamese


class JourneyStage(StrEnum):
    DISCOVERY = "discovery"
    CONSIDERATION = "consideration"
    COMPARISON = "comparison"
    DECISION = "decision"
    SUPPORT = "support"


class SuggestionType(StrEnum):
    NEXT_QUESTION = "next_question"
    RECOMMENDATION = "recommendation"


class InterestEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    name: str
    resolved_id: str
    confidence: float = 1.0


class ConversationInterestState(BaseModel):
    model_config = ConfigDict(frozen=True)

    active_domain: str | None = None
    active_entities: tuple[InterestEntity, ...] = ()
    goals: tuple[str, ...] = ()
    journey_stage: JourneyStage = JourneyStage.DISCOVERY
    answered_topics: tuple[str, ...] = ()
    unresolved_topics: tuple[str, ...] = ()
    recent_intents: tuple[str, ...] = ()
    context_signature: str = "none"


class SuggestionCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestion_id: str
    type: SuggestionType
    label: str
    query: str
    target_intent: Intent
    topic: str
    reason_code: str
    source_task_id: str | None = None
    entity_names: tuple[str, ...] = ()
    resolved_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    base_score: float = 0.5
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SuggestionRejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestion_id: str
    reason: str


class SuggestionSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted: tuple[SuggestionCandidate, ...] = ()
    rejected: tuple[SuggestionRejection, ...] = ()


class ContextualSuggestionEngine:
    """Deterministic next-best-question engine over canonical runtime state."""

    def __init__(self, *, max_suggestions: int = 3, history_limit: int = 12):
        self.max_suggestions = max(0, max_suggestions)
        self.history_limit = max(1, history_limit)

    def build_interest_state(
        self,
        *,
        original_query: str,
        history: dict[str, Any],
        conversation_state: dict[str, Any],
        plan: BoundTaskPlan,
        evidence_pack: EvidencePack,
        valid_task_ids: tuple[str, ...],
    ) -> ConversationInterestState:
        allowed = set(valid_task_ids)
        tasks = [
            task
            for task in plan.tasks
            if task.task_id in allowed and not task.clarification_required
        ]
        primary = tasks[0] if tasks else None
        domain = _domain_for_intent(primary.intent) if primary else None
        entities = tuple(
            InterestEntity(
                type=task.entity_type or "unknown",
                name=name,
                resolved_id=resolved_id,
            )
            for task in tasks
            for name, resolved_id in zip(
                task.entity_names,
                task.resolved_ids,
                strict=False,
            )
        )
        if entities and entities[0].type in {"product", "service"}:
            domain = entities[0].type
        signature = _context_signature(domain, entities)
        previous = _record(
            _record(history.get("state")).get("interest_state")
        )
        previous_topics = (
            _strings(previous.get("answered_topics"))
            if previous.get("context_signature") == signature
            else []
        )
        current_topics = _detect_topics(original_query)
        current_topics.update(
            topic
            for task in tasks
            for topic in _detect_topics(task.effective_query)
        )
        if primary and primary.intent in {
            Intent.PRODUCT_LIST,
            Intent.SERVICE_LIST,
        }:
            current_topics.add("catalog")
        if primary and primary.intent in {
            Intent.PRODUCT_DETAIL,
            Intent.PRODUCT_COMPARE,
            Intent.SERVICE_DETAIL,
        }:
            current_topics.update(_evidence_topics(primary.intent, evidence_pack))
        answered_topics = tuple(
            dict.fromkeys([*previous_topics, *sorted(current_topics)])
        )
        policy_topics = (
            capability_for(primary.intent).suggestion_policy.topic_sequence
            if primary
            else ()
        )
        unresolved_topics = tuple(
            topic for topic in policy_topics if topic not in answered_topics
        )
        recent_intents = tuple(
            dict.fromkeys(
                [
                    *(task.intent.value for task in tasks),
                    *_strings(conversation_state.get("last_intents")),
                ]
            )
        )[:8]
        return ConversationInterestState(
            active_domain=domain,
            active_entities=entities,
            goals=tuple(sorted(_goals_for_topics(set(answered_topics)))),
            journey_stage=_journey_stage(primary.intent if primary else Intent.UNKNOWN),
            answered_topics=answered_topics,
            unresolved_topics=unresolved_topics,
            recent_intents=recent_intents,
            context_signature=signature,
        )

    def generate_candidates(
        self,
        *,
        original_query: str,
        plan: BoundTaskPlan,
        evidence_pack: EvidencePack,
        interest_state: ConversationInterestState,
        valid_task_ids: tuple[str, ...],
    ) -> tuple[SuggestionCandidate, ...]:
        allowed = set(valid_task_ids)
        tasks = [
            task
            for task in plan.tasks
            if task.task_id in allowed and not task.clarification_required
        ]
        if not tasks:
            return ()
        primary = tasks[0]
        evidence = [
            item
            for item in evidence_pack.items
            if item.task_id == primary.task_id
        ]
        evidence_ids = tuple(item.source_id for item in evidence)
        active = interest_state.active_entities
        candidates: list[SuggestionCandidate] = []

        def add(
            *,
            label: str,
            query: str,
            target_intent: Intent,
            topic: str,
            reason_code: str,
            suggestion_type: SuggestionType = SuggestionType.NEXT_QUESTION,
            entity_names: tuple[str, ...] = (),
            resolved_ids: tuple[str, ...] = (),
            source_evidence_ids: tuple[str, ...] = evidence_ids,
            base_score: float = 0.5,
        ) -> None:
            candidates.append(
                SuggestionCandidate(
                    suggestion_id=_suggestion_id(
                        target_intent,
                        query,
                        reason_code,
                    ),
                    type=suggestion_type,
                    label=label,
                    query=query,
                    target_intent=target_intent,
                    topic=topic,
                    reason_code=reason_code,
                    source_task_id=primary.task_id,
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    evidence_ids=source_evidence_ids,
                    base_score=base_score,
                )
            )

        if primary.intent in {Intent.GREETING, Intent.CHITCHAT, Intent.UNKNOWN}:
            add(
                label="Xem danh sách dịch vụ",
                query="Cho tôi danh sách dịch vụ đang có",
                target_intent=Intent.SERVICE_LIST,
                topic="service_catalog",
                reason_code="start_service_discovery",
                source_evidence_ids=(),
                base_score=0.82,
            )
            add(
                label="Xem danh sách sản phẩm",
                query="Cho tôi danh sách sản phẩm đang có",
                target_intent=Intent.PRODUCT_LIST,
                topic="product_catalog",
                reason_code="start_product_discovery",
                source_evidence_ids=(),
                base_score=0.8,
            )
            add(
                label="Xem địa chỉ phòng khám",
                query="Phòng khám ở đâu?",
                target_intent=Intent.CLINIC_INFO,
                topic="clinic_location",
                reason_code="start_clinic_information",
                source_evidence_ids=(),
                base_score=0.7,
            )
        elif primary.intent == Intent.PRODUCT_LIST:
            add(
                label="Chỉ xem sản phẩm còn hàng",
                query="Sản phẩm nào còn hàng?",
                target_intent=Intent.PRODUCT_LIST,
                topic="availability",
                reason_code="refine_product_availability",
                base_score=0.82,
            )
            add(
                label="Sắp xếp theo giá thấp đến cao",
                query="Sắp xếp sản phẩm theo giá từ thấp đến cao",
                target_intent=Intent.PRODUCT_LIST,
                topic="price_sort",
                reason_code="refine_product_price_sort",
                base_score=0.78,
            )
            add(
                label="Xem sản phẩm dưới 1 triệu",
                query="Sản phẩm nào dưới 1 triệu?",
                target_intent=Intent.PRODUCT_LIST,
                topic="budget",
                reason_code="refine_product_budget",
                base_score=0.74,
            )
            top = _top_entity(evidence, "product")
            if top:
                add(
                    label=f"Xem chi tiết {top['name']}",
                    query=f"{top['name']} có thông tin chi tiết gì?",
                    target_intent=Intent.PRODUCT_DETAIL,
                    topic="detail",
                    reason_code="top_result_detail",
                    suggestion_type=SuggestionType.RECOMMENDATION,
                    entity_names=(top["name"],),
                    resolved_ids=(top["id"],),
                    source_evidence_ids=(top["id"],),
                    base_score=0.68,
                )
        elif primary.intent == Intent.SERVICE_LIST:
            add(
                label="Sắp xếp theo giá thấp đến cao",
                query="Sắp xếp dịch vụ theo giá từ thấp đến cao",
                target_intent=Intent.SERVICE_LIST,
                topic="price_sort",
                reason_code="refine_service_price_sort",
                base_score=0.8,
            )
            add(
                label="Xem dịch vụ dưới 2 triệu",
                query="Dịch vụ nào dưới 2 triệu?",
                target_intent=Intent.SERVICE_LIST,
                topic="budget",
                reason_code="refine_service_budget",
                base_score=0.76,
            )
            add(
                label="Xem dịch vụ có thời gian ngắn nhất",
                query="Sắp xếp dịch vụ theo thời gian ngắn nhất",
                target_intent=Intent.SERVICE_LIST,
                topic="duration_sort",
                reason_code="refine_service_duration",
                base_score=0.72,
            )
            top = _top_entity(evidence, "service")
            if top:
                add(
                    label=f"Xem chi tiết {top['name']}",
                    query=f"{top['name']} có thông tin chi tiết gì?",
                    target_intent=Intent.SERVICE_DETAIL,
                    topic="detail",
                    reason_code="top_result_detail",
                    suggestion_type=SuggestionType.RECOMMENDATION,
                    entity_names=(top["name"],),
                    resolved_ids=(top["id"],),
                    source_evidence_ids=(top["id"],),
                    base_score=0.68,
                )
        elif active:
            entity = active[0]
            entity_names = (entity.name,)
            resolved_ids = (entity.resolved_id,)
            entity_evidence = tuple(
                item.source_id
                for item in evidence_pack.items
                if item.source_id == entity.resolved_id
            ) or evidence_ids
            if entity.type == "product":
                add(
                    label="Sản phẩm này phù hợp với ai?",
                    query=f"{entity.name} phù hợp với ai?",
                    target_intent=Intent.FAQ,
                    topic="usage",
                    reason_code="unanswered_product_usage",
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    source_evidence_ids=entity_evidence,
                    base_score=0.84,
                )
                add(
                    label="Có lưu ý an toàn nào không?",
                    query=f"{entity.name} có lưu ý an toàn nào không?",
                    target_intent=Intent.FAQ,
                    topic="safety",
                    reason_code="unanswered_product_safety",
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    source_evidence_ids=entity_evidence,
                    base_score=0.8,
                )
                add(
                    label="Kiểm tra sản phẩm còn hàng",
                    query=f"{entity.name} còn hàng không?",
                    target_intent=Intent.PRODUCT_DETAIL,
                    topic="availability",
                    reason_code="unanswered_product_availability",
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    source_evidence_ids=entity_evidence,
                    base_score=0.76,
                )
            elif entity.type == "service":
                add(
                    label="Dịch vụ này có đau không?",
                    query=f"{entity.name} có đau không?",
                    target_intent=Intent.FAQ,
                    topic="pain",
                    reason_code="unanswered_service_pain",
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    source_evidence_ids=entity_evidence,
                    base_score=0.86,
                )
                add(
                    label="Có chống chỉ định hoặc lưu ý gì?",
                    query=f"{entity.name} có chống chỉ định hoặc lưu ý gì?",
                    target_intent=Intent.FAQ,
                    topic="safety",
                    reason_code="unanswered_service_safety",
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    source_evidence_ids=entity_evidence,
                    base_score=0.82,
                )
                add(
                    label="Dịch vụ này mất bao lâu?",
                    query=f"{entity.name} mất bao lâu?",
                    target_intent=Intent.SERVICE_DETAIL,
                    topic="duration",
                    reason_code="unanswered_service_duration",
                    entity_names=entity_names,
                    resolved_ids=resolved_ids,
                    source_evidence_ids=entity_evidence,
                    base_score=0.78,
                )
            add(
                label="Xem giờ làm việc của phòng khám",
                query="Phòng khám mở cửa lúc mấy giờ?",
                target_intent=Intent.CLINIC_INFO,
                topic="clinic_hours",
                reason_code="move_toward_contact",
                source_evidence_ids=entity_evidence,
                base_score=0.62,
            )
        elif primary.intent == Intent.CLINIC_INFO:
            add(
                label="Xem giờ làm việc",
                query="Phòng khám mở cửa lúc mấy giờ?",
                target_intent=Intent.CLINIC_INFO,
                topic="clinic_hours",
                reason_code="unanswered_clinic_hours",
                base_score=0.82,
            )
            add(
                label="Xem số điện thoại",
                query="Số điện thoại phòng khám là gì?",
                target_intent=Intent.CLINIC_INFO,
                topic="clinic_contact",
                reason_code="unanswered_clinic_contact",
                base_score=0.78,
            )
            add(
                label="Xem địa chỉ",
                query="Phòng khám ở đâu?",
                target_intent=Intent.CLINIC_INFO,
                topic="clinic_location",
                reason_code="unanswered_clinic_location",
                base_score=0.76,
            )

        return tuple(
            candidate
            for candidate in candidates
            if normalize_vietnamese(candidate.query)
            != normalize_vietnamese(original_query)
        )

    def rank_and_gate(
        self,
        *,
        candidates: tuple[SuggestionCandidate, ...],
        plan: BoundTaskPlan,
        evidence_pack: EvidencePack,
        conversation_state: dict[str, Any],
        interest_state: ConversationInterestState,
    ) -> SuggestionSelection:
        if not candidates or not plan.tasks:
            return SuggestionSelection()
        source_policy = capability_for(
            plan.primary_intent
        ).suggestion_policy
        suggestion_state = _record(conversation_state.get("suggestion_state"))
        recent = set(_strings(suggestion_state.get("recent_impressions")))
        trusted_ids = {
            *(
                item.source_id
                for item in evidence_pack.items
                if item.trust_level in {"authoritative", "curated"}
            ),
            *_strings(conversation_state.get("active_product_ids")),
            *_strings(conversation_state.get("active_service_ids")),
        }
        evidence_ids = {item.source_id for item in evidence_pack.items}
        rejected: list[SuggestionRejection] = []
        ranked: list[SuggestionCandidate] = []
        answered = set(interest_state.answered_topics)
        for candidate in candidates:
            if candidate.suggestion_id in recent:
                rejected.append(
                    SuggestionRejection(
                        suggestion_id=candidate.suggestion_id,
                        reason="recently_shown",
                    )
                )
                continue
            reason = _gate_reason(
                candidate,
                source_intent=plan.primary_intent,
                source_policy=source_policy,
                trusted_ids=trusted_ids,
                evidence_ids=evidence_ids,
            )
            if reason:
                rejected.append(
                    SuggestionRejection(
                        suggestion_id=candidate.suggestion_id,
                        reason=reason,
                    )
                )
                continue
            breakdown = {
                "base": candidate.base_score,
                "unanswered_topic": (
                    0.2 if candidate.topic not in answered else -0.45
                ),
                "entity_relevance": 0.15 if candidate.entity_names else 0.0,
                "evidence_support": 0.1 if candidate.evidence_ids else 0.0,
                "journey_progression": (
                    0.08
                    if candidate.type == SuggestionType.RECOMMENDATION
                    else 0.04
                ),
                "repetition": 0.0,
            }
            score = round(sum(breakdown.values()), 6)
            if score <= 0:
                rejected.append(
                    SuggestionRejection(
                        suggestion_id=candidate.suggestion_id,
                        reason="low_score_or_repeated",
                    )
                )
                continue
            ranked.append(
                candidate.model_copy(
                    update={
                        "score": score,
                        "score_breakdown": breakdown,
                    }
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.suggestion_id))
        accepted: list[SuggestionCandidate] = []
        topics: set[str] = set()
        limit = min(self.max_suggestions, source_policy.max_suggestions)
        for candidate in ranked:
            if candidate.topic in topics:
                rejected.append(
                    SuggestionRejection(
                        suggestion_id=candidate.suggestion_id,
                        reason="topic_diversity",
                    )
                )
                continue
            accepted.append(candidate)
            topics.add(candidate.topic)
            if len(accepted) >= limit:
                break
        return SuggestionSelection(
            accepted=tuple(accepted),
            rejected=tuple(rejected),
        )

    def update_suggestion_state(
        self,
        *,
        conversation_state: dict[str, Any],
        selection: SuggestionSelection,
        selected_suggestion_id: str | None,
    ) -> dict[str, list[str]]:
        current = _record(conversation_state.get("suggestion_state"))
        recent = _strings(current.get("recent_impressions"))
        prior_recent = set(recent)
        accepted_ids = _strings(current.get("accepted_suggestion_ids"))
        dismissed_ids = _strings(current.get("dismissed_suggestion_ids"))
        recent = list(
            dict.fromkeys(
                [
                    *(item.suggestion_id for item in selection.accepted),
                    *recent,
                ]
            )
        )[: self.history_limit]
        if selected_suggestion_id and selected_suggestion_id in prior_recent:
            accepted_ids = list(
                dict.fromkeys([selected_suggestion_id, *accepted_ids])
            )[: self.history_limit]
        return {
            "recent_impressions": recent,
            "accepted_suggestion_ids": accepted_ids,
            "dismissed_suggestion_ids": dismissed_ids[: self.history_limit],
        }

    @staticmethod
    def to_chat_suggestions(
        selection: SuggestionSelection,
    ) -> list[ChatSuggestion]:
        return [
            ChatSuggestion(
                suggestion_id=item.suggestion_id,
                type=item.type.value,
                label=item.label,
                query=item.query,
                target_intent=item.target_intent,
                reason_code=item.reason_code,
            )
            for item in selection.accepted
        ]


def _gate_reason(
    candidate: SuggestionCandidate,
    *,
    source_intent: Intent,
    source_policy,
    trusted_ids: set[str],
    evidence_ids: set[str],
) -> str | None:
    if candidate.target_intent not in source_policy.allowed_next_intents:
        return "target_intent_not_allowed"
    if source_policy.entity_required and not trusted_ids:
        return "source_entity_required"
    if source_policy.evidence_required and not evidence_ids:
        return "source_evidence_required"
    target_scope = capability_for(candidate.target_intent).entity_scope
    entity_count = len(candidate.entity_names)
    id_count = len(candidate.resolved_ids)
    if target_scope in {EntityScope.NONE, EntityScope.FILTER_ONLY} and (
        entity_count or id_count
    ):
        return "target_intent_blocks_entity"
    if target_scope == EntityScope.EXACTLY_ONE and (
        entity_count != 1 or id_count != 1
    ):
        return "target_requires_one_entity"
    if target_scope == EntityScope.TWO_OR_MORE and (
        entity_count < 2 or id_count < 2 or entity_count != id_count
    ):
        return "target_requires_multiple_entities"
    if target_scope == EntityScope.OPTIONAL and (
        entity_count > 1 or id_count > 1 or entity_count != id_count
    ):
        return "optional_entity_mismatch"
    if candidate.resolved_ids and not set(candidate.resolved_ids).issubset(
        trusted_ids
    ):
        return "untrusted_entity_id"
    if candidate.entity_names:
        normalized_query = normalize_vietnamese(candidate.query)
        if any(
            normalize_vietnamese(name) not in normalized_query
            for name in candidate.entity_names
        ):
            return "query_missing_canonical_entity"
    if candidate.evidence_ids and not set(candidate.evidence_ids).issubset(
        evidence_ids
    ):
        return "evidence_id_mismatch"
    if source_intent == Intent.CLINIC_INFO and candidate.entity_names:
        return "clinic_info_entity_leak"
    return None


def _top_entity(evidence, source_type: str) -> dict[str, str] | None:
    for item in evidence:
        name = str(item.raw_json.get("name") or "").strip()
        if item.source_type == source_type and name:
            return {"id": item.source_id, "name": name}
    return None


def _evidence_topics(intent: Intent, evidence_pack: EvidencePack) -> set[str]:
    topics: set[str] = set()
    expected_type = (
        "product"
        if intent in {Intent.PRODUCT_DETAIL, Intent.PRODUCT_COMPARE}
        else "service"
    )
    for item in evidence_pack.items:
        if item.source_type != expected_type:
            continue
        if item.raw_json.get("price") is not None:
            topics.add("price")
        if expected_type == "product" and item.raw_json.get("quantity") is not None:
            topics.add("availability")
        if (
            expected_type == "service"
            and item.raw_json.get("duration_minutes") is not None
        ):
            topics.add("duration")
    return topics


def _detect_topics(text: str) -> set[str]:
    normalized = normalize_vietnamese(text)
    topics: set[str] = set()
    phrases = {
        "price": ("gia", "chi phi", "bao nhieu tien"),
        "duration": ("bao lau", "thoi gian", "phut"),
        "availability": ("con hang", "ton kho", "so luong", "dang co"),
        "pain": ("co dau", "dau khong", "e buot"),
        "safety": ("an toan", "rui ro", "chong chi dinh", "luu y"),
        "usage": ("cach dung", "phu hop voi ai", "dung nhu the nao"),
        "comparison": ("so sanh", "khac nhau"),
        "clinic_hours": ("mo cua", "gio lam viec"),
        "clinic_contact": ("so dien thoai", "lien he"),
        "clinic_location": ("o dau", "dia chi"),
        "price_sort": ("gia tu thap den cao", "gia tu cao den thap"),
        "duration_sort": ("thoi gian ngan nhat", "thoi gian dai nhat"),
        "budget": ("duoi 1 trieu", "duoi 2 trieu", "ngan sach"),
    }
    for topic, markers in phrases.items():
        if any(marker in normalized for marker in markers):
            topics.add(topic)
    return topics


def _goals_for_topics(topics: set[str]) -> set[str]:
    goals: set[str] = set()
    if topics & {"price", "price_sort", "budget"}:
        goals.add("budget_research")
    if topics & {"duration", "duration_sort"}:
        goals.add("time_research")
    if topics & {"pain", "safety", "usage"}:
        goals.add("safety_research")
    if "availability" in topics:
        goals.add("purchase_readiness")
    if "comparison" in topics:
        goals.add("comparison")
    if "catalog" in topics:
        goals.add("catalog_discovery")
    return goals


def _journey_stage(intent: Intent) -> JourneyStage:
    if intent in {
        Intent.GREETING,
        Intent.CHITCHAT,
        Intent.UNKNOWN,
        Intent.PRODUCT_LIST,
        Intent.SERVICE_LIST,
    }:
        return JourneyStage.DISCOVERY
    if intent == Intent.PRODUCT_COMPARE:
        return JourneyStage.COMPARISON
    if intent == Intent.CLINIC_INFO:
        return JourneyStage.DECISION
    if intent == Intent.FAQ:
        return JourneyStage.SUPPORT
    return JourneyStage.CONSIDERATION


def _domain_for_intent(intent: Intent) -> str | None:
    return domain_for_intent(intent)


def _context_signature(
    domain: str | None,
    entities: tuple[InterestEntity, ...],
) -> str:
    ids = ",".join(sorted(entity.resolved_id for entity in entities))
    return f"{domain or 'none'}:{ids or 'none'}"


def _suggestion_id(
    target_intent: Intent,
    query: str,
    reason_code: str,
) -> str:
    raw = "|".join(
        (
            target_intent.value,
            normalize_vietnamese(query),
            reason_code,
        )
    )
    return "sg_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _record(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _strings(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return list(
        dict.fromkeys(
            text
            for item in values
            if (text := str(item).strip())
        )
    )
