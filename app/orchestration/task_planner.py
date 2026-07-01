import json
import re
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.constants import Intent
from app.generation.llm_client import LLMClient
from app.orchestration.conversation_context import (
    ConversationContext,
    build_conversation_context,
)
from app.orchestration.intent_registry import (
    EntityScope,
    capability_for,
    detail_intent_for_entity_type,
    domain_for_intent,
    list_intent_for_entity_type,
)
from app.orchestration.query_features import QueryFeatures
from app.orchestration.schemas import PlannedTask, TaskPlan
from app.retrieval.router import IntentRouter


class _PlannerOutput(BaseModel):
    tasks: list[PlannedTask] = Field(default_factory=list)
    global_entities: list[str] = Field(default_factory=list)
    clarification_question: str | None = None


class TaskPlanner:
    def __init__(self) -> None:
        self.router = IntentRouter()

    async def plan(
        self,
        *,
        query: str,
        history: dict[str, Any],
        settings: Settings,
        llm: LLMClient | None = None,
        ollama: LLMClient | None = None,
        known_products: list[str] | None = None,
        known_services: list[str] | None = None,
        known_product_categories: list[str] | None = None,
    ) -> TaskPlan:
        llm = llm or ollama
        known_products = known_products or []
        known_services = known_services or []
        known_product_categories = known_product_categories or []
        if settings.enable_llm_router and llm is not None:
            prompt = self._prompt(
                query=query,
                history=history,
                max_sub_queries=settings.max_sub_queries,
                known_products=known_products,
                known_services=known_services,
                known_product_categories=known_product_categories,
            )
            try:
                try:
                    response = await llm.generate(
                        prompt=prompt,
                        model=settings.llm_router_model,
                        system=TASK_PLANNER_SYSTEM_PROMPT,
                        json_mode=True,
                        timeout_seconds=settings.llm_router_timeout_seconds,
                        num_predict=settings.llm_router_num_predict,
                        num_ctx=settings.llm_router_num_ctx,
                        think=False,
                    )
                except Exception as provider_error:
                    return self._heuristic_plan(
                        query=query,
                        history=history,
                        settings=settings,
                        known_products=known_products,
                        known_services=known_services,
                        known_product_categories=known_product_categories,
                        metadata={
                            "llm_error": str(provider_error),
                            "planner_fallback": "heuristic_after_llm_error",
                        },
                    )
                planner_attempts = [response.trace_metadata()]
                try:
                    payload = self._parse_llm_output(response.text)
                except Exception as first_error:
                    try:
                        repaired = await llm.generate(
                            prompt=self._repair_prompt(response.text, str(first_error)),
                            model=settings.llm_router_model,
                            system=TASK_PLANNER_SYSTEM_PROMPT,
                            json_mode=True,
                            timeout_seconds=settings.llm_router_timeout_seconds,
                            num_predict=settings.llm_router_num_predict,
                            num_ctx=settings.llm_router_num_ctx,
                            think=False,
                        )
                    except Exception as provider_error:
                        return self._heuristic_plan(
                            query=query,
                            history=history,
                            settings=settings,
                            known_products=known_products,
                            known_services=known_services,
                            known_product_categories=known_product_categories,
                            metadata={
                                "llm_error": str(provider_error),
                                "first_parse_error": str(first_error),
                                "planner_fallback": "heuristic_after_llm_repair_error",
                            },
                        )
                    planner_attempts.append(repaired.trace_metadata())
                    try:
                        payload = self._parse_llm_output(repaired.text)
                    except Exception as repair_error:
                        return self._safe_unknown_plan(
                            query=query,
                            metadata={
                                "first_parse_error": str(first_error),
                                "repair_parse_error": str(repair_error),
                                "planner_fallback": "safe_unknown_after_invalid_llm_plan",
                                "attempts": planner_attempts,
                                "usage": _usage_from_attempts(planner_attempts),
                            },
                        )
                metadata: dict[str, Any] = {
                    "llm_prompt_chars": len(prompt),
                    "plan_reviewed": False,
                    "attempts": planner_attempts,
                    "usage": _usage_from_attempts(planner_attempts),
                }
                if settings.enable_plan_review and self._should_review_plan(
                    query,
                    payload,
                ):
                    try:
                        reviewed = await llm.generate(
                            prompt=self._review_prompt(
                                query=query,
                                history=history,
                                candidate=payload,
                                max_sub_queries=settings.max_sub_queries,
                            ),
                            model=settings.llm_router_model,
                            system=TASK_PLANNER_REVIEW_SYSTEM_PROMPT,
                            json_mode=True,
                            timeout_seconds=settings.llm_router_timeout_seconds,
                            num_predict=settings.llm_router_num_predict,
                            num_ctx=settings.llm_router_num_ctx,
                            think=False,
                        )
                        payload = self._parse_llm_output(reviewed.text)
                        review_attempt = reviewed.trace_metadata()
                        metadata["review_attempt"] = review_attempt
                        planner_attempts.append(review_attempt)
                        metadata["attempts"] = planner_attempts
                        metadata["usage"] = _usage_from_attempts(planner_attempts)
                        metadata["plan_reviewed"] = True
                    except Exception as review_error:
                        metadata["plan_review_error"] = str(review_error)
                elif settings.enable_plan_review:
                    metadata["plan_review_skipped"] = "simple_single_task"
                tasks = [
                    self._normalize_task_proposal(task, priority=index)
                    for index, task in enumerate(
                        payload.tasks[: settings.max_sub_queries],
                        start=1,
                    )
                ]
                if tasks:
                    return TaskPlan(
                        tasks=tuple(tasks),
                        planner_global_entities=tuple(
                            dict.fromkeys(
                                [
                                    *payload.global_entities,
                                    *[
                                        entity
                                        for task in tasks
                                        for entity in task.planner_entities
                                    ],
                                ]
                            )
                        ),
                        clarification_question=payload.clarification_question,
                        source=settings.llm_provider.lower().strip(),
                        metadata=metadata,
                    )
                return self._safe_unknown_plan(
                    query=query,
                    metadata={
                        **metadata,
                        "planner_fallback": "safe_unknown_after_empty_llm_plan",
                    },
                )
            except Exception as exc:
                return self._safe_unknown_plan(
                    query=query,
                    metadata={
                        "llm_error": str(exc),
                        "planner_fallback": "safe_unknown_after_llm_error",
                    },
                )
        return self._heuristic_plan(
            query=query,
            history=history,
            settings=settings,
            known_products=known_products,
            known_services=known_services,
            known_product_categories=known_product_categories,
        )

    def _heuristic_plan(
        self,
        *,
        query: str,
        history: dict[str, Any] | None = None,
        settings: Settings,
        known_products: list[str],
        known_services: list[str],
        known_product_categories: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> TaskPlan:
        parts = self._split_query(query)[: settings.max_sub_queries]
        if not parts:
            parts = [query]
        conversation_context = build_conversation_context(history or {})
        full_route = self.router.route(
            query,
            known_products=known_products,
            known_services=known_services,
            known_product_categories=known_product_categories,
        )
        full_features = QueryFeatures.extract(
            query=query,
            entity_names_in_query=tuple(full_route.entities),
        )
        context_entities = full_route.entities or (
            list(conversation_context.last_entity_names)
            if self._is_follow_up_query(
                query,
                conversation_context,
                full_features,
            )
            else []
        )
        tasks: list[PlannedTask] = []
        for index, part in enumerate(parts, start=1):
            routed = self.router.route(
                part,
                known_products=known_products,
                known_services=known_services,
                known_product_categories=known_product_categories,
            )
            features = QueryFeatures.extract(
                query=part,
                entity_names_in_query=tuple(routed.entities),
            )
            entities = routed.entities or (
                context_entities
                if self._is_follow_up_query(part, conversation_context, features)
                else []
            )
            intent = self._follow_up_intent(
                part,
                routed.intent,
                conversation_context,
                features,
            )
            intent = self._catalog_intent_for_unbound_detail(
                intent,
                entities,
                conversation_context,
            )
            task = PlannedTask(
                task_id=f"t{index}",
                intent=intent,
                planner_query=part,
                planner_entities=tuple(entities),
                priority=index,
                planner_needs_clarification=routed.needs_clarification,
                planner_clarification_question=routed.clarification_message,
            )
            tasks.append(self._normalize_task_proposal(task, priority=index))
        return TaskPlan(
            tasks=tuple(tasks),
            planner_global_entities=tuple(
                dict.fromkeys(
                    entity
                    for task in tasks
                    for entity in task.planner_entities
                )
            ),
            source="heuristic",
            metadata=metadata or {},
        )

    @staticmethod
    def _safe_unknown_plan(
        *,
        query: str,
        metadata: dict[str, Any] | None = None,
    ) -> TaskPlan:
        task = PlannedTask(
            task_id="t1",
            intent=Intent.UNKNOWN,
            planner_query=query,
            priority=1,
            planner_needs_clarification=True,
            planner_clarification_question=(
                "Bạn có thể nói rõ hơn bạn muốn hỏi về sản phẩm, dịch vụ, "
                "FAQ hay thông tin phòng khám không?"
            ),
        )
        return TaskPlan(
            tasks=(TaskPlanner._normalize_task_proposal(task, priority=1),),
            planner_global_entities=(),
            clarification_question=task.planner_clarification_question,
            source="safe_unknown",
            metadata=metadata or {},
        )

    @staticmethod
    def _split_query(query: str) -> list[str]:
        parts = [
            part.strip(" ,.;")
            for part in re.split(r"\s*(?:,|;|\bvà\b|\bngoài ra\b|\bđồng thời\b)\s*", query)
        ]
        return [part for part in parts if len(part) >= 4]

    @staticmethod
    def _normalize_task_proposal(
        task: PlannedTask,
        *,
        priority: int | None = None,
    ) -> PlannedTask:
        return task.model_copy(
            update={
                "planner_entity_type": task.planner_entity_type
                or _domain(task.intent),
                "priority": priority if priority is not None else task.priority,
            }
        )

    @staticmethod
    def _follow_up_intent(
        query: str,
        routed_intent: Intent,
        ctx: ConversationContext,
        features: QueryFeatures,
    ) -> Intent:
        if features.is_social:
            return Intent.CHITCHAT
        if features.is_schedule_query:
            return Intent.CLINIC_INFO
        if ctx.is_fresh_session:
            return routed_intent
        if (
            ctx.has_active_entity
            and not features.has_entity_mention
            and (features.is_short or features.has_context_reference)
        ):
            return TaskPlanner._resolve_domain_intent(routed_intent, features, ctx)
        if routed_intent == Intent.UNKNOWN and ctx.has_active_entity:
            if features.is_question:
                return Intent.FAQ
        return routed_intent

    @staticmethod
    def _catalog_intent_for_unbound_detail(
        intent: Intent,
        entities: list[str],
        ctx: ConversationContext,
    ) -> Intent:
        capability = capability_for(intent)
        if (
            capability.entity_scope != EntityScope.EXACTLY_ONE
            or entities
            or not ctx.is_fresh_session
        ):
            return intent
        return list_intent_for_entity_type(capability.entity_domain) or intent

    @staticmethod
    def _resolve_domain_intent(
        routed_intent: Intent,
        features: QueryFeatures,
        ctx: ConversationContext,
    ) -> Intent:
        domain = ctx.active_domain
        if domain is None:
            return routed_intent
        if features.is_availability or features.is_price:
            return detail_intent_for_entity_type(domain) or routed_intent
        if features.is_duration:
            return detail_intent_for_entity_type(domain) or routed_intent
        if features.is_question:
            return Intent.FAQ
        return routed_intent

    @staticmethod
    def _is_follow_up_query(
        query: str,
        ctx: ConversationContext | None = None,
        features: QueryFeatures | None = None,
    ) -> bool:
        features = features or QueryFeatures.extract(query)
        if features.is_social or features.is_schedule_query or features.has_entity_mention:
            return False
        if ctx is None:
            return features.is_short and (
                features.is_question or features.has_context_reference
            )
        if ctx.is_fresh_session or not ctx.has_active_entity:
            return False
        return (
            features.is_short
            or features.has_context_reference
            or features.is_question
        )

    @staticmethod
    def _history_context(history: dict[str, Any]) -> dict[str, Any]:
        return build_conversation_context(history).as_legacy_dict()

    @staticmethod
    def _parse_llm_output(text: str) -> _PlannerOutput:
        payload = _coerce_planner_payload(json.loads(_json_payload(text)))
        return _PlannerOutput.model_validate(payload)

    @staticmethod
    def _should_review_plan(query: str, payload: _PlannerOutput) -> bool:
        if len(payload.tasks) != 1:
            return True
        normalized = IntentRouter._normalize(query)
        if any(
            marker in normalized
            for marker in (" va ", " ngoai ra ", " dong thoi ", ";", ",")
        ):
            return True
        task = payload.tasks[0]
        return (
            task.intent == Intent.UNKNOWN
            or task.planner_needs_clarification
        )

    @staticmethod
    def _prompt(
        *,
        query: str,
        history: dict[str, Any],
        max_sub_queries: int,
        known_products: list[str],
        known_services: list[str],
        known_product_categories: list[str],
    ) -> str:
        products = IntentRouter._relevant_terms(query, known_products, limit=12)
        services = IntentRouter._relevant_terms(query, known_services, limit=12)
        categories = IntentRouter._relevant_terms(query, known_product_categories, limit=12)
        schema = {
            "tasks": [
                {
                    "task_id": "t1",
                    "intent": "SERVICE_DETAIL",
                    "query": "tẩy trắng răng giá bao nhiêu",
                    "entities": ["Tẩy trắng răng"],
                    "domain": "service",
                    "operation": "detail_lookup",
                    "selection": {
                        "mode": "explicit_entities",
                        "entity_type": "service",
                        "mentions": ["Tẩy trắng răng"],
                        "filters": {
                            "service_names": ["Tẩy trắng răng"],
                            "price_max": None,
                            "duration_max": None,
                            "feature_terms": [],
                            "symptom_terms": [],
                        },
                        "sort": {"field": None, "direction": None},
                        "limit": 10,
                    },
                    "required_tools": ["service_tool"],
                    "priority": 1,
                    "needs_clarification": False,
                    "clarification_question": None,
                }
            ],
            "global_entities": ["Tẩy trắng răng"],
            "clarification_question": None,
        }
        return (
            "Phân tích user query thành các task độc lập cho chatbot nha khoa.\n"
            f"Tối đa {max_sub_queries} task. Query đơn giản thì chỉ trả 1 task.\n"
            "Mỗi task có đúng một intent. Intent hợp lệ: "
            f"{', '.join(intent.value for intent in Intent)}.\n"
            "Không trả lời user. Chỉ trả JSON object hợp lệ.\n"
            "required_tools chỉ là gợi ý; server sẽ áp chính sách tool theo intent.\n"
            "Luôn cố gắng điền domain/operation/selection. "
            "selection dùng một schema chung: mode=explicit_entities/filter/all/auto, "
            "mentions là list string chỉ cho tên entity cụ thể, filters là object constraints, "
            "sort là object hoặc null.\n"
            "Không nhét filter như giá/danh mục/tính năng vào entities. "
            "Đưa chúng vào selection.filters.\n"
            "Product filters hỗ trợ: product_names, product_ids, category_terms, "
            "category_codes, brands, feature_terms, price_min, price_max, "
            "quantity_min, quantity_max, stock.\n"
            "Service filters hỗ trợ: service_names, service_ids, category_terms, "
            "category_codes, feature_terms, symptom_terms, price_min, price_max, "
            "duration_min, duration_max.\n"
            "Sort format: {\"field\":\"price|quantity|duration|name|category\", "
            "\"direction\":\"asc|desc\"}.\n"
            "Phân biệt theo nhiệm vụ, không chỉ theo entity:\n"
            "- 'sản phẩm nào/dịch vụ nào', điều kiện giá, thời lượng, sort => LIST.\n"
            "- Hỏi giá/thời lượng của một tên cụ thể => DETAIL.\n"
            "- Hỏi đau, an toàn, cách dùng, rủi ro, sau điều trị, 'có ... không' => FAQ.\n"
            "- Một câu có cả dữ liệu chi tiết và câu hỏi y khoa phải tách thành 2 task.\n"
            "Ví dụ: 'Dịch vụ nào liên quan implant và dưới 2 triệu?' => SERVICE_LIST, "
            "entities rỗng, filters={category_terms:['implant'], price_max:2000000}.\n"
            "Ví dụ: 'Tẩy trắng giá bao nhiêu và có đau không?' => SERVICE_DETAIL + FAQ.\n"
            "Nếu query hỏi nhiều ý, tách thành nhiều task. Nếu là follow-up, dùng history "
            "để khôi phục entity.\n\n"
            f"Schema ví dụ:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Conversation history:\n{json.dumps(history, ensure_ascii=False, default=str)}\n\n"
            f"Sản phẩm liên quan: {products or []}\n"
            f"Dịch vụ liên quan: {services or []}\n"
            f"Danh mục sản phẩm liên quan: {categories or []}\n"
            f"User query: {query}"
        )

    @staticmethod
    def _review_prompt(
        *,
        query: str,
        history: dict[str, Any],
        candidate: _PlannerOutput,
        max_sub_queries: int,
    ) -> str:
        return (
            "Kiểm tra và sửa candidate task plan cho chatbot nha khoa. "
            f"Giữ tối đa {max_sub_queries} task và trả đúng JSON schema như candidate.\n"
            "Bắt buộc kiểm tra đủ các ý trong query, đặc biệt phần sau 'và'.\n"
            "Quy tắc:\n"
            "1. 'sản phẩm nào/dịch vụ nào', filter, sort, dưới/trên một mức => LIST, "
            "không chọn một entity cụ thể.\n"
            "2. Giá/thời lượng của một tên cụ thể => DETAIL.\n"
            "3. Đau, ê buốt, an toàn, rủi ro, cách dùng, sau điều trị, "
            "'có ... không' => FAQ dù có tên sản phẩm/dịch vụ.\n"
            "4. Câu hỏi nhiều ý phải có task riêng cho từng ý; FAQ follow-up giữ entity "
            "của task trước để lấy đúng context.\n"
            "Không trả lời user. Không viết markdown.\n\n"
            f"Query: {query}\n"
            f"History: {json.dumps(history, ensure_ascii=False, default=str)}\n"
            f"Candidate: {candidate.model_dump_json()}"
        )

    @staticmethod
    def _repair_prompt(raw_text: str, error: str) -> str:
        return (
            "Sửa output sau thành đúng một JSON object hợp lệ theo schema task planner. "
            "Không trả lời user, không thêm markdown.\n"
            f"Lỗi: {error}\n"
            f"Output:\n{raw_text}"
        )


def _domain(intent: Intent) -> str | None:
    return domain_for_intent(intent)


def _coerce_planner_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"tasks": []}
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    coerced_tasks = []
    for index, raw_task in enumerate(tasks, start=1):
        if not isinstance(raw_task, dict):
            continue
        task = dict(raw_task)
        task["task_id"] = str(task.get("task_id") or f"t{index}")
        task["query"] = str(task.get("query") or payload.get("query") or "").strip()
        task["intent"] = _coerce_intent(task.get("intent"), task["query"])
        raw_entities = task.get("entities")
        mentions = _string_values(raw_entities)
        task["entities"] = mentions
        selection = task.get("selection")
        if not isinstance(selection, dict):
            selection = {}
        constraints = task.get("constraints")
        if isinstance(constraints, dict) and not selection.get("filters"):
            selection["filters"] = constraints
        filters = task.get("filters")
        if isinstance(filters, dict) and not selection.get("filters"):
            selection["filters"] = filters
        sort = task.get("sort")
        if isinstance(sort, dict) and not selection.get("sort"):
            selection["sort"] = sort
        if task.get("limit") is not None and selection.get("limit") is None:
            selection["limit"] = task.get("limit")
        if not selection.get("mentions"):
            selection["mentions"] = mentions
        if not selection.get("mode"):
            selection["mode"] = "auto"
        task["selection"] = selection
        coerced_tasks.append(task)
    payload["tasks"] = coerced_tasks
    payload["global_entities"] = _string_values(payload.get("global_entities"))
    return payload


def _coerce_intent(value: object, query: str) -> str:
    text = str(value or "").strip().upper()
    if text in {intent.value for intent in Intent}:
        return text
    if text in {"THANK_YOU", "THANKS"}:
        return Intent.CHITCHAT.value
    routed = IntentRouter().route(query)
    return routed.intent.value


def _string_values(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def _resolved_ids(payload: dict[str, Any], source_type: str) -> list[str]:
    values: list[str] = []
    direct_key = f"active_{source_type}_ids"
    values.extend(_string_values(payload.get(direct_key)))
    evidence_ids = payload.get("evidence_ids")
    if isinstance(evidence_ids, list):
        for item in evidence_ids:
            if not isinstance(item, dict):
                continue
            if item.get("type") == source_type and item.get("id"):
                values.append(str(item["id"]))
    return list(dict.fromkeys(values))


def _usage_from_attempts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    prompt_tokens = sum(int(attempt.get("prompt_eval_count") or 0) for attempt in attempts)
    completion_tokens = sum(int(attempt.get("eval_count") or 0) for attempt in attempts)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _json_payload(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return stripped[start : end + 1]
        return stripped


TASK_PLANNER_SYSTEM_PROMPT = (
    "Bạn là task planner cho chatbot nha khoa. "
    "Nhiệm vụ duy nhất là tách query thành task và trả JSON đúng schema."
)

TASK_PLANNER_REVIEW_SYSTEM_PROMPT = (
    "Bạn là reviewer của task planner chatbot nha khoa. "
    "Bạn phải sửa intent/decomposition sai và chỉ trả JSON đúng schema."
)
