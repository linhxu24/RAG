import json
import re
from typing import Any

from app.config import Settings
from app.constants import Intent
from app.generation.llm_client import LLMClient
from app.generation.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_chitchat_prompt,
    build_generation_prompt,
    build_json_fix_prompt,
    build_synthesis_json_fix_prompt,
    build_synthesis_prompt,
)
from app.generation.schemas import (
    EntityReference,
    GeneratedResponse,
    ResultBody,
    ResultItem,
    SafetyInfo,
    SourceReference,
    SynthesisOutput,
)
from app.generation.validator import ResponseValidationError, ResponseValidator
from app.orchestration.intent_registry import EntityScope, IntentCapability, capability_for


class GenerationValidationError(ResponseValidationError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


class GroundedGenerator:
    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm
        self.validator = ResponseValidator()

    async def generate_with_retry(
        self,
        *,
        query: str,
        intent: Intent,
        confidence: float,
        entities: list[str],
        context: dict[str, Any],
        session,
    ) -> tuple[GeneratedResponse, dict[str, Any]]:
        prompt = build_generation_prompt(
            query=query,
            intent=intent,
            confidence=confidence,
            entities=entities,
            context=context,
        )
        first = await self.llm.generate(
            prompt=prompt,
            model=self.settings.llm_generation_model,
            system=SYSTEM_PROMPT,
            json_mode=True,
            timeout_seconds=self.settings.llm_generation_timeout_seconds,
            num_predict=self.settings.llm_generation_num_predict,
            num_ctx=self.settings.llm_generation_num_ctx,
        )
        attempts = [first.trace_metadata()]
        try:
            response = self.validator.validate(first.text, context=context, session=session)
            return response, {
                "prompt": prompt,
                "llm": {
                    "model": first.model,
                    "latency_ms": first.latency_ms,
                    "attempt_count": 1,
                    "attempts": attempts,
                },
                "validation": {"valid": True, "retried": False},
            }
        except ResponseValidationError as first_error:
            fixed = await self.llm.generate(
                prompt=build_json_fix_prompt(first.text, str(first_error)),
                model=self.settings.llm_generation_model,
                system=SYSTEM_PROMPT,
                json_mode=True,
                timeout_seconds=self.settings.llm_generation_timeout_seconds,
                num_predict=self.settings.llm_generation_num_predict,
                num_ctx=self.settings.llm_generation_num_ctx,
            )
            attempts.append(fixed.trace_metadata())
            try:
                response = self.validator.validate(fixed.text, context=context, session=session)
            except ResponseValidationError as second_error:
                raise GenerationValidationError(
                    str(second_error),
                    {
                        "model": fixed.model,
                        "latency_ms": sum(attempt["latency_ms"] for attempt in attempts),
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                        "validation": {
                            "valid": False,
                            "retried": True,
                            "first_error": str(first_error),
                            "second_error": str(second_error),
                        },
                    },
                ) from second_error
            return response, {
                "prompt": prompt,
                "llm": {
                    "model": fixed.model,
                    "latency_ms": first.latency_ms + fixed.latency_ms,
                    "attempt_count": 2,
                    "attempts": attempts,
                },
                "validation": {
                    "valid": True,
                    "retried": True,
                    "first_error": str(first_error),
                },
            }

    async def generate_chitchat_with_retry(
        self,
        *,
        query: str,
        confidence: float,
        session,
        intent: Intent = Intent.CHITCHAT,
    ) -> tuple[GeneratedResponse, dict[str, Any]]:
        prompt = build_chitchat_prompt(
            query=query,
            intent=intent,
            confidence=confidence,
        )
        context = {"items": [], "total_chars": 0}
        first = await self.llm.generate(
            prompt=prompt,
            model=self.settings.llm_generation_model,
            system=CHITCHAT_SYSTEM_PROMPT,
            json_mode=True,
            timeout_seconds=self.settings.llm_generation_timeout_seconds,
            num_predict=self.settings.llm_generation_num_predict,
            num_ctx=self.settings.llm_generation_num_ctx,
        )
        attempts = [first.trace_metadata()]
        try:
            response = self.validator.validate(first.text, context=context, session=session)
            return response, {
                "prompt": prompt,
                "llm": {
                    "model": first.model,
                    "latency_ms": first.latency_ms,
                    "attempt_count": 1,
                    "attempts": attempts,
                },
                "validation": {"valid": True, "retried": False},
            }
        except ResponseValidationError as first_error:
            fixed = await self.llm.generate(
                prompt=build_json_fix_prompt(first.text, str(first_error)),
                model=self.settings.llm_generation_model,
                system=CHITCHAT_SYSTEM_PROMPT,
                json_mode=True,
                timeout_seconds=self.settings.llm_generation_timeout_seconds,
                num_predict=self.settings.llm_generation_num_predict,
                num_ctx=self.settings.llm_generation_num_ctx,
            )
            attempts.append(fixed.trace_metadata())
            try:
                response = self.validator.validate(fixed.text, context=context, session=session)
            except ResponseValidationError as second_error:
                raise GenerationValidationError(
                    str(second_error),
                    {
                        "model": fixed.model,
                        "latency_ms": sum(attempt["latency_ms"] for attempt in attempts),
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                        "validation": {
                            "valid": False,
                            "retried": True,
                            "first_error": str(first_error),
                            "second_error": str(second_error),
                        },
                    },
                ) from second_error
            return response, {
                "prompt": prompt,
                "llm": {
                    "model": fixed.model,
                    "latency_ms": first.latency_ms + fixed.latency_ms,
                    "attempt_count": 2,
                    "attempts": attempts,
                },
                "validation": {
                    "valid": True,
                    "retried": True,
                    "first_error": str(first_error),
                },
            }

    async def generate_synthesis_with_retry(
        self,
        *,
        query: str,
        intent: Intent,
        confidence: float,
        evidence_pack: dict[str, Any],
        context: dict[str, Any],
        session,
    ) -> tuple[GeneratedResponse, dict[str, Any]]:
        prompt = build_synthesis_prompt(
            query=query,
            intent=intent,
            confidence=confidence,
            evidence_pack=evidence_pack,
        )
        first = await self.llm.generate(
            prompt=prompt,
            model=self.settings.llm_generation_model,
            system=SYNTHESIS_SYSTEM_PROMPT,
            json_mode=True,
            timeout_seconds=self.settings.llm_generation_timeout_seconds,
            num_predict=self.settings.llm_generation_num_predict,
            num_ctx=self.settings.llm_generation_num_ctx,
        )
        attempts = [first.trace_metadata()]
        try:
            synthesis = self._validate_synthesis_completion(first)
            response = self._build_synthesis_response(
                synthesis=synthesis,
                intent=intent,
                confidence=confidence,
                context=context,
                session=session,
            )
            return response, {
                "prompt": prompt,
                "llm": {
                    "model": first.model,
                    "latency_ms": first.latency_ms,
                    "attempt_count": 1,
                    "attempts": attempts,
                },
                "validation": {"valid": True, "retried": False},
            }
        except ResponseValidationError as first_error:
            fixed = await self.llm.generate(
                prompt=build_synthesis_json_fix_prompt(first.text, str(first_error)),
                model=self.settings.llm_generation_model,
                system=SYNTHESIS_SYSTEM_PROMPT,
                json_mode=True,
                timeout_seconds=self.settings.llm_generation_timeout_seconds,
                num_predict=self.settings.llm_generation_num_predict,
                num_ctx=self.settings.llm_generation_num_ctx,
            )
            attempts.append(fixed.trace_metadata())
            try:
                synthesis = self._validate_synthesis_completion(fixed)
                response = self._build_synthesis_response(
                    synthesis=synthesis,
                    intent=intent,
                    confidence=confidence,
                    context=context,
                    session=session,
                )
            except ResponseValidationError as second_error:
                raise GenerationValidationError(
                    str(second_error),
                    {
                        "model": fixed.model,
                        "latency_ms": sum(attempt["latency_ms"] for attempt in attempts),
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                        "validation": {
                            "valid": False,
                            "retried": True,
                            "first_error": str(first_error),
                            "second_error": str(second_error),
                        },
                    },
                ) from second_error
            return response, {
                "prompt": prompt,
                "llm": {
                    "model": fixed.model,
                    "latency_ms": first.latency_ms + fixed.latency_ms,
                    "attempt_count": 2,
                    "attempts": attempts,
                },
                "validation": {
                    "valid": True,
                    "retried": True,
                    "first_error": str(first_error),
                },
            }

    @staticmethod
    def _validate_synthesis_completion(response) -> SynthesisOutput:
        if response.done_reason == "length":
            raise ResponseValidationError("Ollama truncated synthesis output")
        try:
            payload = json.loads(_json_payload(response.text))
            return SynthesisOutput.model_validate(payload)
        except Exception as exc:
            if isinstance(exc, ResponseValidationError):
                raise
            raise ResponseValidationError(
                f"Synthesis schema validation failed: {exc}"
            ) from exc

    def _build_synthesis_response(
        self,
        *,
        synthesis: SynthesisOutput,
        intent: Intent,
        confidence: float,
        context: dict[str, Any],
        session,
    ) -> GeneratedResponse:
        context_items = list(context.get("items", []))
        by_id = {
            str(item["source_id"]): item
            for item in context_items
            if item.get("source_id")
        }
        invalid_ids = [
            source_id
            for source_id in synthesis.used_source_ids
            if source_id not in by_id
        ]
        if invalid_ids:
            raise ResponseValidationError(
                f"Synthesis referenced unknown source IDs: {invalid_ids}"
            )
        selected_ids = list(dict.fromkeys(synthesis.used_source_ids))
        selected = (
            [by_id[source_id] for source_id in selected_ids]
            if selected_ids
            else context_items
        )
        required_task_ids = {
            str(task.get("task_id"))
            for task in context.get("tasks", [])
            if isinstance(task, dict) and task.get("task_id")
        }
        selected_task_ids = {
            str(item.get("source", {}).get("task_id"))
            for item in selected
            if item.get("source", {}).get("task_id")
        }
        missing_task_ids = sorted(required_task_ids - selected_task_ids)
        if missing_task_ids:
            raise ResponseValidationError(
                "Synthesis did not cite evidence for task IDs: "
                f"{missing_task_ids}"
            )
        result_items = [self._result_item(item) for item in selected]
        sources = [
            SourceReference(
                source_type=item["source_type"],
                source_id=item["source_id"],
                doc_id=item.get("source", {}).get("doc_id"),
                page_number=item.get("source", {}).get("page_number"),
            )
            for item in selected
        ]
        entities = [
            EntityReference(
                type=self._entity_type(item["source_type"]),
                name=str(
                    item.get("raw_json", {}).get("name")
                    or item.get("raw_json", {}).get("question")
                    or item.get("raw_json", {}).get("key")
                    or item["source_type"]
                ),
                matched_id=item["source_id"],
            )
            for item in selected
            if item["source_type"] in {"product", "service", "faq", "clinic_info"}
        ]
        disclaimer_task = any(
            self._intent_requires_medical_disclaimer(task.get("intent"))
            for task in context.get("tasks", [])
            if isinstance(task, dict)
        )
        response = GeneratedResponse(
            intent=intent,
            confidence=confidence,
            answer_type="rag",
            entities=entities,
            result=ResultBody(
                text=synthesis.answer,
                items=result_items,
                sources=sources,
            ),
            safety=SafetyInfo(
                medical_disclaimer_required=(
                    synthesis.medical_disclaimer_required
                    or self._intent_requires_medical_disclaimer(intent)
                    or disclaimer_task
                ),
                needs_human_support=synthesis.needs_human_support,
            ),
        )
        return self.validator.validate(
            response.model_dump(mode="json"),
            context=context,
            session=session,
        )

    def direct_response(
        self,
        *,
        intent: Intent,
        confidence: float,
        context: dict[str, Any],
        clarification_reason: str | None = None,
        clarification_message: str | None = None,
    ) -> GeneratedResponse:
        items = context.get("items", [])
        capability = capability_for(intent)
        if clarification_reason:
            messages = {
                "structured_entity_not_found": (
                    "Tôi chưa xác định được chính xác sản phẩm hoặc dịch vụ bạn muốn hỏi. "
                    "Bạn vui lòng nhập tên đầy đủ hơn."
                ),
                "fuzzy_or_ambiguous_structured_match": (
                    "Có nhiều dữ liệu gần giống với yêu cầu của bạn. "
                    "Bạn vui lòng nhập rõ tên sản phẩm hoặc dịch vụ."
                ),
                "compare_entities_incomplete": (
                    "Bạn vui lòng nêu rõ ít nhất hai tên sản phẩm cần so sánh."
                ),
            }
            return self._simple(
                intent,
                confidence,
                "clarification",
                clarification_message
                or messages.get(
                    clarification_reason,
                    "Bạn vui lòng cung cấp thêm thông tin để tôi truy vấn chính xác.",
                ),
            )
        simple_response = self._simple_direct_response(intent)
        if simple_response is not None:
            answer_type, text = simple_response
            return self._simple(
                intent,
                confidence,
                answer_type,
                text,
            )
        if self._uses_product_compare_format(capability) and len(items) < max(
            capability.evidence_contract.minimum_items,
            2,
        ):
            return self._simple(
                intent,
                confidence,
                "clarification",
                "Tôi chưa tìm thấy đủ hai sản phẩm trong dữ liệu hiện tại để so sánh. "
                "Bạn vui lòng nêu rõ tên các sản phẩm.",
            )
        if not items:
            return self._simple(
                intent,
                confidence,
                "fallback",
                self._empty_fallback_text(capability),
            )

        text = self._format_direct_text(items, capability)

        result_items = [self._result_item(item) for item in items]
        entities = [
            EntityReference(
                type=self._entity_type(item["source_type"]),
                name=str(
                    item.get("raw_json", {}).get("name")
                    or item.get("raw_json", {}).get("question")
                    or item.get("raw_json", {}).get("key")
                    or item["source_type"]
                ),
                matched_id=item["source_id"],
            )
            for item in items
            if item["source_type"] in {"product", "service", "faq", "clinic_info"}
        ]
        return GeneratedResponse(
            intent=intent,
            confidence=confidence,
            answer_type="direct_data",
            entities=entities,
            result=ResultBody(
                text=text,
                items=result_items,
                sources=[
                    SourceReference(
                        source_type=item["source_type"],
                        source_id=item["source_id"],
                        doc_id=item.get("source", {}).get("doc_id"),
                        page_number=item.get("source", {}).get("page_number"),
                    )
                    for item in items
                ],
            ),
            safety=SafetyInfo(
                medical_disclaimer_required="faq"
                in capability.evidence_contract.allowed_source_types,
                needs_human_support=False,
            ),
        )

    def fallback_from_context(
        self,
        *,
        intent: Intent,
        confidence: float,
        context: dict[str, Any],
    ) -> GeneratedResponse:
        response = self.direct_response(
            intent=intent,
            confidence=confidence,
            context=self._focused_fallback_context(intent, context),
        )
        response.degraded = True
        if response.answer_type == "direct_data":
            response.answer_type = "fallback"
            response.result.text += (
                "\n\nĐây là câu trả lời dự phòng từ dữ liệu đã truy xuất vì bước diễn đạt "
                "tự nhiên không hoàn tất."
            )
        return response

    def partial_evidence_response(
        self,
        *,
        intent: Intent,
        confidence: float,
        context: dict[str, Any],
        missing_info: list[str],
    ) -> GeneratedResponse:
        response = self.direct_response(
            intent=intent,
            confidence=confidence,
            context=context,
        )
        if response.answer_type == "direct_data":
            response.answer_type = "fallback"
        if missing_info:
            response.result.text += "\n\n" + "\n".join(
                f"- {message}" for message in missing_info
            )
        return response

    @staticmethod
    def _simple_direct_response(intent: Intent) -> tuple[str, str] | None:
        responses = {
            Intent.GREETING: (
                "greeting",
                "Xin chào! Tôi có thể hỗ trợ bạn về sản phẩm, dịch vụ, FAQ "
                "hoặc thông tin phòng khám.",
            ),
            Intent.CHITCHAT: (
                "chitchat",
                "Cảm ơn bạn. Tôi luôn sẵn sàng hỗ trợ các câu hỏi về nha khoa "
                "và phòng khám.",
            ),
            Intent.UNKNOWN: (
                "clarification",
                "Mình chưa hiểu rõ bạn muốn hỏi về sản phẩm, dịch vụ, FAQ hay "
                "thông tin phòng khám. Bạn có thể hỏi cụ thể hơn không?",
            ),
        }
        return responses.get(intent)

    @staticmethod
    def _intent_requires_medical_disclaimer(intent: Intent | object) -> bool:
        try:
            capability = capability_for(
                intent if isinstance(intent, Intent) else Intent(str(intent))
            )
        except ValueError:
            return False
        return "faq" in capability.evidence_contract.allowed_source_types

    @staticmethod
    def _empty_fallback_text(capability: IntentCapability) -> str:
        if capability.entity_scope == EntityScope.FILTER_ONLY:
            messages = {
                "product": (
                    "Tôi chưa tìm thấy sản phẩm nào đáp ứng các điều kiện bạn yêu cầu."
                ),
                "service": (
                    "Tôi chưa tìm thấy dịch vụ nào đáp ứng các điều kiện bạn yêu cầu."
                ),
            }
            message = messages.get(capability.entity_domain)
            if message:
                return message
        return "Hiện tại tôi chưa có đủ thông tin trong dữ liệu của phòng khám."

    @classmethod
    def _format_direct_text(
        cls,
        items: list[dict[str, Any]],
        capability: IntentCapability,
    ) -> str:
        if cls._uses_product_compare_format(capability):
            # Product comparison has a side-by-side output shape; source_type alone
            # cannot infer that presentation without the capability contract.
            return cls._format_product_compare(items)

        source_type = cls._preferred_source_type(items, capability)
        source_items = [
            item for item in items if item.get("source_type") == source_type
        ]
        if not source_items:
            return "\n".join(item["text"] for item in items)

        single_formatters = {
            "faq": cls._format_faq,
        }
        single_formatter = single_formatters.get(source_type)
        if single_formatter is not None:
            return single_formatter(source_items[0])

        detail_formatters = {
            "product": cls._format_product_detail,
            "service": cls._format_service_detail,
        }
        if capability.entity_scope == EntityScope.EXACTLY_ONE and len(source_items) == 1:
            detail_formatter = detail_formatters.get(source_type)
            if detail_formatter is not None:
                return detail_formatter(source_items[0])

        list_formatters = {
            "product": cls._format_product_list,
            "service": cls._format_service_list,
        }
        list_formatter = list_formatters.get(source_type)
        if list_formatter is not None:
            return list_formatter(source_items)
        return "\n".join(item["text"] for item in items)

    @staticmethod
    def _preferred_source_type(
        items: list[dict[str, Any]],
        capability: IntentCapability,
    ) -> str:
        for source_type in capability.evidence_contract.allowed_source_types:
            if any(item.get("source_type") == source_type for item in items):
                return source_type
        return str(items[0].get("source_type") or "")

    @staticmethod
    def _uses_product_compare_format(capability: IntentCapability) -> bool:
        return (
            capability.entity_scope == EntityScope.TWO_OR_MORE
            and capability.entity_domain == "product"
        )

    @staticmethod
    def _focused_fallback_context(
        intent: Intent,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        items = list(context.get("items", []))
        capability = capability_for(intent)
        preferred_types: set[str] | None = None
        limit: int | None = None
        source_types = set(capability.evidence_contract.allowed_source_types)
        if "faq" in source_types:
            preferred_types = {"faq"}
        elif source_types:
            preferred_types = source_types
        if capability.entity_scope == EntityScope.EXACTLY_ONE:
            limit = 1
        elif capability.entity_scope == EntityScope.TWO_OR_MORE:
            limit = max(capability.evidence_contract.minimum_items, 2)

        if preferred_types:
            focused = [item for item in items if item["source_type"] in preferred_types]
            if focused:
                items = focused[:limit]
        return {
            **context,
            "items": items,
            "total_chars": sum(len(item["text"]) for item in items),
        }

    @staticmethod
    def _result_item(item: dict[str, Any]) -> ResultItem:
        source_type = item["source_type"]
        source = item.get("source", {})
        raw = item.get("raw_json", {})
        return ResultItem(
            type=source_type,
            id=item["source_id"],
            name=raw.get("name") or raw.get("question"),
            chunk_id=item["source_id"] if source_type == "chunk" else None,
            row_id=item["source_id"] if source_type == "table_row" else source.get("row_id"),
            doc_id=source.get("doc_id"),
            asset_ids=[raw["asset_id"]] if raw.get("asset_id") else [],
            data=raw,
        )

    @staticmethod
    def _entity_type(source_type: str) -> str:
        return (
            source_type
            if source_type in {"product", "service", "faq", "clinic_info"}
            else "unknown"
        )

    @staticmethod
    def _format_money(value: Any, currency: str | None = "VND") -> str | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return f"{value} {currency or ''}".strip()
        if currency == "VND":
            return f"{numeric:,.0f} VND".replace(",", ".")
        return f"{numeric:g} {currency or ''}".strip()

    @classmethod
    def _format_product_list(cls, items: list[dict[str, Any]]) -> str:
        products = [item for item in items if item["source_type"] == "product"]
        if not products:
            return "Hiện tại tôi chưa tìm thấy sản phẩm phù hợp trong dữ liệu phòng khám."
        lines = [f"Mình tìm thấy {len(products)} sản phẩm phù hợp:"]
        for index, item in enumerate(products[:10], start=1):
            raw = item.get("raw_json", {})
            details = []
            price = cls._format_money(raw.get("price"), raw.get("currency"))
            if price:
                details.append(f"giá {price}")
            if raw.get("category"):
                details.append(f"danh mục {raw['category']}")
            if raw.get("quantity") is not None:
                details.append(f"còn {raw['quantity']}")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(f"{index}. {raw.get('name') or item['text']}{suffix}")
        if len(products) > 10:
            lines.append(f"Còn {len(products) - 10} sản phẩm khác trong kết quả.")
        return "\n".join(lines)

    @classmethod
    def _format_service_list(cls, items: list[dict[str, Any]]) -> str:
        services = [item for item in items if item["source_type"] == "service"]
        if not services:
            return "Hiện tại tôi chưa tìm thấy dịch vụ phù hợp trong dữ liệu phòng khám."
        lines = [f"Mình tìm thấy {len(services)} dịch vụ phù hợp:"]
        for index, item in enumerate(services[:10], start=1):
            raw = item.get("raw_json", {})
            details = []
            if raw.get("duration_minutes") is not None:
                details.append(f"{raw['duration_minutes']} phút")
            price = cls._format_money(raw.get("price"), raw.get("currency"))
            if price:
                details.append(f"giá {price}")
            category = raw.get("source_category") or raw.get("category_code")
            if category:
                details.append(f"nhóm {category}")
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(f"{index}. {raw.get('name') or item['text']}{suffix}")
        if len(services) > 10:
            lines.append(f"Còn {len(services) - 10} dịch vụ khác trong kết quả.")
        return "\n".join(lines)

    @classmethod
    def _format_product_detail(cls, item: dict[str, Any]) -> str:
        raw = item.get("raw_json", {})
        name = raw.get("name") or "Sản phẩm này"
        parts = [str(name)]
        price = cls._format_money(raw.get("price"), raw.get("currency"))
        if price:
            parts.append(f"có giá {price}")
        if raw.get("quantity") is not None:
            parts.append(f"số lượng hiện có là {raw['quantity']}")
        if raw.get("category"):
            parts.append(f"thuộc danh mục {raw['category']}")
        sentence = ", ".join(parts) + "."
        if raw.get("description"):
            sentence += f" {raw['description']}"
        if raw.get("link"):
            sentence += f" Link tham khảo: {raw['link']}."
        return sentence

    @classmethod
    def _format_service_detail(cls, item: dict[str, Any]) -> str:
        raw = item.get("raw_json", {})
        name = raw.get("name") or "Dịch vụ này"
        parts = [str(name)]
        if raw.get("duration_minutes") is not None:
            parts.append(f"thường kéo dài khoảng {raw['duration_minutes']} phút")
        price = cls._format_money(raw.get("price"), raw.get("currency"))
        if price:
            parts.append(f"chi phí là {price}")
        category = raw.get("source_category") or raw.get("category_code")
        if category:
            parts.append(f"thuộc nhóm {category}")
        sentence = ", ".join(parts) + "."
        if raw.get("description"):
            sentence += f" {raw['description']}"
        return sentence

    @classmethod
    def _format_product_compare(cls, items: list[dict[str, Any]]) -> str:
        products = [item for item in items if item["source_type"] == "product"][:4]
        if len(products) < 2:
            return (
                "Tôi chưa tìm thấy đủ hai sản phẩm trong dữ liệu hiện tại để so sánh. "
                "Bạn vui lòng nêu rõ tên các sản phẩm."
            )
        lines = ["Dựa trên dữ liệu hiện có, có thể so sánh nhanh như sau:"]
        for item in products:
            raw = item.get("raw_json", {})
            details = []
            price = cls._format_money(raw.get("price"), raw.get("currency"))
            if price:
                details.append(f"giá {price}")
            if raw.get("category"):
                details.append(f"danh mục {raw['category']}")
            if raw.get("quantity") is not None:
                details.append(f"còn {raw['quantity']}")
            name = raw.get("name") or item["source_id"]
            summary = ", ".join(details) or item["text"]
            lines.append(f"- {name}: {summary}")
        return "\n".join(lines)

    @staticmethod
    def _format_faq(item: dict[str, Any]) -> str:
        raw = item.get("raw_json", {})
        answer = str(raw.get("answer") or item["text"])
        question = raw.get("question")
        if question:
            return f"Theo FAQ của phòng khám về “{question}”: {answer}"
        return answer

    @staticmethod
    def _simple(
        intent: Intent,
        confidence: float,
        answer_type: str,
        text: str,
    ) -> GeneratedResponse:
        return GeneratedResponse(
            intent=intent,
            confidence=confidence,
            answer_type=answer_type,
            result=ResultBody(text=text),
        )


def _json_payload(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped
