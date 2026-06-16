from typing import Any

from app.config import Settings
from app.constants import Intent
from app.generation.ollama_client import OllamaClient
from app.generation.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_chitchat_prompt,
    build_generation_prompt,
    build_json_fix_prompt,
)
from app.generation.schemas import (
    EntityReference,
    GeneratedResponse,
    ResultBody,
    ResultItem,
    SafetyInfo,
    SourceReference,
)
from app.generation.validator import ResponseValidationError, ResponseValidator


class GenerationValidationError(ResponseValidationError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


class GroundedGenerator:
    def __init__(self, settings: Settings, ollama: OllamaClient):
        self.settings = settings
        self.ollama = ollama
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
        first = await self.ollama.generate(
            prompt=prompt,
            model=self.settings.ollama_generation_model,
            system=SYSTEM_PROMPT,
            json_mode=True,
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
            fixed = await self.ollama.generate(
                prompt=build_json_fix_prompt(first.text, str(first_error)),
                model=self.settings.ollama_generation_model,
                system=SYSTEM_PROMPT,
                json_mode=True,
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
    ) -> tuple[GeneratedResponse, dict[str, Any]]:
        prompt = build_chitchat_prompt(
            query=query,
            intent=Intent.CHITCHAT,
            confidence=confidence,
        )
        context = {"items": [], "total_chars": 0}
        first = await self.ollama.generate(
            prompt=prompt,
            model=self.settings.ollama_generation_model,
            system=CHITCHAT_SYSTEM_PROMPT,
            json_mode=True,
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
            fixed = await self.ollama.generate(
                prompt=build_json_fix_prompt(first.text, str(first_error)),
                model=self.settings.ollama_generation_model,
                system=CHITCHAT_SYSTEM_PROMPT,
                json_mode=True,
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
        if intent == Intent.GREETING:
            return self._simple(
                intent,
                confidence,
                "greeting",
                "Xin chào! Tôi có thể hỗ trợ bạn về sản phẩm, dịch vụ, FAQ "
                "hoặc thông tin phòng khám.",
            )
        if intent == Intent.CHITCHAT:
            return self._simple(
                intent,
                confidence,
                "chitchat",
                "Cảm ơn bạn. Tôi luôn sẵn sàng hỗ trợ các câu hỏi về nha khoa và phòng khám.",
            )
        if intent == Intent.UNKNOWN:
            return self._simple(
                intent,
                confidence,
                "clarification",
                "Mình chưa hiểu rõ bạn muốn hỏi về sản phẩm, dịch vụ, FAQ hay thông tin "
                "phòng khám. Bạn có thể hỏi cụ thể hơn không?",
            )
        if intent == Intent.PRODUCT_COMPARE and len(items) < 2:
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
                "Hiện tại tôi chưa có đủ thông tin trong dữ liệu của phòng khám.",
            )

        if intent == Intent.FAQ and items[0]["source_type"] == "faq":
            text = str(items[0].get("raw_json", {}).get("answer") or items[0]["text"])
        elif intent in {Intent.PRODUCT_LIST, Intent.SERVICE_LIST}:
            label = "Sản phẩm" if intent == Intent.PRODUCT_LIST else "Dịch vụ"
            text = f"Tìm thấy {len(items)} {label.lower()} phù hợp."
        else:
            text = "\n".join(item["text"] for item in items)

        result_items = [self._result_item(item) for item in items]
        entities = [
            EntityReference(
                type=self._entity_type(item["source_type"]),
                name=str(item.get("raw_json", {}).get("name") or item["source_type"]),
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
                medical_disclaimer_required=intent == Intent.FAQ,
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
        if response.answer_type == "direct_data":
            response.answer_type = "rag"
            response.result.text += (
                "\n\nThông tin trên được tổng hợp trực tiếp từ dữ liệu hiện có của phòng khám."
            )
        return response

    @staticmethod
    def _focused_fallback_context(
        intent: Intent,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        items = list(context.get("items", []))
        preferred_types: set[str] | None = None
        limit: int | None = None
        if intent == Intent.PRODUCT_DETAIL:
            preferred_types = {"product"}
            limit = 1
        elif intent == Intent.SERVICE_DETAIL:
            preferred_types = {"service"}
            limit = 1
        elif intent == Intent.PRODUCT_COMPARE:
            preferred_types = {"product"}
            limit = 2
        elif intent == Intent.FAQ:
            preferred_types = {"faq"}
            limit = 1

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
