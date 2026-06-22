import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz

from app.config import Settings
from app.constants import Intent
from app.observability.logging import get_logger
from app.retrieval.normalization import normalize_vietnamese

logger = get_logger(__name__)


_SUPPORTED_ENTITY_TYPES = {"product", "service", "faq", "clinic_info", "unknown"}
_ENTITY_TYPE_ALIASES = {
    "clinic": "clinic_info",
    "clinic_fact": "clinic_info",
    "condition": "unknown",
    "product_category": "product",
    "service_category": "service",
    "symptom": "unknown",
    "symptom_or_condition": "unknown",
    "topic": "unknown",
}


class RouterEntity(BaseModel):
    type: str = "unknown"
    name: str | None = None
    role: str | None = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> str:
        normalized = str(value or "unknown").strip().lower()
        normalized = _ENTITY_TYPE_ALIASES.get(normalized, normalized)
        if normalized not in _SUPPORTED_ENTITY_TYPES:
            return "unknown"
        return normalized

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class RouterLLMOutput(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    entities: list[RouterEntity | str] = Field(default_factory=list)
    question_type: str | None = None
    answer_strategy: str | None = None
    needs_rag: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    reason_code: str | None = None

    @property
    def entity_names(self) -> list[str]:
        names: list[str] = []
        for entity in self.entities:
            name = entity.strip() if isinstance(entity, str) else entity.name
            if name and name not in names:
                names.append(name)
        return names

    @property
    def entity_details(self) -> list[dict[str, str | None]]:
        details: list[dict[str, str | None]] = []
        for entity in self.entities:
            if isinstance(entity, str):
                details.append({"type": "unknown", "name": entity, "role": None})
            else:
                details.append(entity.model_dump(mode="json"))
        return details


@dataclass
class RouterResult:
    intent: Intent
    confidence: float
    entities: list[str] = field(default_factory=list)
    needs_rag: bool = False
    needs_clarification: bool = False
    source: str = "rules"
    llm_attempted: bool = False
    llm_latency_ms: int | None = None
    fallback_reason: str | None = None
    clarification_message: str | None = None
    question_type: str | None = None
    answer_strategy: str | None = None
    reason_code: str | None = None
    entity_details: list[dict[str, Any]] = field(default_factory=list)
    llm_prompt_chars: int | None = None
    llm_raw_output: str | None = None
    llm_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "entities": self.entities,
            "needs_rag": self.needs_rag,
            "needs_clarification": self.needs_clarification,
            "source": self.source,
            "llm_attempted": self.llm_attempted,
            "llm_latency_ms": self.llm_latency_ms,
            "fallback_reason": self.fallback_reason,
            "clarification_message": self.clarification_message,
            "question_type": self.question_type,
            "answer_strategy": self.answer_strategy,
            "reason_code": self.reason_code,
            "entity_details": self.entity_details,
            "llm_prompt_chars": self.llm_prompt_chars,
            "llm_raw_output": self.llm_raw_output,
            "llm_metadata": self.llm_metadata,
        }


class IntentRouter:
    def __init__(self) -> None:
        self._consecutive_llm_failures = 0
        self._circuit_open_until = 0.0

    GREETINGS = {
        "hi",
        "hello",
        "xin chao",
        "chao",
        "chao ban",
        "alo",
    }
    COMPARE_WORDS = (
        "so sanh",
        "khac nhau",
        " khac ",
        " vs ",
        " voi ",
        "tot hon",
    )
    LIST_WORDS = (
        "danh sach",
        "bang",
        "co nhung",
        "tat ca",
        "liet ke",
        "xem cac",
        "san pham nao",
        "dich vu nao",
        "nhung san pham",
        "nhung dich vu",
        "dang ban",
        "hien co",
        "lien quan den",
        "phu hop voi",
        "thuoc loai",
    )
    PRODUCT_WORDS = (
        "san pham",
        "ban chai",
        "kem danh rang",
        "chi nha khoa",
        "oral-b",
        "tam nuoc",
    )
    SERVICE_WORDS = (
        "dich vu",
        "tay trang",
        "nho rang",
        "nieng rang",
        "trong rang",
        "implant",
        "cao voi",
    )
    CLINIC_WORDS = (
        "dia chi",
        "so dien thoai",
        "hotline",
        "email",
        "gio lam",
        "mo cua",
        "facebook",
        "zalo",
        "phong kham o dau",
    )
    FAQ_WORDS = (
        "dau rang",
        "chay mau",
        "sung",
        "e buot",
        "bao lau",
        "co phai",
        "co dau khong",
        "co an toan",
        "co lam",
        "co can",
        "co tot",
        "co hai",
        "co mon",
        "hoi mieng",
        "sau rang",
        "nguy hiem",
        "lam gi",
        "nen lam gi",
        "sau khi",
        "truoc khi",
        "tai sao",
    )
    DENTAL_WORDS = ("rang", "nuou", "loi", "nha khoa", "phong kham", "mieng")
    CHITCHAT_WORDS = (
        "cam on",
        "ban la ai",
        "khoe khong",
        "tam biet",
        "huu ich",
    )

    def route(
        self,
        query: str,
        *,
        known_products: list[str] | None = None,
        known_services: list[str] | None = None,
        known_product_categories: list[str] | None = None,
    ) -> RouterResult:
        normalized = self._normalize(query)
        if normalized in self.GREETINGS or any(
            normalized.startswith(f"{greeting} ") for greeting in self.GREETINGS
        ):
            return RouterResult(Intent.GREETING, 0.99)
        if any(word in normalized for word in self.CHITCHAT_WORDS):
            return RouterResult(Intent.CHITCHAT, 0.94)
        if any(word in normalized for word in self.CLINIC_WORDS):
            return RouterResult(Intent.CLINIC_INFO, 0.96)

        has_product = any(word in normalized for word in self.PRODUCT_WORDS)
        has_service = any(word in normalized for word in self.SERVICE_WORDS)
        product_entities = self._known_mentions(normalized, known_products or [])
        service_entities = self._known_mentions(normalized, known_services or [])
        category_entities = self._known_mentions(
            normalized,
            known_product_categories or [],
        )
        has_product = has_product or bool(product_entities)
        has_product = has_product or bool(category_entities)
        has_service = has_service or bool(service_entities)
        is_compare = any(word in f" {normalized} " for word in self.COMPARE_WORDS)
        is_list = any(word in normalized for word in self.LIST_WORDS) or any(
            word in normalized
            for word in ("thap den cao", "cao den thap", "tang dan", "giam dan", "loc")
        )
        has_filter_or_sort = any(
            word in normalized
            for word in (
                "duoi",
                "tren",
                "toi da",
                "toi thieu",
                "khong qua",
                "nho hon",
                "re nhat",
                "dat nhat",
                "ngan nhat",
                "lau nhat",
                "sap xep",
                "thu tu",
            )
        )
        is_faq = any(word in normalized for word in self.FAQ_WORDS)
        is_post_treatment = any(
            phrase in normalized
            for phrase in (
                "sau nho",
                "sau khi nho",
                "sau cay",
                "sau khi cay",
                "sau dieu tri",
            )
        ) and any(
            phrase in normalized
            for phrase in (
                "an uong",
                "cham soc",
                "kieng",
                "ve sinh",
                "lam gi",
            )
        )
        is_faq = is_faq or is_post_treatment
        asks_structured_detail = any(
            word in normalized
            for word in (
                "gia",
                "chi phi",
                "bao nhieu",
                "mat bao lau",
                "thoi gian",
            )
        )

        if is_compare and has_product:
            return RouterResult(
                Intent.PRODUCT_COMPARE,
                0.95,
                entities=product_entities,
                needs_rag=False,
            )
        if is_faq and not asks_structured_detail:
            return RouterResult(Intent.FAQ, 0.84, needs_rag=True)
        if has_product and (is_list or (has_filter_or_sort and not product_entities)):
            return RouterResult(Intent.PRODUCT_LIST, 0.97)
        if has_service and (is_list or (has_filter_or_sort and not service_entities)):
            return RouterResult(Intent.SERVICE_LIST, 0.97)
        if has_product:
            return RouterResult(
                Intent.PRODUCT_DETAIL,
                0.91 if product_entities else 0.86,
                entities=product_entities,
            )
        if has_service:
            return RouterResult(
                Intent.SERVICE_DETAIL,
                0.91 if service_entities else 0.86,
                entities=service_entities,
            )
        if is_faq or (
            normalized.endswith("?") and any(word in normalized for word in self.DENTAL_WORDS)
        ):
            return RouterResult(Intent.FAQ, 0.78, needs_rag=True)
        return RouterResult(
            Intent.UNKNOWN,
            0.4,
            needs_clarification=True,
        )

    async def route_with_optional_llm(
        self,
        query: str,
        settings: Settings,
        llm_client,
        *,
        known_products: list[str] | None = None,
        known_services: list[str] | None = None,
        known_product_categories: list[str] | None = None,
    ) -> RouterResult:
        if not settings.enable_llm_router:
            return self.route(
                query,
                known_products=known_products,
                known_services=known_services,
                known_product_categories=known_product_categories,
            )

        prompt = self._llm_prompt(
            query,
            known_products or [],
            known_services or [],
            known_product_categories or [],
        )
        started = time.perf_counter()
        if time.monotonic() < self._circuit_open_until:
            baseline = self._safe_fallback_route(query)
            baseline.source = "rules_circuit_open"
            baseline.fallback_reason = "RouterLLM circuit breaker is open"
            baseline.llm_prompt_chars = len(prompt)
            baseline.llm_metadata = {
                "circuit_open_until": self._circuit_open_until,
                "fallback_kind": "safe_fallback",
            }
            return baseline

        raw_outputs: list[str] = []
        attempts: list[dict[str, Any]] = []
        validation_meta: dict[str, Any] = {"retried": False}
        try:
            response = await llm_client.generate(
                prompt=prompt,
                model=settings.llm_router_model,
                system=ROUTER_SYSTEM_PROMPT,
                json_mode=True,
                timeout_seconds=settings.llm_router_timeout_seconds,
                num_predict=settings.llm_router_num_predict,
                num_ctx=settings.llm_router_num_ctx,
                think=False,
            )
            raw_outputs.append(response.text)
            attempts.append(response.trace_metadata())
            try:
                payload = self._parse_llm_output(response.text)
            except Exception as parse_exc:
                validation_meta = {
                    "retried": True,
                    "first_error": str(parse_exc),
                }
                fix_response = await llm_client.generate(
                    prompt=self._json_fix_prompt(response.text, str(parse_exc)),
                    model=settings.llm_router_model,
                    system=ROUTER_SYSTEM_PROMPT,
                    json_mode=True,
                    timeout_seconds=settings.llm_router_timeout_seconds,
                    num_predict=settings.llm_router_num_predict,
                    num_ctx=settings.llm_router_num_ctx,
                    think=False,
                )
                raw_outputs.append(fix_response.text)
                attempts.append(fix_response.trace_metadata())
                response = fix_response
                payload = self._parse_llm_output(response.text)
            self._consecutive_llm_failures = 0
            self._circuit_open_until = 0.0
            llm_latency_ms = int((time.perf_counter() - started) * 1000)
            llm_metadata = {
                **response.trace_metadata(),
                "attempts": attempts,
                "raw_outputs": raw_outputs,
                "validation": validation_meta,
            }
            return RouterResult(
                intent=payload.intent,
                confidence=payload.confidence,
                entities=payload.entity_names,
                needs_rag=payload.needs_rag,
                needs_clarification=(
                    payload.needs_clarification or payload.confidence < 0.65
                ),
                clarification_message=payload.clarification_question,
                source=settings.llm_provider.lower().strip(),
                llm_attempted=True,
                llm_latency_ms=llm_latency_ms,
                question_type=payload.question_type,
                answer_strategy=payload.answer_strategy,
                reason_code=payload.reason_code,
                entity_details=payload.entity_details,
                llm_prompt_chars=len(prompt),
                llm_raw_output=response.text,
                llm_metadata=llm_metadata,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self._consecutive_llm_failures += 1
            if self._consecutive_llm_failures >= settings.router_failure_threshold:
                self._circuit_open_until = (
                    time.monotonic() + settings.router_circuit_breaker_seconds
                )
            logger.warning("RouterLLM failed; using safe fallback: %s", exc)
            baseline = self._safe_fallback_route(query)
            baseline.source = "rules_fallback"
            baseline.llm_attempted = True
            baseline.llm_latency_ms = latency_ms
            baseline.fallback_reason = str(exc)
            baseline.llm_prompt_chars = len(prompt)
            baseline.llm_raw_output = raw_outputs[-1] if raw_outputs else None
            baseline.llm_metadata = {
                "attempts": attempts,
                "raw_outputs": raw_outputs,
                "validation": validation_meta,
                "fallback_kind": "safe_fallback",
            }
            return baseline

    @staticmethod
    def _parse_llm_output(text: str) -> RouterLLMOutput:
        return RouterLLMOutput.model_validate(json.loads(IntentRouter._json_payload(text)))

    @staticmethod
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

    @staticmethod
    def _json_fix_prompt(raw_output: str, error: str) -> str:
        schema = {
            "intent": "FAQ",
            "confidence": 0.92,
            "entities": [
                {"type": "service", "name": "Cạo vôi răng", "role": "topic"}
            ],
            "question_type": "knowledge_or_advice",
            "answer_strategy": "faq_retrieval",
            "needs_rag": True,
            "needs_clarification": False,
            "clarification_question": None,
            "reason_code": "fixed_router_json",
        }
        return (
            "Output router trước đó không parse được. "
            "Hãy chuyển nó thành đúng một JSON object hợp lệ theo schema, "
            "không thêm markdown hay giải thích.\n"
            f"Lỗi parse/validation: {error}\n"
            f"Schema mẫu: {json.dumps(schema, ensure_ascii=False)}\n"
            f"Output cần sửa:\n{raw_output}"
        )

    @staticmethod
    def _llm_prompt(
        query: str,
        known_products: list[str],
        known_services: list[str],
        known_product_categories: list[str],
    ) -> str:
        products = (
            ", ".join(IntentRouter._relevant_terms(query, known_products))
            or "(không cần thiết cho query này)"
        )
        services = (
            ", ".join(IntentRouter._relevant_terms(query, known_services))
            or "(không cần thiết cho query này)"
        )
        categories = (
            ", ".join(IntentRouter._relevant_terms(query, known_product_categories))
            or "(không cần thiết cho query này)"
        )
        intents = ", ".join(intent.value for intent in Intent)
        schema = {
            "intent": "FAQ",
            "confidence": 0.92,
            "entities": [
                {"type": "service", "name": "Cạo vôi răng", "role": "topic"}
            ],
            "question_type": "knowledge_or_advice",
            "answer_strategy": "faq_retrieval",
            "needs_rag": True,
            "needs_clarification": False,
            "clarification_question": None,
            "reason_code": "service_entity_with_risk_question",
        }
        return (
            f"Intent hợp lệ: {intents}.\n"
            "Bạn phải phân loại intent cho chatbot phòng khám nha khoa. "
            "Chỉ trả về một JSON object hợp lệ, không markdown, không giải thích.\n\n"
            "Phân biệt theo MỤC ĐÍCH của câu hỏi, không chỉ theo entity xuất hiện:\n"
            "- GREETING: lời chào ngắn.\n"
            "- CHITCHAT: xã giao, cảm ơn, hỏi bạn là ai, trò chuyện không cần dữ liệu phòng khám.\n"
            "- CLINIC_INFO: địa chỉ, hotline, email, giờ mở cửa/làm việc của phòng khám.\n"
            "- PRODUCT_LIST: hỏi danh sách/bảng/lọc/sắp xếp các sản phẩm.\n"
            "- SERVICE_LIST: hỏi danh sách/bảng/lọc/sắp xếp các dịch vụ.\n"
            "- PRODUCT_DETAIL: hỏi giá, còn hàng, số lượng, link, mô tả "
            "hoặc chi tiết một sản phẩm cụ thể.\n"
            "- SERVICE_DETAIL: hỏi giá, thời lượng, mô tả, chỉ định/chống chỉ định "
            "hoặc chi tiết một dịch vụ cụ thể.\n"
            "- PRODUCT_COMPARE: so sánh ít nhất hai sản phẩm cụ thể.\n"
            "- FAQ: hỏi kiến thức, rủi ro, tác dụng phụ, cách dùng/chăm sóc, hậu điều trị, "
            "có đau không, có tốt không, có hại không, có làm yếu/mòn không, nên làm gì. "
            "Nếu câu có tên sản phẩm/dịch vụ nhưng hỏi các vấn đề này thì vẫn là FAQ.\n"
            "- UNKNOWN: ngoài phạm vi nha khoa/phòng khám hoặc thiếu thông tin để phân loại.\n\n"
            "Quy tắc quan trọng:\n"
            "- Entity chỉ cho biết câu hỏi nói về cái gì; intent phải dựa vào "
            "việc user muốn làm gì.\n"
            "- 'Cạo vôi răng có làm yếu răng không' là FAQ, không phải SERVICE_DETAIL.\n"
            "- 'Dịch vụ cạo vôi răng giá bao nhiêu' là SERVICE_DETAIL.\n"
            "- 'Kem đánh răng làm trắng có làm mòn men răng không' là FAQ, "
            "không phải PRODUCT_DETAIL.\n"
            "- 'dich vu nho rang khon co dau khong' là FAQ nếu hỏi có đau không; "
            "là SERVICE_DETAIL nếu hỏi giá/thời lượng/chi tiết dịch vụ.\n"
            "- Nếu cần hỏi lại, đặt needs_clarification=true và viết clarification_question.\n\n"
            "answer_strategy hợp lệ nên là một trong: template, no_rag_llm, direct_sql, "
            "structured_sql, faq_retrieval, structured_then_llm, clarify.\n"
            f"Schema ví dụ:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Sản phẩm active liên quan: {products}\n"
            f"Danh mục sản phẩm active liên quan: {categories}\n"
            f"Dịch vụ active liên quan: {services}\n"
            f"Query cần phân loại: {query}"
        )

    def _safe_fallback_route(self, query: str) -> RouterResult:
        normalized = self._normalize(query)
        padded = f" {normalized} "
        if normalized in self.GREETINGS or any(
            normalized.startswith(f"{greeting} ") for greeting in self.GREETINGS
        ):
            return RouterResult(Intent.GREETING, 0.82)
        if any(word in normalized for word in self.CHITCHAT_WORDS):
            return RouterResult(Intent.CHITCHAT, 0.76)
        if any(word in normalized for word in self.CLINIC_WORDS):
            return RouterResult(Intent.CLINIC_INFO, 0.78)

        is_list = any(word in normalized for word in self.LIST_WORDS) or any(
            word in normalized
            for word in ("thap den cao", "cao den thap", "tang dan", "giam dan", "loc")
        )
        has_filter_or_sort = any(
            word in normalized
            for word in (
                "duoi",
                "tren",
                "toi da",
                "toi thieu",
                "khong qua",
                "nho hon",
                "re nhat",
                "dat nhat",
                "ngan nhat",
                "lau nhat",
                "sap xep",
                "thu tu",
            )
        )
        has_product_word = any(word in normalized for word in self.PRODUCT_WORDS)
        has_service_word = any(word in normalized for word in self.SERVICE_WORDS)
        if (is_list or has_filter_or_sort) and has_product_word:
            return RouterResult(Intent.PRODUCT_LIST, 0.78)
        if (is_list or has_filter_or_sort) and has_service_word:
            return RouterResult(Intent.SERVICE_LIST, 0.78)

        if self._looks_like_faq(normalized, padded):
            return RouterResult(Intent.FAQ, 0.72, needs_rag=True)

        return RouterResult(
            Intent.UNKNOWN,
            0.35,
            needs_clarification=True,
            clarification_message=(
                "Bạn có thể nói rõ hơn bạn muốn hỏi về sản phẩm, dịch vụ, "
                "FAQ hay thông tin phòng khám không?"
            ),
        )

    def _looks_like_faq(self, normalized: str, padded: str) -> bool:
        asks_structured_detail = any(
            word in normalized
            for word in (
                "gia",
                "chi phi",
                "bao nhieu",
                "mat bao lau",
                "thoi gian",
                "con hang",
                "so luong",
            )
        )
        if asks_structured_detail:
            return False
        has_dental_signal = any(
            word in normalized
            for word in (*self.DENTAL_WORDS, *self.PRODUCT_WORDS, *self.SERVICE_WORDS)
        )
        if not has_dental_signal:
            return False
        has_faq_pattern = any(word in normalized for word in self.FAQ_WORDS) or (
            " co " in padded and " khong " in padded
        )
        return has_faq_pattern

    @staticmethod
    def _normalize(query: str) -> str:
        return normalize_vietnamese(query)

    @staticmethod
    def _known_mentions(query: str, names: list[str]) -> list[str]:
        direct = list(
            dict.fromkeys(
                name for name in names if normalize_vietnamese(name) in query
            )
        )
        if direct:
            return direct

        query_tokens = set(re.findall(r"\w+", query))
        ranked = sorted(
            (
                (
                    name,
                    fuzz.WRatio(normalize_vietnamese(name), query),
                    len(query_tokens & set(re.findall(r"\w+", normalize_vietnamese(name)))),
                )
                for name in names
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            name
            for name, score, overlap in ranked[:2]
            if score >= 70 and overlap >= 2
        ]

    @staticmethod
    def _relevant_terms(query: str, names: list[str], limit: int = 12) -> list[str]:
        normalized_query = normalize_vietnamese(query)
        query_tokens = set(re.findall(r"\w+", normalized_query))
        ranked: list[tuple[str, int, int, bool]] = []
        for name in dict.fromkeys(names):
            normalized_name = normalize_vietnamese(name)
            name_tokens = set(re.findall(r"\w+", normalized_name))
            overlap = len(query_tokens & name_tokens)
            direct = bool(normalized_name and normalized_name in normalized_query)
            score = fuzz.WRatio(normalized_name, normalized_query)
            if direct or overlap >= 1 or score >= 62:
                ranked.append((name, score, overlap, direct))
        ranked.sort(key=lambda item: (item[3], item[2], item[1]), reverse=True)
        return [name for name, *_ in ranked[:limit]]


ROUTER_SYSTEM_PROMPT = (
    "Bạn là intent router cho chatbot nha khoa. "
    "Nhiệm vụ duy nhất là phân loại intent và trả JSON đúng schema. "
    "Không trả lời câu hỏi của user ở bước router."
)
