# ruff: noqa: E501
"""app/orchestration/query_rewriter.py

Contextual query rewriting: bước chạy ngay sau `memory_load`, trước
`task_planning`/`entity_span_extraction`. Nhận câu hỏi hiện tại + lịch sử
hội thoại thô + danh sách entity gần đây, trả về một câu hỏi ĐỘC LẬP,
đầy đủ nghĩa, đã resolve hết đại từ/tham chiếu ngầm/elip câu.

Nguyên tắc bắt buộc:
- Không tự sinh fact nghiệp vụ (giá, tồn kho, thời lượng...). Model chỉ viết
  lại CÂU HỎI, không trả lời câu hỏi.
- Không phá vỡ nguyên tắc "structured SQL là authoritative cho fact nghiệp vụ"
  đã có trong AGENTS.md — bước này nằm hoàn toàn trước entity resolution/
  retrieval, không thay thế chúng.
- Nếu parse JSON lỗi hoặc model timeout: fallback an toàn là dùng nguyên
  câu hỏi gốc (is_standalone=True), KHÔNG được chặn luồng chat.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.generation.llm_client import LLMClient  # provider-neutral client hiện có
from app.orchestration.query_features import QueryFeatures
from app.retrieval.normalization import normalize_vietnamese

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Bạn là bộ viết lại câu hỏi (query rewriter) cho một chatbot RAG của phòng khám \
nha khoa. Nhiệm vụ DUY NHẤT của bạn là viết lại tin nhắn mới nhất của người \
dùng thành một câu hỏi ĐỘC LẬP, đầy đủ nghĩa, có thể hiểu được mà KHÔNG CẦN \
đọc lịch sử hội thoại. Bạn không được trả lời câu hỏi, không được bịa thêm \
thông tin nghiệp vụ nào không có trong lịch sử.

QUY TẮC:
1. Nếu câu hỏi hiện tại đã độc lập, đầy đủ nghĩa (không có đại từ, không có \
tham chiếu ngầm như "cái đó", "loại kia", "còn...", không bị elip chủ ngữ) \
=> giữ nguyên gần như không đổi, is_standalone=true.
2. Nếu câu hỏi phụ thuộc ngữ cảnh (đại từ, tham chiếu ngầm, câu hỏi rút gọn, \
tên sản phẩm/dịch vụ viết tắt) => viết lại bằng cách thay thế phần phụ \
thuộc đó bằng TÊN ĐẦY ĐỦ, CHÍNH XÁC của entity liên quan gần nhất trong \
lịch sử hội thoại hoặc trong danh sách "entity gần đây" được cung cấp.
3. Nếu có danh sách "entity gần đây", LUÔN dùng đúng tên chính xác trong \
danh sách đó khi viết lại — không tự đặt tên khác, không rút gọn, không \
viết sai chính tả/dấu.
4. Nếu câu hỏi hiện tại tự nêu rõ một entity MỚI, khác với entity đang được \
nhắc tới gần nhất trong lịch sử, PHẢI ưu tiên entity mới người dùng vừa nêu. \
Không được giữ lại entity cũ chỉ vì nó xuất hiện gần đây hơn trong state.
5. Một từ/cụm từ trùng khớp ngẫu nhiên với tên loại dịch vụ/sản phẩm khác \
(ví dụ câu hỏi so sánh 2 sản phẩm nhưng có chữ trùng tên một dịch vụ khác) \
KHÔNG được coi là entity switch, trừ khi người dùng thực sự đang hỏi về \
dịch vụ/sản phẩm đó.
6. Không được thêm số liệu, giá, tình trạng tồn kho, hoặc bất kỳ fact nghiệp \
vụ nào không có sẵn trong lịch sử/câu hỏi gốc.
7. Giữ nguyên ngôn ngữ (tiếng Việt), giữ nguyên ý định (intent) của câu hỏi \
gốc — không đổi câu hỏi giá thành câu hỏi chi tiết sản phẩm hay ngược lại.
8. Nếu thực sự không đủ căn cứ để xác định entity nào đang được nhắc tới \
(mơ hồ thật, không chỉ là thiếu tên đầy đủ) => đặt needs_clarification=true, \
is_standalone=false, và rewritten_query giữ nguyên câu gốc.
9. CHỈ trả về JSON hợp lệ theo đúng schema bên dưới. Không thêm text giải \
thích, không dùng markdown code fence, không thêm ký tự nào trước/sau JSON.

SCHEMA ĐẦU RA (JSON, đúng thứ tự field):
{
  "rewritten_query": "câu hỏi đã viết lại, hoặc câu gốc nếu đã độc lập",
  "is_standalone": true hoặc false,
  "needs_clarification": true hoặc false,
  "referenced_entities": ["tên đầy đủ entity 1", "tên đầy đủ entity 2"]
}
"""


FEW_SHOT_EXAMPLES = """\
### Ví dụ 1
Lịch sử hội thoại:
- user: Cho tôi thông tin về Tẩy Trắng Răng Tại Phòng Khám
- assistant: Tẩy trắng răng tại phòng khám là dịch vụ... giá 2.500.000 VND.
- user: Chi phí bao nhiêu?
- assistant: Chi phí cho dịch vụ tẩy trắng răng tại phòng khám là 2.500.000 VND.
Entity gần đây: [{"name": "Tẩy trắng răng tại phòng khám", "type": "service"}]
Câu hỏi hiện tại: "Sau khi làm xong cần kiêng gì không?"
Kết quả:
{
  "rewritten_query": "Sau khi làm dịch vụ tẩy trắng răng tại phòng khám cần kiêng gì không?",
  "is_standalone": false,
  "needs_clarification": false,
  "referenced_entities": ["Tẩy trắng răng tại phòng khám"]
}

### Ví dụ 2
Lịch sử hội thoại:
- user: So sánh AquaJet Mini Water Flosser và SilkLine Waxed Dental Floss
- assistant: [so sánh 2 sản phẩm...]
- user: Loại nào phù hợp hơn cho người niềng răng?
- assistant: [tư vấn theo 2 sản phẩm...]
Entity gần đây: [
  {"name": "AquaJet Mini Water Flosser", "type": "product"},
  {"name": "SilkLine Waxed Dental Floss", "type": "product"}
]
Câu hỏi hiện tại: "AquaJet giá bao nhiêu?"
Kết quả:
{
  "rewritten_query": "AquaJet Mini Water Flosser giá bao nhiêu?",
  "is_standalone": false,
  "needs_clarification": false,
  "referenced_entities": ["AquaJet Mini Water Flosser"]
}

### Ví dụ 3 (entity mới lấn át entity đang active — không bị hijack bởi từ khóa trùng)
Lịch sử hội thoại:
- user: So sánh AquaJet Mini Water Flosser và SilkLine Waxed Dental Floss
- assistant: [so sánh 2 sản phẩm...]
Entity gần đây: [
  {"name": "AquaJet Mini Water Flosser", "type": "product"},
  {"name": "SilkLine Waxed Dental Floss", "type": "product"}
]
Câu hỏi hiện tại: "Loại nào phù hợp hơn cho người niềng răng?"
Kết quả:
{
  "rewritten_query": "Giữa AquaJet Mini Water Flosser và SilkLine Waxed Dental Floss, loại nào phù hợp hơn cho người niềng răng?",
  "is_standalone": false,
  "needs_clarification": false,
  "referenced_entities": ["AquaJet Mini Water Flosser", "SilkLine Waxed Dental Floss"]
}

### Ví dụ 4 (tham chiếu ngầm dạng "còn... thì sao")
Lịch sử hội thoại:
- user: Giữa AquaJet Mini Water Flosser và SilkLine Waxed Dental Floss,
  loại nào phù hợp hơn cho người niềng răng?
- assistant: [trả lời...]
- user: AquaJet Mini Water Flosser giá bao nhiêu?
- assistant: AquaJet Mini Water Flosser có giá 1.250.000 VND.
Entity gần đây: [
  {"name": "AquaJet Mini Water Flosser", "type": "product"},
  {"name": "SilkLine Waxed Dental Floss", "type": "product"}
]
Câu hỏi hiện tại: "Còn loại kia thì sao?"
Kết quả:
{
  "rewritten_query": "SilkLine Waxed Dental Floss giá bao nhiêu?",
  "is_standalone": false,
  "needs_clarification": false,
  "referenced_entities": ["SilkLine Waxed Dental Floss"]
}

### Ví dụ 5 (câu hỏi đã độc lập, không đổi)
Lịch sử hội thoại:
- user: AquaJet Mini Water Flosser giá bao nhiêu?
- assistant: [trả lời...]
Câu hỏi hiện tại: "Phòng khám mở cửa mấy giờ?"
Kết quả:
{
  "rewritten_query": "Phòng khám mở cửa mấy giờ?",
  "is_standalone": true,
  "needs_clarification": false,
  "referenced_entities": []
}
"""


class RewrittenQuery(BaseModel):
    rewritten_query: str
    is_standalone: bool
    needs_clarification: bool = False
    referenced_entities: list[str] = Field(default_factory=list)


@dataclass
class RecentEntity:
    name: str
    type: str  # "product" | "service" | "faq" | "clinic_info"


@dataclass
class RawTurn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class QueryRewriteInput:
    current_query: str
    turns: list[RawTurn] = field(default_factory=list)  # đã cắt còn N turn gần nhất
    recent_entities: list[RecentEntity] = field(default_factory=list)


def _format_history(turns: list[RawTurn]) -> str:
    lines = [f"- {t.role}: {t.text}" for t in turns]
    return "\n".join(lines) if lines else "(không có lịch sử — đây là turn đầu tiên)"


def _format_entities(entities: list[RecentEntity]) -> str:
    if not entities:
        return "[]"
    return json.dumps(
        [{"name": e.name, "type": e.type} for e in entities],
        ensure_ascii=False,
    )


def build_user_message(payload: QueryRewriteInput) -> str:
    return (
        f"{FEW_SHOT_EXAMPLES}\n"
        "### Yêu cầu thực tế\n"
        f"Lịch sử hội thoại:\n{_format_history(payload.turns)}\n"
        f"Entity gần đây: {_format_entities(payload.recent_entities)}\n"
        f'Câu hỏi hiện tại: "{payload.current_query}"\n'
        "Kết quả:"
    )


def _entity_in_query(query: str, entities: list[RecentEntity]) -> bool:
    normalized_query = normalize_vietnamese(query).lower()
    return any(
        normalize_vietnamese(entity.name).lower() in normalized_query
        for entity in entities
        if entity.name
    )


def _join_entity_names(entities: list[RecentEntity]) -> str:
    names = [entity.name for entity in entities if entity.name]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + f" và {names[-1]}"


def _repair_context_miss(
    rewritten: RewrittenQuery,
    payload: QueryRewriteInput,
) -> RewrittenQuery:
    """Repair only obvious misses where LLM ignored a single recent entity.

    This keeps business truth outside the LLM: the repair only makes the query
    explicit enough for the normal DB-backed entity resolver to accept or reject.
    """
    if (
        rewritten.needs_clarification
        or rewritten.referenced_entities
        or not payload.recent_entities
        or not payload.turns
        or _entity_in_query(payload.current_query, payload.recent_entities)
    ):
        return rewritten

    features = QueryFeatures.extract(payload.current_query)
    if (
        not features.has_implicit_reference
        or features.is_schedule_query
        or (features.asks_list and not features.asks_compare)
    ):
        return rewritten

    if len(payload.recent_entities) == 1:
        entity = payload.recent_entities[0]
        return RewrittenQuery(
            rewritten_query=f"{entity.name}: {payload.current_query}",
            is_standalone=False,
            needs_clarification=False,
            referenced_entities=[entity.name],
        )

    if features.asks_compare:
        entity_names = [entity.name for entity in payload.recent_entities if entity.name]
        return RewrittenQuery(
            rewritten_query=(
                f"Giữa {_join_entity_names(payload.recent_entities)}, "
                f"{payload.current_query}"
            ),
            is_standalone=False,
            needs_clarification=False,
            referenced_entities=entity_names,
        )

    return rewritten


async def rewrite_query(
    llm_client: LLMClient,
    payload: QueryRewriteInput,
    *,
    model: str | None = None,
    timeout_s: float = 8.0,
) -> RewrittenQuery:
    """Gọi LLM để viết lại câu hỏi follow-up. Luôn có fallback an toàn.

    fallback: nếu lỗi bất kỳ (timeout, parse JSON fail, model từ chối...)
    => trả về câu hỏi gốc nguyên văn, is_standalone=True. Không bao giờ raise
    để chặn luồng chat chính — đây chỉ là bước hỗ trợ, không phải nguồn sự thật.
    """
    fallback = RewrittenQuery(
        rewritten_query=payload.current_query,
        is_standalone=True,
        needs_clarification=False,
        referenced_entities=[],
    )

    try:
        response = await llm_client.generate(
            prompt=build_user_message(payload),
            model=model or "",
            system=SYSTEM_PROMPT,
            json_mode=True,
            timeout_seconds=max(1, int(timeout_s)),
            think=False,
        )
        raw = response.text
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
        return _repair_context_miss(RewrittenQuery.model_validate(data), payload)
    except Exception:  # noqa: BLE001 - fallback path phải luôn chạy được
        logger.warning(
            "query_rewrite_failed_fallback_to_original_query",
            extra={"current_query": payload.current_query},
            exc_info=True,
        )
        return fallback
