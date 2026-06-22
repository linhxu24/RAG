import json
from typing import Any

from app.constants import Intent

SYSTEM_PROMPT = """Bạn là chatbot hỗ trợ khách hàng cho phòng khám nha khoa.
Bạn chỉ được trả lời dựa trên dữ liệu được cung cấp trong retrieved_context.
Không tự bịa giá, dịch vụ, sản phẩm, lịch làm việc, chính sách hoặc thông tin y tế.
Nếu dữ liệu không có, hãy nói: "Hiện tại tôi chưa có đủ thông tin trong dữ liệu của phòng khám."
Với câu hỏi về sản phẩm/dịch vụ, ưu tiên trả lời ngắn gọn, có cấu trúc.
Với câu hỏi so sánh, chỉ so sánh các thuộc tính có trong dữ liệu.
Với câu hỏi y tế, chỉ tư vấn định hướng, không chẩn đoán chắc chắn,
và khuyến nghị gặp nha sĩ nếu có dấu hiệu nguy hiểm.
Giữ nguyên asset token dạng [asset:...] trong phần text.
Trả về JSON hợp lệ theo schema được yêu cầu.
Không viết markdown bên ngoài JSON."""


CHITCHAT_SYSTEM_PROMPT = """Bạn là chatbot hỗ trợ khách hàng cho phòng khám nha khoa.
Bạn đang trả lời một câu xã giao hoặc trò chuyện ngắn, không cần truy xuất dữ liệu.
Trả lời tự nhiên, lịch sự, ngắn gọn bằng tiếng Việt.
Không bịa giá, lịch làm việc, sản phẩm, dịch vụ, chính sách hoặc thông tin y tế cụ thể.
Nếu user hỏi sang dữ liệu phòng khám, hãy nói bạn có thể tra cứu khi họ hỏi cụ thể.
Trả về JSON hợp lệ theo schema được yêu cầu.
Không viết markdown bên ngoài JSON."""


SYNTHESIS_SYSTEM_PROMPT = """Bạn là chatbot hỗ trợ khách hàng cho phòng khám nha khoa.
Bạn chỉ được trả lời dựa trên evidence_pack được cung cấp.
Evidence có thể đến từ SQL nghiệp vụ, FAQ đã duyệt, chunks tài liệu, table rows hoặc clinic info.
Ưu tiên evidence có trust_level='authoritative' cho giá, số lượng, thời lượng và giờ mở cửa.
Nếu evidence thiếu hoặc mâu thuẫn, nói rõ phần thiếu hoặc cần phòng khám xác nhận.
Không tự bịa giá, dịch vụ, sản phẩm, lịch làm việc, chính sách hoặc thông tin y tế.
Với câu hỏi y tế, chỉ tư vấn định hướng, không chẩn đoán chắc chắn.
Mỗi khẳng định cụ thể phải được nêu trực tiếp trong evidence của đúng task.
Không suy rộng thông tin từ điều trị khác. Nếu evidence chỉ nói ê buốt sau điều trị,
không được biến thành kết luận về đau trong lúc điều trị.
Không viết chain-of-thought. Không dùng thẻ <think>.
Không tự tạo source ID, asset ID hoặc asset token.
Trả về JSON hợp lệ theo schema được yêu cầu.
Không viết markdown bên ngoài JSON."""


def build_chitchat_prompt(
    *,
    query: str,
    intent: Intent,
    confidence: float,
) -> str:
    answer_type = "greeting" if intent == Intent.GREETING else "chitchat"
    schema = {
        "intent": intent.value,
        "confidence": confidence,
        "answer_type": answer_type,
        "entities": [],
        "result": {
            "text": "",
            "items": [],
            "assets": [],
            "sources": [],
            "missing_assets": [],
        },
        "safety": {
            "medical_disclaimer_required": False,
            "needs_human_support": False,
        },
    }
    return (
        f"user_query:\n{query}\n\n"
        f"detected_intent: {intent.value}\n"
        f"router_confidence: {confidence}\n\n"
        "Return exactly one JSON object following this shape. "
        f"Use answer_type='{answer_type}'. Keep result.items, result.assets, "
        "result.sources and result.missing_assets as empty arrays:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def build_synthesis_prompt(
    *,
    query: str,
    intent: Intent,
    confidence: float,
    evidence_pack: dict[str, Any],
) -> str:
    schema = {
        "answer": "",
        "used_source_ids": ["source_id_from_evidence"],
        "medical_disclaimer_required": False,
        "needs_human_support": False,
    }
    return (
        f"user_query:\n{query}\n\n"
        f"primary_intent: {intent.value}\n"
        f"confidence: {confidence}\n\n"
        f"evidence_pack:\n{json.dumps(evidence_pack, ensure_ascii=False, default=str)}\n\n"
        "Hãy tổng hợp câu trả lời tự nhiên từ evidence_pack. Nếu có nhiều task, trả lời đủ từng ý. "
        "used_source_ids chỉ được chứa source_id xuất hiện trong evidence. "
        "Server sẽ tự xây entities, items, sources và assets; không đưa các trường đó vào output. "
        "Return exactly one JSON object following this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def build_synthesis_json_fix_prompt(raw_text: str, validation_error: str) -> str:
    schema = {
        "answer": "",
        "used_source_ids": [],
        "medical_disclaimer_required": False,
        "needs_human_support": False,
    }
    return (
        "Sửa output thành đúng một JSON object theo schema dưới đây. "
        "Không thêm dữ kiện hoặc source ID mới. Không viết markdown.\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n"
        f"Lỗi: {validation_error}\n"
        f"Output:\n{raw_text}"
    )


def build_generation_prompt(
    *,
    query: str,
    intent: Intent,
    confidence: float,
    entities: list[str],
    context: dict[str, Any],
) -> str:
    schema = {
        "intent": intent.value,
        "confidence": confidence,
        "answer_type": "rag",
        "entities": [
            {"type": "product|service|faq|clinic_info|unknown", "name": "", "matched_id": None}
        ],
        "result": {
            "text": "",
            "items": [
                {
                    "type": "",
                    "id": "",
                    "name": None,
                    "chunk_id": None,
                    "row_id": None,
                    "doc_id": None,
                    "asset_ids": [],
                    "data": {},
                }
            ],
            "assets": [],
            "sources": [],
            "missing_assets": [],
        },
        "safety": {
            "medical_disclaimer_required": False,
            "needs_human_support": False,
        },
    }
    return (
        f"user_query:\n{query}\n\n"
        f"detected_intent: {intent.value}\n"
        f"router_confidence: {confidence}\n"
        f"entities: {json.dumps(entities, ensure_ascii=False)}\n\n"
        f"retrieved_context:\n{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
        "Return exactly one JSON object following this shape. Use only source IDs that appear "
        "in retrieved_context. Keep result.assets and result.missing_assets as empty arrays; "
        "the server resolves asset tokens after generation. result.items[].asset_ids may contain "
        "only UUID values copied from retrieved_context.raw_json.asset_id, never an asset token, "
        f"token suffix, or object:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def build_json_fix_prompt(raw_text: str, validation_error: str) -> str:
    return (
        "Fix the following output into valid JSON matching the requested schema. "
        "Do not add facts or IDs. result.assets and result.missing_assets must be empty arrays. "
        "result.items[].asset_ids must contain only source UUID strings. Return JSON only.\n"
        f"Validation error: {validation_error}\n"
        f"Output:\n{raw_text}"
    )
