# PRODUCT_VISION.md — Multimodal RAG Chatbot với Asset Masking

## Tầm nhìn sản phẩm

Xây dựng một **Multimodal RAG Chatbot** cho doanh nghiệp nha khoa có khả năng:
- Trả lời mọi câu hỏi của người dùng dựa hoàn toàn vào dữ liệu nội bộ (không hallucinate)
- Xử lý đúng mọi intent hiện có và dễ dàng mở rộng thêm intent mới
- Trả lời dưới dạng đa phương tiện: text, bảng dữ liệu, và hình ảnh (asset)
- Vận hành production với OpenAI API, dùng Ollama chỉ cho development/testing

---

## Nguyên tắc cốt lõi (không được vi phạm)

### 1. Dữ liệu nội bộ là nguồn sự thật duy nhất
- Mọi câu trả lời về giá, giờ mở cửa, sản phẩm, dịch vụ PHẢI đến từ PostgreSQL
- LLM chỉ được phép diễn đạt lại bằng ngôn ngữ tự nhiên — không được tự bịa thêm facts
- Khi không có dữ liệu: nói thẳng "không có thông tin" thay vì đoán mò

### 2. Intent-driven, không phải semantic-driven
- Pipeline xử lý theo intent (GREETING, FAQ, PRODUCT_DETAIL, v.v.), không phải theo từng câu cụ thể
- Khi thêm intent mới, chỉ cần định nghĩa intent trong IntentCapabilityRegistry —
  pipeline tự động handle mà không cần thêm if-else cho từng trường hợp
- Không được viết code xử lý riêng cho từng câu hỏi cụ thể (ví dụ: if "còn hàng không" → ...)
  Phải xử lý theo pattern tổng quát của intent

### 3. Multimodal output là first-class citizen
- Response có thể chứa đồng thời: text (answer), table (items), ảnh (assets)
- Hình ảnh được resolve từ asset tokens `[asset:<hash>]` trong chunks/table rows
- Frontend phải render đủ 3 loại output — không chỉ hiển thị text
- Asset resolution không phải afterthought — phải chạy trước khi trả response

### 4. Pipeline tổng quát, mở rộng được
- Thêm intent mới: định nghĩa capability trong intent_registry.py → tự hoạt động
- Thêm retrieval method mới: implement retriever interface → plug vào tool_executor
- Thêm document type mới: extend ingestion/pipeline.py → tự ingest
- Không được có code phụ thuộc vào tên entity cụ thể hay nội dung document cụ thể

### 5. LLM provider là config, không phải hard-code
- Production: OpenAI API (GPT-4.1 cho generation, GPT-4.1-nano cho routing)
- Development/testing: Ollama local (qwen2.5 models)
- Switch provider bằng LLM_PROVIDER env var — không cần sửa code
- Ollama timeout không được làm crash pipeline — phải có fallback graceful

---

## Kiến trúc pipeline (tổng quát, không phụ thuộc intent cụ thể)

```
User Query
    │
    ▼
[Memory Load] ──── Conversation history + entity state
    │
    ▼
[Task Planning] ── LLM decomposes query thành 1..N tasks
    │               Mỗi task có intent riêng, entity riêng
    ▼
[NER Extraction] ── GLiNER extract entity spans từ query gốc
    │
    ▼
[Context Binding] ── Resolve entity reference theo priority:
    │                  explicit span > same-turn resolved > memory > planner proposal
    ▼
[Entity Resolution] ── Match entity name → authoritative DB ID
    │
    ▼
[Consistency Gate] ── Validate BoundTask trước khi execute tools
    │
    ▼
[Tool Execution] ──── Mỗi intent có allowed_tools riêng (từ IntentCapabilityRegistry)
    │                  product_tool → SQL products table
    │                  service_tool → SQL services table
    │                  clinic_info_tool → SQL clinic_info table
    │                  faq_tool → FAQ search
    │                  document_rag_tool → Dense + Sparse + RRF + Rerank
    ▼
[Evidence Merge] ──── Dedup, prioritize authoritative, check conflicts
    │
    ▼
[Evidence Gate] ───── Validate evidence trước synthesis
    │
    ▼
[Context Builder] ─── Build prompt context từ evidence (token budget aware)
    │
    ▼
[LLM Generation] ──── OpenAI (production) / Ollama (dev)
    │                  Input: effective_query + evidence context
    │                  Output: answer text + used_source_ids + safety flags
    ▼
[Validation] ──────── Check: entity grounding, price grounding, no hallucination
    │
    ▼
[Asset Resolution] ── Resolve [asset:<hash>] tokens → real URLs
    │
    ▼
[Response Rendering] ─ Build final response:
                         answer.text (LLM output)
                         answer.items (table rows / structured data)
                         answer.assets (resolved image URLs)
                         answer.sources (evidence citations)
```

---

## Output types (multimodal)

### Text
- Mọi intent đều có text answer
- GREETING/CHITCHAT: LLM generation không có RAG context
- Các intent khác: LLM synthesis từ evidence

### Table (items)
- PRODUCT_LIST, SERVICE_LIST: danh sách records từ SQL
- PRODUCT_COMPARE: side-by-side comparison rows
- PRODUCT_DETAIL, SERVICE_DETAIL: single record với đầy đủ fields
- Format: `answer.items[]` — mỗi item có name, price, description, asset_ids

### Image (assets)
- Tự động resolve từ `[asset:<hash>]` tokens trong evidence
- Mỗi item có thể có `asset_ids[]` → frontend render ảnh kèm data
- Clinic info, FAQ cũng có thể có assets nếu document gốc có hình

---

## LLM provider strategy

### Production (OpenAI)
```
LLM_PROVIDER=openai
OPENAI_ROUTER_MODEL=gpt-4.1-nano      # routing + planning (nhanh, rẻ)
OPENAI_GENERATION_MODEL=gpt-4.1       # synthesis (chất lượng cao)
```

### Development (Ollama)
```
LLM_PROVIDER=ollama
OLLAMA_ROUTER_MODEL=qwen2.5:7b-instruct
OLLAMA_GENERATION_MODEL=qwen2.5:14b-instruct-q4_K_M
```

**Quan trọng về Ollama:**
- Ollama chỉ là development fallback — không phải production path
- Ollama timeout phải có graceful fallback, không crash pipeline
- Audit và test với OpenAI timeout ngắn hơn để phản ánh production latency
- `audit_chatbot_completeness.py` phải dùng OpenAI khi chạy để tránh null results do Ollama timeout

---

## Intent extensibility contract

Khi thêm intent mới (ví dụ APPOINTMENT_BOOKING):
1. Thêm `APPOINTMENT_BOOKING` vào `app/constants.py`
2. Định nghĩa `IntentCapability` trong `app/orchestration/intent_registry.py`:
   - entity_scope, inheritance_rule, allowed_tools, evidence_contract
3. Implement tool nếu cần, đăng ký trong `tool_executor.py`
4. Thêm business logic trong `retrieval/structured_query.py` nếu cần SQL filter mới
5. Chạy `validate_intent_registry()` — tự động phát hiện missing coverage

Không được:
- Thêm if-else trong chat.py cho intent mới
- Hard-code intent name trong tool_executor.py
- Viết separate handler cho từng intent mới

---

## Open source ưu tiên

Thay vì tự code từ đầu, ưu tiên dùng các thư viện đã được kiểm chứng:

| Nhiệm vụ | Thư viện ưu tiên |
|---|---|
| Document parsing | Docling |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embedding | BAAI/bge-m3 (sentence-transformers) |
| NER | GLiNER (urchade/gliner_multi-v2.1) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Vector search | pgvector |
| Full-text search | PostgreSQL tsvector/tsquery |
| Observability | Langfuse (optional) + internal PostgreSQL traces |
| Evaluation | Custom scripts + Ragas (optional) |

---

## Định nghĩa "done" cho từng capability

| Capability | Done khi |
|---|---|
| Intent routing | Bất kỳ paraphrase nào của intent → classify đúng |
| Entity binding | Follow-up implicit → entity từ memory; explicit mới → entity từ DB |
| Multi-task | "giá bao nhiêu và có đau không" → 2 tasks riêng biệt |
| Multimodal output | Response có text + items + assets khi evidence có đủ |
| Asset masking | Token trong chunk → URL trong response |
| Memory | Entity từ turn N → available ở turn N+5 cùng session |
| Provider switch | LLM_PROVIDER=openai → pipeline dùng OpenAI, không cần restart |
| New intent | Thêm intent mới không cần sửa chat.py hay tool_executor.py |
| Hallucination guard | Giá/giờ/địa chỉ bịa → validator reject, trả fallback |
| Grounded fallback | LLM fail → trả best evidence text, không crash |

---

## Các điều KHÔNG được làm

1. **Không viết semantic fix cho từng câu cụ thể**
   Bad: `if "còn hàng" in query: return PRODUCT_DETAIL`
   Good: ConversationContext + general follow-up rules theo domain

2. **Không hard-code entity names hay product names trong code**
   Bad: `if entity_name == "AquaJet": ...`
   Good: Đọc từ DB, xử lý bất kỳ entity nào

3. **Không dùng LLM để quyết định business facts**
   Bad: LLM tự generate giá sản phẩm
   Good: SQL evidence → LLM chỉ diễn đạt lại

4. **Không để Ollama timeout làm null toàn bộ response**
   Bad: Timeout → crash → null answer
   Good: Timeout → grounded fallback từ evidence

5. **Không bỏ qua asset resolution**
   Bad: Response chỉ có text, bỏ qua `[asset:...]` tokens
   Good: Mọi token được resolve trước khi response

6. **Không tách biệt OpenAI và Ollama thành 2 code path khác nhau**
   Bad: `if provider == "openai": ... else: ...` trong chat.py
   Good: LLM client interface duy nhất, provider là implementation detail
