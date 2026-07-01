# AGENTS.md - Hướng dẫn làm việc với SimplyDent_RAG

Trước khi bắt đầu bất kỳ task nào trong repo này, đọc hai file sau:

- `PRODUCT_VISION.md`

Nếu nội dung trong tài liệu cũ và code hiện tại mâu thuẫn nhau, ưu tiên code hiện tại. Nếu code hiện tại mâu thuẫn với nguyên tắc sản phẩm trong `PRODUCT_VISION.md`, cần nêu rõ và sửa theo hướng tổng quát, không sửa theo từng câu hỏi hoặc từng tên entity cụ thể.

## Tổng Quan Project

SimplyDent_RAG là hệ thống chatbot RAG đa phương thức cho doanh nghiệp nha khoa, kèm web Control Center để vận hành, ingest tài liệu, debug retrieval, xem trace và chạy evaluation.

Chức năng chính hiện tại:

- Ingest tài liệu và dữ liệu nghiệp vụ từ PDF/DOCX/PPTX/HTML/MD, TXT, CSV, XLSX và ảnh. Docling được dùng khi cài optional ingestion extra; CSV/XLSX dùng pandas/openpyxl; ảnh dùng Pillow.
- Lưu dữ liệu vào PostgreSQL làm data layer thống nhất: documents, chunks, tables, table rows, assets, products, services, FAQs, clinic info, conversation memory, traces và evaluation.
- Dùng `pgvector` cho dense retrieval và PostgreSQL full-text search/trigram cho sparse retrieval tiếng Việt.
- Asset masking cho ảnh: stage file trong `assets/.staging/<doc_id>/`, tạo token `[asset:<stable_hash>]`, promote sang `assets/<doc_id>/`, resolve token/asset UUID khi render response.
- Chat evidence-first: contextual query rewriting (chuẩn hoá follow-up thành câu hỏi độc lập trước khi vào pipeline), task planning, entity resolution trên câu đã rewrite, registry-owned tool policy, evidence merge/gate, synthesis/fallback, validation, asset resolution và response rendering.
- Follow-up/conversation memory được xử lý chủ yếu bằng một bước LLM rewrite câu hỏi dựa trên transcript thô (không dùng con trỏ "entity đang active" duy nhất làm nguồn sự thật cho ngữ cảnh). Xem chi tiết ở mục "Chat Evidence-First Mặc Định" và `app/orchestration/query_rewriter.py`.
- Hỗ trợ conversation memory bằng `conversation_sessions`, `conversation_turns`, `conversation_summaries`; không dùng trace làm memory.
- Có evaluation framework cho router, retrieval, generation, assets, E2E và conversation/multi-turn.
- Frontend là Vite React Control Center, không phải landing page.

Công nghệ chính:

- Backend: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.x, Alembic, psycopg, httpx.
- Database: PostgreSQL, `pgvector`, `pg_trgm`, `unaccent`, generated `tsvector`, HNSW vector indexes.
- Ingestion/RAG: Docling, pandas, openpyxl, python-docx fallback, Pillow, `langchain-text-splitters`, sentence-transformers, GLiNER optional, rapidfuzz.
- LLM: provider-neutral client trong `app/generation/llm_client.py`; `LLM_PROVIDER=ollama` cho development/demo, `LLM_PROVIDER=openai` cho production.
- Frontend: Vite, React 19, TypeScript, React Router, TanStack Query, TailwindCSS v4, Recharts, Lucide icons, Vitest + Testing Library.

## Cấu Trúc Thư Mục

Thư mục/file quan trọng:

- `app/main.py`: tạo FastAPI app, CORS, static asset serving, startup validation, cleanup asset staging, warmup embedding/reranker/GLiNER và include routers. Không đặt business logic ở đây.
- `app/config.py`: Pydantic Settings và defaults cho database, embedding, LLM provider, retrieval, assets, observability, suggestions.
- `app/constants.py`: enum `Intent`, `RetrievalMode`, danh sách trace steps.
- `app/db/`: SQLAlchemy models, session factory, init DB, Alembic migrations.
- `app/ingestion/`: ingestion pipeline, parser, normalizer, asset masker, table classifier/processor, embedder, dedup, quality/smoke checks, approval/review.
- `app/assets/`: asset storage và resolver. Resolver phải xử lý cả token trong text/data và explicit `asset_ids`.
- `app/retrieval/`: router legacy/single-intent, entity resolver, typed structured query parsing, structured/dense/sparse retrievers, RRF, reranker, context builder.
- `app/orchestration/`: evidence-first task contracts, `query_rewriter.py` (LLM viết lại follow-up thành câu hỏi độc lập — chạy đầu tiên, ngay sau memory_load), planner, context binder (đã đơn giản hoá, không còn giữ pointer entity "đang active" xuyên turn), resolver/canonicalizer, consistency gate, intent registry, tool executor, evidence merger, contextual suggestions.
- `app/generation/`: provider-neutral LLM client, prompts, generator/fallback, response schemas, semantic validator, renderer.
- `app/services/chat.py`: chat orchestrator trung tâm. Đây là nơi điều phối runtime evidence-first và legacy debug path.
- `app/memory/`: conversation memory riêng biệt với trace.
- `app/ner/`: GLiNER/catalog span extraction cho query gốc và ingestion.
- `app/evaluation/`: dataset loader, metric, runner, component evaluators, conversation evaluation, source/ground-truth helpers.
- `app/observability/`: internal PostgreSQL tracing, metrics, optional Langfuse.
- `app/api/`: FastAPI routes. Routes phải mỏng, chỉ validate request/response và gọi service/pipeline/runner.
- `frontend/`: Vite React Control Center. API clients ở `frontend/src/api/`, types ở `frontend/src/types/`, pages/components theo domain ở `frontend/src/components/`.
- `scripts/`: script seed/verify/audit/smoke/preflight. Script nên idempotent và nói rõ cần PostgreSQL, Ollama hay seeded data.
- `tests/`: pytest backend và Vitest frontend. Backend có nhiều contract tests cho ingestion, orchestration, retrieval, generation, assets, observability và memory.
- `eval_datasets/`: datasets JSONL/JSON và fixtures CSV cho evaluation.
- `dental_chatbot_sample_data/`: sample products/services/faqs/images để demo/seed.
- `assets/`, `uploads/`: runtime file storage. Không commit generated asset/upload content trừ khi có yêu cầu rõ.

Không cần liệt kê tất cả file trong `AGENTS.md`. Khi làm task cụ thể, đọc file owner và test liên quan trước khi sửa.

## Luồng Hoạt Động Chính

### Startup Backend

`app/main.py` chạy các bước chính:

1. `validate_intent_registry()` để đảm bảo mỗi `Intent` có capability hợp lệ.
2. Tạo thư mục assets/uploads.
3. Cleanup staging assets và file asset không còn được database track.
4. Tạo/cached `ChatService`.
5. Nếu config bật, validate embedding model/dimension với database vector columns.
6. Nếu bật, warmup reranker và GLiNER.
7. Mount static `/assets` và include routers chat, ingestion, admin, evaluation, Control Center.

### Ingestion

Flow thực tế nằm trong `app/ingestion/pipeline.py`:

1. Upload hoặc ingest file có sẵn trên server.
2. Tính checksum và áp dụng duplicate policy: `reject`, `reuse`, `replace`, `force`.
3. Tạo `documents` status `draft` và `ingestion_runs` status `running`.
4. Parse bằng Docling nếu khả dụng, fallback TXT/CSV/XLSX/DOCX/image theo code hiện tại.
5. Normalize text/table/asset blocks, classify document/table, validate business rows.
6. Stage companion/document assets, tạo stable asset identity/token, mask asset position trong text.
7. Chạy entity span extraction cho ingestion metadata.
8. Lưu full table vào `tables`, row vào `table_rows`, sync row hợp lệ vào `products`, `services`, `faqs`, `clinic_info`.
9. Chunk chỉ text blocks, không chunk table như text.
10. Embed chunks/table rows/FAQ questions khi `create_embeddings=true`.
11. Link asset với chunks qua `chunk_assets`; link companion images vào product/service qua `asset_id` khi có `image_reference`.
12. Chạy smoke checks và quality report.
13. Apply status cascade: `active` nếu auto-approve và không có blocker, ngược lại `review_required`.
14. Promote asset staging ngay trước commit cuối; nếu lỗi thì rollback DB và cleanup staged/final files, document/run thành `failed`.

Approval/manual classification nằm trong `app/ingestion/review.py` và `app/api/routes_ingestion.py`. Manual approval phải rerun smoke checks và recompute business blockers từ persisted rows, không chỉ tin `quality_report` cũ.

### Chat Evidence-First Mặc Định

Khi `ENABLE_MULTI_TASK_PLANNER=true`, `ChatService` dùng runtime `evidence_first`:

1. Tạo `rag_traces`.
2. `memory_load`: load **raw transcript** (N turn gần nhất, dạng text thô user/assistant) theo `session_id` từ `conversation_turns`, cùng một danh sách rút gọn "entity gần đây" (tên chuẩn + type) dùng làm gợi ý, không phải nguồn sự thật duy nhất.
3. `contextual_query_rewrite` (bước mới, thay thế phần lớn vai trò của context binder cũ trong việc resolve tham chiếu ngầm): gọi LLM (`app/orchestration/query_rewriter.py`) với raw transcript + entity gần đây + câu hỏi hiện tại, trả về `RewrittenQuery { rewritten_query, is_standalone, needs_clarification, referenced_entities }`. Model chỉ viết lại câu hỏi, không được bịa fact nghiệp vụ, không trả lời câu hỏi. Nếu lỗi/timeout, fallback dùng nguyên câu hỏi gốc — bước này không bao giờ được chặn luồng chat. Nếu `needs_clarification=true`, đi thẳng tới nhánh clarification hiện có thay vì cố resolve entity.
4. `task_planning`: PlannerLLM/task planner tạo `TaskPlan` gồm `PlannedTask` bất biến, dựa trên `rewritten_query` (không phải câu hỏi thô) — chỉ là proposal.
5. `entity_span_extraction`: GLiNER optional + catalog fallback tìm span trên `rewritten_query` (đã đầy đủ tên entity, không còn đại từ/tham chiếu ngầm) thay vì câu gốc — độ chính xác match cao hơn hẳn vì không cần đoán ngữ cảnh.
6. `context_binding`, `entity_resolution`, `task_canonicalization`: `TaskBindingPipeline` tạo `BoundTaskPlan` từ span đã trích trên câu rewrite, database resolution và canonical filters. Không còn logic pronoun-resolution/entity "đang active" xuyên turn ở bước này — việc đó đã được xử lý ở bước 3.
7. `bound_task_consistency`: gate trước tool. Vẫn giữ các gate về evidence contract/cardinality của registry; bỏ các violation code chuyên biệt cho stale/switch entity (`no_stale_entity`...) vì lớp lỗi đó không còn tồn tại về mặt cấu trúc sau khi có bước rewrite.
8. `tool_execution`: `ToolExecutor` chỉ chạy tools được khai báo trong `IntentCapabilityRegistry`.
9. `evidence_merging` và `evidence_consistency`: merge, dedupe, ưu tiên evidence authoritative/curated/retrieved và chặn evidence/task mâu thuẫn.
10. `context_builder`: tạo context/prompt payload đã qua firewall. Planner query/global entities/rejected proposals không được đi vào synthesis payload.
11. Generation:
    - `GREETING`/`CHITCHAT`: no-RAG social generation qua shared schema.
    - intent có evidence và `ENABLE_EVIDENCE_SYNTHESIS=true`: LLM synthesis chỉ tạo answer text, used sources và safety flags.
    - nếu synthesis tắt, thiếu evidence, LLM/validation lỗi: fallback grounded từ evidence.
12. `json_validation`: validate schema và semantic grounding, gồm entity/fact grounding.
13. Contextual suggestions nếu `ENABLE_CONTEXTUAL_SUGGESTIONS=true`: interest state, candidate generation, consistency gate.
14. `memory_save`: lưu raw turn (query gốc, không phải rewritten_query, cùng answer text) vào `conversation_turns` để làm nguồn transcript cho bước rewrite ở lượt sau; đồng thời lưu BoundTask resolved, registry-eligible, qua evidence gate làm "entity gần đây" gợi ý. Không lưu planner-only entities/list rows/failed task. `rewritten_query` chỉ lưu trong trace, không dùng làm nguồn transcript, để tránh lỗi rewrite tích luỹ qua nhiều lượt.
15. `asset_resolver`: resolve token và explicit asset UUIDs.
16. `response_rendering`: trả `ChatResponse` — **bắt buộc bao gồm field entity/nguồn thực sự được dùng để sinh câu trả lời** (lấy từ evidence đã merge vào synthesis, không lấy từ span extraction thô), cho mọi intent kể cả FAQ. Đây là field trước đây bị thiếu ở một số nhánh và gây khó debug/hiển thị UI. Finish trace, optional Langfuse export.

Khi `ENABLE_MULTI_TASK_PLANNER=false`, code dùng runtime `legacy_single_intent_debug`: router -> entity resolution -> structured retrieval -> retrieval plan -> optional HyDE -> dense/sparse -> RRF -> optional reranker -> context -> generation/direct response -> validation -> asset resolution. Đây là debug path, không chạy contextual query rewrite, không nên thêm capability mới chỉ ở legacy path.

### Retrieval Và Tool Policy

- `IntentCapabilityRegistry` là nguồn chính cho entity scope, inheritance, allowed reference modes, allowed tools, evidence contract, memory persistence và suggestion policy.
- `ToolExecutor` dùng tool registry (`product_tool`, `service_tool`, `clinic_info_tool`, `faq_tool`, `document_rag_tool`) và có `register_tool()` cho extension. Không mở rộng tool bằng if-else rải rác.
- Structured SQL là authoritative cho product/service/clinic facts: giá, số lượng, thời lượng, giờ làm việc, policy, exact record IDs.
- `structured_query.py` parse typed filters/sorts; `structured_retriever.py` execute SQL. Không cho LLM tạo SQL.
- Dense/sparse retrievers phải lọc cả record status active và owning document active.
- HyDE là optional retrieval aid cho FAQ/unstructured RAG, không phải evidence, không ghi memory.
- FAQ/unstructured retrieval cho follow-up phải chạy trên `rewritten_query` (đã có tên entity đầy đủ), không chạy trên câu hỏi rút gọn gốc — tránh lấy nhầm FAQ không liên quan đến entity đang nói tới.

### Asset Resolution/Rendering

`ResponseRenderer.resolve_assets()` gọi `AssetResolver.resolve()` với:

- text answer;
- serialized `result.items[].data`;
- explicit `result.items[].asset_ids`;
- doc scope từ result items/sources.

Resolver bỏ qua template token như `[asset:product_XX]`, dedupe asset theo `asset_id`, chỉ trả asset/document active, và báo `missing_assets` thay vì crash.

### Frontend Control Center

Frontend route chính:

- `/chatbot`
- `/upload`
- `/documents`
- `/ingestion`
- `/retrieval`
- `/evaluation`
- `/observability`
- `/traces`
- `/assets`
- `/data`
- `/settings`

Frontend gọi backend qua API clients trong `frontend/src/api/` và shared types trong `frontend/src/types/`. Không reimplement routing, retrieval, binding, validation, business rules hoặc data filtering ở frontend. Nếu backend trả 404/missing data, UI phải hiện empty/error state, không crash toàn app.

### Evaluation

`POST /evaluation/run` dùng `app/evaluation/runner.py`.

- `profile=deterministic`: tắt LLM router, HyDE, reranker và clamp Ollama timeout để baseline lặp lại được.
- `profile=production`: dùng runtime settings hiện tại.
- Runner gọi `ChatService` thật, lưu aggregate metrics vào `evaluation_runs` và từng case vào `evaluation_case_results`.
- Conversation evaluation đọc trace steps để chấm entity binding, follow-up memory, multi-task decomposition và scenario pass rate.

### Migration: Thay Pointer-Based Entity Binding Bằng Contextual Query Rewriting

Bối cảnh: eval `conversation_scenario_results` cho thấy `follow_up_accuracy` và
`entity_switch_accuracy` thấp hơn hẳn các metric khác, với các lỗi lặp lại
dạng "entity sai bị dính qua nhiều turn", "entity mới bị từ khóa trùng lấn át",
"tên rút gọn không match", "FAQ follow-up lấy nhầm entity". Nguyên nhân gốc:
kiến trúc cũ coi ngữ cảnh hội thoại là một *con trỏ entity đang active* được
cập nhật/ghi đè qua từng turn bằng nhiều luật symbolic rời rạc (pronoun
resolution, switch detection, stale detection...) — mỗi luật chỉ vá được
một biến thể, không tự sửa khi có biến thể mới.

Hướng thay thế: thêm bước `contextual_query_rewrite` (mục "Chat Evidence-First
Mặc Định" ở trên, module `app/orchestration/query_rewriter.py`) viết lại câu
hỏi follow-up thành câu hỏi độc lập dựa trên toàn bộ raw transcript, TRƯỚC
khi vào entity span extraction. Sau khi có bước này, phần lớn logic
"pointer-based" ở các bước sau trở nên thừa vì input của chúng luôn đã là
câu hỏi đầy đủ, không còn đại từ/tham chiếu ngầm.

Khi triển khai, xử lý theo owning layer, không đổi cấu trúc thư mục:

1. Thêm `contextual_query_rewrite` vào danh sách trace steps trong
   `app/constants.py`.
2. Thêm `app/orchestration/query_rewriter.py` (đã có bản khởi tạo sẵn: prompt,
   `RewrittenQuery` schema, hàm `rewrite_query()` với fallback an toàn về
   câu hỏi gốc khi lỗi/timeout).
3. Trong `app/services/chat.py`, chèn bước gọi `rewrite_query()` ngay sau
   `memory_load`, trước `task_planning`. Truyền `rewritten_query` (không
   phải câu hỏi gốc) vào task planner và vào `entity_span_extraction`. Giữ
   câu hỏi gốc trong trace/log để debug và để `memory_save` lưu đúng raw
   transcript.
4. Trong `app/orchestration/context_binder.py`: gỡ phần logic giữ/ghi đè một
   entity "đang active" xuyên turn (pronoun resolution, switch-back, stale
   detection). Entity resolution sau bước rewrite chỉ cần hoạt động như một
   câu hỏi mới, không cần biết "turn trước đang nói về gì" nữa vì thông tin
   đó đã nằm sẵn trong `rewritten_query`.
5. Trong `app/orchestration/consistency_gate.py`: bỏ các violation code
   chuyên biệt cho stale/switch entity (ví dụ `no_stale_entity`); giữ lại
   các gate về evidence contract/cardinality không liên quan tới vấn đề này.
6. `app/memory/`: `memory_load` phải trả về raw turns (text thô, không chỉ
   entity id) để `query_rewriter` dùng; `memory_save` tiếp tục lưu raw turn
   như hiện tại, không lưu `rewritten_query` làm nguồn transcript (tránh lỗi
   rewrite tích luỹ dồn qua nhiều lượt — mỗi lượt phải rewrite lại từ raw
   text thật).
7. `app/generation/renderer.py`: đảm bảo `ChatResponse` luôn expose entity/
   nguồn thực sự dùng để sinh câu trả lời (lấy từ evidence đã merge), cho mọi
   intent kể cả FAQ — đây là bug độc lập với migration nhưng nên sửa cùng đợt
   vì cùng ảnh hưởng tới follow-up correctness quan sát được qua eval.
8. Cập nhật tests cùng layer: `tests/` cho `query_rewriter.py` (bao gồm case
   fallback khi LLM lỗi/timeout), cập nhật lại các test hiện có của
   `context_binder.py`/`consistency_gate.py` đang assert theo logic pointer
   cũ, cập nhật conversation evaluation dataset nếu cần format mới.
9. Không xóa `app/memory/` hay bảng `conversation_sessions`/`conversation_turns`/
   `conversation_summaries` — chúng vẫn là nguồn transcript chính, chỉ đổi
   cách chúng được dùng.
10. Chạy lại `scripts/verify_retrieval_pipeline.py` và bộ eval
    `conversation_scenario` sau khi migration để xác nhận
    `follow_up_accuracy`/`entity_switch_accuracy` cải thiện trước khi coi
    migration hoàn tất.

## Quy Ước Code

Quy ước có thể suy ra từ project:

- Python file/function/module dùng `snake_case`; class/Pydantic model/dataclass dùng `PascalCase`.
- Backend dùng type hints, dataclass/Pydantic model cho contract, SQLAlchemy ORM cho database.
- Pydantic task contracts quan trọng (`PlannedTask`, `BoundTask`, `EvidenceItem`, response schemas) nên được giữ immutable/frozen khi đã khai báo như hiện tại.
- API modules trong `app/api/` chỉ nên validate và delegate. Business orchestration thuộc `app/services/chat.py`, `app/ingestion/pipeline.py`, `app/orchestration/`, `app/retrieval/`, `app/generation/`.
- Khi đổi database schema, cập nhật cả `app/db/models.py`, migration trong `app/db/migrations/versions/`, tests và docs/config nếu cần.
- Khi đổi API payload, cập nhật backend schema, frontend type trong `frontend/src/types/`, API client trong `frontend/src/api/`, và tests liên quan.
- Khi thêm intent mới:
  1. Thêm `Intent` nếu chưa có trong `app/constants.py`.
  2. Thêm capability trong `app/orchestration/intent_registry.py`.
  3. Nếu cần tool mới, đăng ký handler trong `ToolExecutor` qua registry và update registry validation.
  4. Thêm typed query parsing/execution nếu cần SQL filters mới.
  5. Thêm tests cho planner/binder/gate/tool/evidence/validator.
  6. Không thêm handler riêng cho từng intent trong route hoặc chat orchestrator nếu registry/tool contract có thể xử lý.
- Test naming theo pytest: `tests/test_*.py`. Frontend tests nằm gần component/client và dùng Vitest.

Những điều cần tránh:

- Không hard-code tên sản phẩm, dịch vụ, entity, clinic fact hoặc câu tiếng Việt cụ thể để fix bug.
- Không thêm per-sentence semantic heuristic vào evidence-first path. Legacy router có deterministic fallback phrase lists, nhưng không nên biến nó thành business authority mới.
- Không để LLM quyết định fact nghiệp vụ. Giá, tồn kho, thời lượng, giờ làm việc, policy và catalog identity phải đến từ PostgreSQL evidence.
- Không mutate `PlannedTask` thành executable state. Chỉ `TaskCanonicalizer` tạo `BoundTask`.
- Không đưa planner global entities, rejected planner entities, candidates hoặc raw trace diagnostic vào synthesis payload/memory.
- Không dùng `rag_traces` làm conversation memory.
- Không bỏ qua validator, evidence gates, asset resolver hoặc response renderer cho direct SQL response.
- Không split table như text chunk thông thường.
- Không sửa lớn/refactor cross-layer nếu task không yêu cầu.
- Không xóa logic hiện có nếu chưa đọc tests và hiểu tại sao nó tồn tại.
- Không thay đổi destructive reset/delete semantics nếu không có yêu cầu rõ và tests.

## Hướng Dẫn Chạy Project

### Prerequisites

- Python 3.11 hoặc 3.12.
- `uv`.
- PostgreSQL 15+ với extensions: `vector`, `pg_trgm`, `unaccent`.
- Ollama nếu dùng `LLM_PROVIDER=ollama`.
- Node.js/npm cho frontend.

### Cài Backend Dependencies

Full development setup:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev --extra ingestion --extra models
cp .env.example .env
```

Core-only:

```bash
uv sync --extra dev
```

Core-only không đảm bảo parse PDF/Docling hoặc model embedding/GLiNER. Nếu chỉ cần dev không embedding, có thể dùng config trong README: `create_embeddings=false` hoặc tắt validation/fallback một cách có chủ đích. Không dùng hash fallback cho production.

### Database

Tạo database/extensions ví dụ:

```sql
CREATE DATABASE dental_rag;
\c dental_rag
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
```

Chạy migration:

```bash
alembic upgrade head
alembic current
```

Migration head hiện tại trong repo là `0009_service_aliases`.

### Chạy Backend

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

`status=degraded` nghĩa là API đã lên nhưng database chưa connected hoặc check chưa pass.

### Chạy Ollama Local

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama serve
```

OpenAI production dùng `LLM_PROVIDER=openai` và `OPENAI_API_KEY`. Không hard-code provider/model trong code.

### Chạy Frontend

```bash
cd frontend
npm install
npm run dev
```

Mở `http://127.0.0.1:5173`.

`frontend/README_FRONTEND.md` có nhắc `cp .env.example .env`, nhưng tại thời điểm rà soát chưa thấy `frontend/.env.example` trong repo. Nếu cần, tạo `frontend/.env` local với:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_PUBLIC_ASSETS_BASE_URL=http://localhost:8000/assets
```

### Test, Lint, Build

Backend:

```bash
uv run python scripts/preflight_phase1.py
uv run ruff check .
uv run pytest
```

Integration test PostgreSQL thật:

```bash
TEST_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/dental_rag \
  uv run pytest -m integration tests/test_db_connection.py
```

Frontend:

```bash
cd frontend
npm run lint
npm run test
npm run build
```

Backend chưa thấy command build riêng trong project. `pyproject.toml` dùng hatchling package metadata, nhưng workflow vận hành hiện tại là lint/test/migration/run server.

### Script Vận Hành Hay Dùng

```bash
uv run python scripts/seed_evaluation_fixtures.py
uv run python scripts/verify_retrieval_pipeline.py
uv run python scripts/verify_retrieval_pipeline.py --disable-llm-router
uv run python scripts/audit_chatbot_completeness.py
```

Một số script smoke/audit cần PostgreSQL, Ollama, seeded data hoặc backend đang chạy. Đọc đầu file/script và README trước khi chạy.

## Hướng Dẫn Cho Codex Khi Chỉnh Sửa Project

- Luôn đọc `PRODUCT_VISION.md`, `PRODUCT_ISSUES.md`, file owner và tests liên quan trước khi sửa.
- Chạy `git status --short` trước khi edit. Worktree có thể đang dirty; không revert/xóa thay đổi của người dùng.
- Xác định owning layer trước: API, chat service, orchestration, retrieval, generation, ingestion, assets, memory, evaluation, observability, frontend hay scripts.
- Sửa root cause trong code, không sửa đoán theo một query failing.
- Giữ thay đổi nhỏ và đúng pattern hiện có. Không đổi cấu trúc thư mục nếu không cần.
- Nếu thêm behavior mới, cập nhật tests cùng layer. Nếu thay đổi cross-contract backend/frontend, cập nhật cả backend tests và frontend tests.
- Nếu thay đổi flow, config, API contract, schema, migration hoặc command vận hành, cập nhật README/AGENTS/frontend README khi cần.
- Khi sửa bug retrieval/chat, xem trace step nào sai trước: task planning, span extraction, binding, resolution, gate, tool execution, evidence, synthesis, validation, asset resolver hay rendering.
- Khi sửa ingestion, cần nghĩ đến transaction/rollback, status cascade, asset staging/promotion, smoke checks và business dedup.
- Khi sửa frontend, dùng API clients/types hiện có, handle missing endpoint/empty state, không đưa business rule vào component.
- Trước khi kết thúc, chạy checks liên quan. Nếu không chạy được do thiếu DB/Ollama/model/network/thời gian, nói rõ command chưa chạy và lý do.

## Các Lưu Ý Quan Trọng

Module trung tâm không nên sửa tùy tiện:

- `app/services/chat.py`
- `app/orchestration/query_rewriter.py`
- `app/orchestration/intent_registry.py`
- `app/orchestration/schemas.py`
- `app/orchestration/context_binder.py`
- `app/orchestration/task_canonicalizer.py`
- `app/orchestration/consistency_gate.py`
- `app/orchestration/tool_executor.py`
- `app/generation/validator.py`
- `app/generation/renderer.py`
- `app/assets/resolver.py`
- `app/ingestion/pipeline.py`
- `app/ingestion/review.py`
- `app/db/models.py` và migrations
- `frontend/src/types/index.ts` và `frontend/src/api/*`

Config/env quan trọng:

- Database: `DATABASE_URL` hoặc `POSTGRES_*`.
- Embedding: `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `VALIDATE_EMBEDDING_ON_STARTUP`, `STRICT_EMBEDDING`, `ALLOW_EMBEDDING_FALLBACK`.
- LLM: `LLM_PROVIDER`, `OLLAMA_*`, `OPENAI_*`.
- Evidence-first: `ENABLE_MULTI_TASK_PLANNER`, `ENABLE_PLAN_REVIEW`, `ENABLE_EVIDENCE_SYNTHESIS`.
- Contextual query rewriting: `ENABLE_QUERY_REWRITE`, `QUERY_REWRITE_MODEL` (có thể khác `LLM_PROVIDER` model chính — production nên dùng model rẻ/nhanh), `QUERY_REWRITE_HISTORY_TURNS` (mặc định load N turn gần nhất), `QUERY_REWRITE_TIMEOUT_S`. Khi tắt hoặc lỗi, luôn fallback dùng câu hỏi gốc, không chặn luồng chat.
- NER/binding: `ENABLE_GLINER_NER`, `GLINER_*`, `ENABLE_CONTEXT_BINDER`.
- Suggestions: `ENABLE_CONTEXTUAL_SUGGESTIONS`, `MAX_CONTEXTUAL_SUGGESTIONS`, `SUGGESTION_HISTORY_LIMIT`.
- Retrieval: `DENSE_*`, `SPARSE_*`, `RRF_*`, `ENABLE_HYDE`, `ENABLE_RERANKER`, `ENTITY_MATCH_THRESHOLD`, `ENTITY_AMBIGUITY_MARGIN`.
- Assets/uploads: `ASSET_STORAGE_DIR`, `ASSET_PUBLIC_BASE_URL`, `UPLOAD_DIR`.
- Frontend/CORS: `CORS_ORIGINS`, `VITE_API_BASE_URL`, `VITE_PUBLIC_ASSETS_BASE_URL`.
- Optional observability: `ENABLE_LANGFUSE`, `LANGFUSE_*`.

Điểm dễ gây lỗi:

- `EMBEDDING_DIM` phải khớp model và PostgreSQL `vector(n)` columns. Startup có thể fail fast.
- `app/config.py` và `.env.example` hiện không cùng default cho `ENTITY_MATCH_THRESHOLD`/`ENTITY_AMBIGUITY_MARGIN`; khi debug entity matching phải kiểm tra giá trị runtime trong `.env`/settings endpoint.
- Direct SQL response vẫn phải qua validator/renderer/asset resolver. Ảnh product/service có thể đến từ `asset_id` trong item, không nhất thiết từ token trong answer text.
- FAQ evidence có thể đến từ curated FAQ, sparse/dense FAQ và document/table RAG tùy allowed tools. Không để FAQ lấy sai entity do query chung chung.
- Data active/stale: retrievers phải lọc active business rows và active source documents.
- `delete_document` và `reset_data` là destructive; backend có confirmation checks. Không bypass chúng trong script/UI.
- Debug payload chỉ nên trả khi request `debug=true` và server `DEBUG=true`. Trace/admin endpoints cần auth/proxy policy nếu production.
- `frontend/.env.example` chưa thấy trong repo tại thời điểm rà soát; dùng `frontend/.env` local nếu cần biến Vite.
- Auth/login chưa thấy trong frontend/backend hiện tại. Nếu thêm auth, cần thiết kế contract riêng thay vì chèn logic vào từng page/route.
- `contextual_query_rewrite` thêm một lệnh gọi LLM mỗi turn có khả năng là follow-up. Với `LLM_PROVIDER=ollama` (local dev/demo), latency bước này cộng dồn vào tổng latency turn và có thể đáng kể; với `LLM_PROVIDER=openai` (production) thường không đáng kể. Khi đo latency demo trên Ollama, tách riêng thời gian bước rewrite khỏi thời gian generation chính trong trace để không đánh giá nhầm nguyên nhân chậm. Có thể set `QUERY_REWRITE_MODEL` nhỏ hơn model chính để giảm latency, và có thể tắt bước rewrite (`ENABLE_QUERY_REWRITE=false`) khi `entity_span_extraction` đã tìm thấy span rõ ràng trong câu gốc, để tránh gọi LLM không cần thiết.
- Đã bỏ cách tiếp cận "một con trỏ entity đang active" được cập nhật/ghi đè qua từng turn (nguồn gốc của nhóm bug stale entity/entity switch không tự sửa được). Nếu thấy code cũ còn giữ pattern này ở `context_binder.py`/`consistency_gate.py`, đó là nợ kỹ thuật cần dọn theo hướng tổng quát (xem mục "Chat Evidence-First Mặc Định"), không vá thêm luật riêng cho từng case entity switch/stale.