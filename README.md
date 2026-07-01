# SimplyDent Multimodal RAG

Web Control Center được đặt tại [`frontend/`](frontend/) và có hướng dẫn riêng trong
[`frontend/README_FRONTEND.md`](frontend/README_FRONTEND.md).

Backend FastAPI cho chatbot doanh nghiệp nha khoa, dùng PostgreSQL làm data layer thống nhất:

- dữ liệu tài liệu và dữ liệu nghiệp vụ;
- vector search với `pgvector`;
- full-text search PostgreSQL;
- asset masking cho ảnh;
- tracing ingestion/chat;
- evaluation datasets và evaluation runs.

Hệ thống ưu tiên direct SQL cho dữ liệu có cấu trúc, chỉ dùng RAG/LLM khi cần. Qdrant không
được sử dụng.

## Architecture

Luồng ingestion:

```text
Upload -> checksum/dedup -> Docling/fallback parser
       -> normalize thành text_blocks / table_blocks / image_blocks
       -> auto-detect document type + validate business schema
       -> stage + mask assets -> process full tables + row records
       -> taxonomy mapping + sync products/services/FAQ/clinic info
       -> chunk riêng phần text -> strict embeddings
       -> smoke checks -> promote assets + commit PostgreSQL
       -> active hoặc review_required
```

Luồng chat:

```text
RouterLLM-first intent classification -> PostgreSQL entity resolver
            -> retrieval planner
            -> structured SQL
            -> optional HyDE
            -> dense + sparse trên chunks/table_rows/faqs
            -> weighted canonical RRF -> optional reranker
       -> context -> direct response or Ollama JSON
       -> validation -> asset resolution -> API response
```

Luồng chat evidence-first mặc định:

```text
Conversation memory -> Task planner / query decomposition
  -> Entity span detection
  -> Context binding decisions (không mutate planner proposal)
  -> Database entity resolution
  -> Atomic TaskCanonicalizer -> immutable BoundTask
  -> pre-tool consistency gate
  -> IntentCapabilityRegistry -> allowed tools
  -> Tool executor: structured SQL + FAQ + document/table RAG
  -> Evidence merger: dedupe + trust priority + conflict/missing-info report
  -> evidence consistency gate
  -> Synthesis LLM từ payload chỉ chứa BoundTask/evidence đã kiểm tra
  -> semantic validation -> asset resolution
  -> persist conversation state từ BoundTask đã pass -> API response
```

Planner chọn rõ `TEMPLATE`, `NO_RAG_LLM`, `DIRECT_SQL`, `STRUCTURED_ONLY`,
`STRUCTURED_THEN_HYBRID`, `HYBRID` hoặc `CLARIFY`. Product/service/FAQ được hợp nhất theo
canonical key nên cùng một entity tìm thấy từ SQL, row, dense và sparse chỉ chiếm một vị trí
trong context.

`PRODUCT_LIST`, `SERVICE_LIST`, `CLINIC_INFO`, `PRODUCT_DETAIL` và `SERVICE_DETAIL` không dùng
RAG. CHITCHAT dùng LLM no-RAG để sinh câu trả lời xã giao theo JSON schema. Product list hỗ trợ
lọc theo taxonomy, khoảng giá và sắp xếp theo giá/tên/danh mục/số lượng. Product/service detail
dùng exact/fuzzy SQL; khi tên thiếu hoặc mơ hồ, chatbot yêu cầu làm rõ thay vì chạy
dense/reranker/LLM. FAQ exact/fuzzy được trả trực tiếp; semantic fallback chỉ tìm collection
`faqs`, không tìm chunks/table rows và không gọi generation LLM.

Khi `ENABLE_MULTI_TASK_PLANNER=true`, hệ thống dùng pipeline mới: user query được tách thành
tối đa `MAX_SUB_QUERIES` task, mỗi task gọi các tool nội bộ phù hợp và mọi kết quả được chuẩn
hóa thành evidence. SQL/FAQ/chunks/table rows không còn là các đường trả lời riêng biệt mà là
nguồn bằng chứng cho synthesis LLM. Bật thêm `ENABLE_EVIDENCE_SYNTHESIS=true` để LLM tổng hợp
câu trả lời cuối từ evidence pack; nếu tắt flag này, hệ thống vẫn dùng evidence context nhưng
fallback về direct response.

`PlannedTask` là proposal bất biến và không chứa `resolved_ids` hoặc `effective_query`.
Context Binding tạo quyết định riêng; database resolver và `TaskCanonicalizer` sau đó tạo
`BoundTask` bất biến với entity canonical, ID, typed filters và executable query. Tool policy
được lấy từ `IntentCapabilityRegistry`, không lấy từ Planner. Hai consistency gate chặn task
hoặc evidence mâu thuẫn trước khi dữ liệu được gửi tới generation LLM. Planner query,
planner-global entities và rejected entities chỉ tồn tại trong trace diagnostic, không đi vào
synthesis prompt hoặc conversation memory.

Mỗi chat request tạo một `rag_traces` và các `rag_trace_steps`. Bước không cần chạy vẫn được
ghi với trạng thái `skipped`.

Ingestion giữ `documents` và `ingestion_runs` làm audit record. Các content record của một lần
ingest được commit cùng nhau; ảnh được ghi vào vùng staging trước và chỉ promote sang
`assets/<doc_id>/` ngay trước commit cuối. Nếu parse, embedding hoặc database stage phát sinh
exception, content transaction bị rollback, file staging/final của lần chạy bị xóa và
document/run được đánh dấu `failed`. Smoke check không đạt vẫn được lưu để review, nhưng
document không được chuyển sang `active`.

## Requirements

- Python 3.11 hoặc 3.12 được khuyến nghị cho Docling và sentence-transformers.
- PostgreSQL 15+.
- extension `vector` (pgvector), `pg_trgm` và `unaccent`.
- Ollama nếu cần generation/HyDE/LLM router.

Docling, model embedding và GLiNER được tách thành optional extras. Cấu hình mặc định là strict
cho embedding: application kiểm tra model/dimension khi startup và ingestion thất bại nếu model
không khả dụng hoặc dimension không khớp. Nếu không cài GLiNER, chat vẫn dùng catalog/regex span
fallback và ghi trạng thái degraded trong trace.

## Install

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

Core-only vẫn parse TXT/CSV và image assets, đồng thời dùng deterministic entity-span fallback.
PDF cần Docling. Nếu chưa cài model embedding, chỉ chạy ingestion với
`create_embeddings=false` hoặc bật fallback một cách tường minh cho development:

```env
STRICT_EMBEDDING=false
ALLOW_EMBEDDING_FALLBACK=true
VALIDATE_EMBEDDING_ON_STARTUP=false
```

Không dùng hash fallback trong production.

## PostgreSQL + pgvector

Ví dụ dùng PostgreSQL local:

```sql
CREATE DATABASE dental_rag;
\c dental_rag
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
```

Cấu hình `.env`:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/dental_rag
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
VALIDATE_EMBEDDING_ON_STARTUP=true
STRICT_EMBEDDING=true
ALLOW_EMBEDDING_FALLBACK=false
```

`EMBEDDING_DIM` phải khớp dimension migration và model. Khi đổi dimension sau khi đã migrate,
cần migration mới cho ba cột vector. Startup kiểm tra đồng thời dimension của model và các cột
`chunks.embedding`, `table_rows.embedding`, `faqs.embedding`; mismatch làm application fail
fast thay vì bỏ qua insert.

Chạy migration:

```bash
alembic upgrade head
```

Các migration hiện tại bật `vector`, `pg_trgm`, `unaccent`, tạo GIN/HNSW/trigram indexes,
checksum lookup index, stable asset identity và bảng nối `chunk_assets`. Migration
`0005_vietnamese_retrieval` rebuild các generated `tsvector` bằng cấu hình `simple` trên text
đã bỏ dấu. Migration `0006_business_record_dedup` bảo đảm chỉ có một product, service, FAQ
hoặc clinic key đã chuẩn hóa được active tại một thời điểm. Migration
`0007_catalog_taxonomy` thêm taxonomy, alias, document auto-detection, image references và
FAQ lineage. Migration `0008_conversation_memory` thêm bảng conversation memory cho pipeline
evidence-first. Kiểm tra revision:

```bash
alembic current
```

## Ollama

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama serve
```

`.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_ROUTER_MODEL=qwen2.5:7b-instruct
OLLAMA_GENERATION_MODEL=qwen2.5:14b-instruct-q4_K_M
OLLAMA_ROUTER_TIMEOUT_SECONDS=30
OLLAMA_GENERATION_TIMEOUT_SECONDS=120
OLLAMA_KEEP_ALIVE=30m
OLLAMA_ROUTER_NUM_PREDICT=512
OLLAMA_GENERATION_NUM_PREDICT=2048
OLLAMA_ROUTER_NUM_CTX=8192
OLLAMA_GENERATION_NUM_CTX=16384
ENABLE_LLM_ROUTER=true
ENABLE_PLAN_REVIEW=true
ROUTER_TIMEOUT_SECONDS=30
ENABLE_HYDE=false
PRELOAD_RERANKER_ON_STARTUP=true
```

RouterLLM là đường chính khi `ENABLE_LLM_ROUTER=true`. Router và generation đều phải có timeout
rõ ràng để lỗi provider degrade thành fallback có trace thay vì treo request. Nếu file `.env`
cũ còn đặt timeout bằng `0`, `Settings` sẽ clamp về default an toàn theo provider. Deterministic
fallback chỉ còn là safe fallback tối thiểu khi RouterLLM lỗi hoặc circuit breaker đang mở: xử
lý greeting/chitchat/clinic info, list rõ ràng và FAQ nha khoa rõ ràng; không tự đóng vai full
rule-based router cho product/service detail.

Sau khi tải model xong:

```bash
ollama list
```

Đặt trong `.env`:

```env
OLLAMA_ROUTER_MODEL=qwen2.5:7b-instruct
ENABLE_LLM_ROUTER=true
ENABLE_PLAN_REVIEW=true
ROUTER_TIMEOUT_SECONDS=30
```

Direct SQL responses không phụ thuộc Ollama. Khi Ollama lỗi ở một RAG route, chatbot trả về
fallback chỉ từ retrieved context và ghi lỗi vào trace. Trace Ollama ghi riêng wall latency,
model load, prompt evaluation, token generation và từng JSON-repair attempt. Startup dùng
chung embedding instance với `ChatService`; nếu reranker được bật, có thể preload trước request
đầu tiên bằng `PRELOAD_RERANKER_ON_STARTUP=true`.

Evidence-first pipeline:

```env
ENABLE_MULTI_TASK_PLANNER=true
ENABLE_PLAN_REVIEW=true
ENABLE_EVIDENCE_SYNTHESIS=true
MAX_SUB_QUERIES=3
MAX_EVIDENCE_ITEMS=12
CONVERSATION_HISTORY_TURNS=8
```

Pipeline này cần Ollama cho task planning và synthesis. Conversation memory dùng
`conversation_sessions`, `conversation_turns` và `conversation_summaries`.

## Run

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

`status=degraded` nghĩa là API đã chạy nhưng chưa kết nối được PostgreSQL.

## Ingest Documents

Upload và ingest ngay:

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@./data/dental-products.pdf" \
  -F "document_type=auto" \
  -F "duplicate_policy=reject"
```

Upload CSV/XLSX kèm ảnh sản phẩm:

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@./data/products.csv" \
  -F "asset_files=@./data/images/oral_b_pro_500.png" \
  -F "asset_files=@./data/images/aquajet_mini.png" \
  -F "document_type=auto"
```

Cột `image_reference`/`Tên file ảnh` phải khớp basename của file upload. Pipeline tạo asset
token ổn định, lưu asset trong `assets/<doc_id>/` và gắn `products.asset_id` hoặc
`services.asset_id`. Ảnh companion không bắt buộc phải xuất hiện trong text chunk.

Business file contracts:

```text
Product: name,brand,model,category,description,price,currency,quantity,link,
         image_reference,aliases
Service: service_name,category,description,duration_minutes,price,currency,symptoms,
         indications,contraindications,image_reference
FAQ:     question,answer,category,keywords,aliases
```

Category được map vào taxonomy chuẩn, ví dụ `Bàn chải điện`, `Kem đánh răng`, `Máy tăm
nước`, `Nước súc miệng`, `Chỉ nha khoa`. Category không nhận diện được, category trùng tên
sản phẩm, giá sai định dạng hoặc `image_reference` không tồn tại sẽ đưa document về
`review_required`.

Với `document_type=auto`, pipeline suy luận `product_catalog`, `service_catalog`, `faq`,
`clinic_info`, `mixed_business_data`, `policy` hoặc `general_document`. Kết quả và confidence
được trả trong response upload và lưu ở `documents`.

`duplicate_policy` hỗ trợ:

- `reject` (mặc định): trả HTTP 409 nếu checksum đang tồn tại;
- `reuse`: trả document/run hiện có;
- `replace`: ingest version mới rồi archive các version cũ sau khi thành công;
- `force`: tạo thêm version mà không archive version cũ.

Đây là document-level dedup theo checksum. Business-level dedup chạy riêng:

- key được chuẩn hóa lowercase, bỏ dấu và chuẩn hóa punctuation/whitespace;
- row trùng product/service/FAQ/clinic key trong cùng ingestion chỉ sync business record một lần;
- document ở trạng thái review không thay thế dữ liệu active hiện tại;
- khi auto-approval hoặc manual approval thành công, bản active cũ cùng business key được
  archive/disable và bản mới được activate trong cùng transaction;
- `products.version` và `services.version` tăng khi thay thế bản active cũ;
- unique partial indexes trong PostgreSQL ngăn hai bản cùng normalized key đồng thời active.

Pipeline chuẩn hóa từng table, lưu `classification_confidence`, `classification_reasons` và
`column_mapping` trong metadata. Table không xác định hoặc confidence thấp buộc document về
`review_required`, kể cả khi `AUTO_APPROVE_INGESTION=true`.

Pipeline không đưa table vào `RecursiveCharacterTextSplitter`. Docling export phần text không
bao gồm table; mỗi table được lưu nguyên bản và xử lý từng dòng riêng.

Ingest một file đã nằm trên server:

```bash
curl -X POST http://localhost:8000/ingest/run \
  -H "Content-Type: application/json" \
  -d '{"source_path":"/absolute/path/to/services.xlsx"}'
```

Xem run:

```bash
curl http://localhost:8000/ingest/runs/INGESTION_RUN_UUID
```

Nếu `AUTO_APPROVE_INGESTION=false`, document và các record liên quan có status
`review_required`. Approval chạy lại smoke checks trước khi kích hoạt:

```bash
curl -X POST http://localhost:8000/documents/DOCUMENT_UUID/approve
```

Xóa vĩnh viễn một document:

```bash
curl -X DELETE \
  "http://localhost:8000/api/documents/DOCUMENT_UUID?confirm=true"
```

Reset content đã ingest nhưng giữ taxonomy:

```bash
curl -X POST http://localhost:8000/api/admin/data-reset \
  -H "Content-Type: application/json" \
  -d '{"scope":"content","confirmation":"DELETE SIMPLYDENT CONTENT"}'
```

Đổi `scope` thành `runtime` để xóa thêm traces và evaluation data. Web Control Center có cùng
thao tác tại Document Store. Trang chatbot có hai tab `Chat` và `FAQ`; tab FAQ đọc trực tiếp
các record active từ PostgreSQL.

Approval trả HTTP 409 nếu thiếu embedding, thiếu file asset, asset không liên kết với
chunk hoặc business record,
token không resolve được, table row lệch số lượng, table chưa phân loại, business record mất
source row hoặc còn document active trùng checksum. Khi pass, status được cascade tới
`documents`, `chunks`, `tables`, `table_rows`, `assets`, `products`, `services`,
`clinic_info` và FAQ liên quan.

Phân loại thủ công một table chưa xác định trước khi approve:

```bash
curl -X POST \
  http://localhost:8000/documents/DOCUMENT_UUID/tables/TABLE_UUID/classify \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "clinic_info",
    "column_mapping": {"0": "key", "1": "value"}
  }'
```

## Ingestion Storage Map

Một file có thể tạo record ở nhiều bảng. Bảng đích được quyết định theo loại block và schema
của từng table, không quyết định chỉ bằng extension hoặc tên file:

- `documents`: luôn có một source record cho file upload;
- `ingestion_runs`: audit, latency từng stage, số lượng và quality/smoke report;
- `chunks`: chỉ chứa text block đã normalize/chunk, không chứa table bị cắt dở;
- `tables`: bản table đầy đủ dưới dạng Markdown/JSON;
- `table_rows`: từng dòng table, có embedding và source metadata;
- `assets`: metadata ảnh/file đã extract;
- `chunk_assets`: quan hệ many-to-many giữa token asset và các chunk sử dụng token;
- `products`: dòng table được phân loại là product;
- `services`: dòng table được phân loại là service;
- `faqs`: dòng có cột question/answer;
- `clinic_info`: table key/value hoặc lịch làm việc.

Business table được chọn theo schema của từng table:

- product columns -> `products`;
- service columns -> `services`;
- question/answer columns -> `faqs`;
- key/value hoặc lịch làm việc -> `clinic_info`;
- mọi table luôn được giữ nguyên trong `tables` và từng dòng trong `table_rows`.

Archive:

```bash
curl -X POST http://localhost:8000/documents/DOCUMENT_UUID/archive
```

Ảnh được ghi trước vào `assets/.staging/<doc_id>/`, validate, rồi promote sang
`assets/<doc_id>/` ngay trước database commit. Nếu commit lỗi, thư mục final được dọn trong
rollback. Token mặc định có dạng `[asset:<stable_hash>]`, được tính từ checksum
document, source reference và checksum asset. Nếu tài liệu đã có token cụ thể, pipeline giữ
token đó và ưu tiên token trong table để nối `products.asset_id`/`services.asset_id`.

Token placeholder như `[asset:product_XX]` không được coi là asset thật. Token cụ thể không có
record tương ứng vẫn làm smoke check thất bại. API `/chat` resolve token hợp lệ sang
`/assets/<doc_id>/<file_name>`.

Quality report của run chứa:

- classification và warning theo từng table;
- embedding backend/success/failure;
- số chunk/table/row/asset/business record;
- latency theo stage;
- `business_dedup` gồm số row trùng bị skip và số phiên bản active cũ bị supersede;
- smoke checks gồm embedding, table consistency, token resolution, file asset, source lineage
  và duplicate active checksum.

## Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message":"Tôi muốn xem bàn chải điện Oral-B",
    "session_id":"demo-session",
    "history":[]
  }'
```

Debug chỉ được trả khi cả request `debug=true` và server `DEBUG=true`. Trace payload mặc định
không hiển thị qua admin API.

## Retrieval Verification

Chạy kiểm chứng thật qua `ChatService`, PostgreSQL và trace:

```bash
python scripts/verify_retrieval_pipeline.py
```

Script mặc định kiểm tra một product detail structured-only và một FAQ hybrid. Có thể tách
retrieval khỏi RouterLLM để lấy baseline ổn định:

```bash
python scripts/verify_retrieval_pipeline.py --disable-llm-router
```

Sparse retrieval chuẩn hóa tiếng Việt, loại cụm hỏi không có giá trị phân biệt, tìm
`chunks.content_tsv`, `table_rows.row_tsv`, `faqs.question_tsv` và fallback sang trigram.
Dense retrieval tìm collection theo intent. FAQ rows dùng `source_row_id` để hợp nhất với
record trong `faqs`; product/service rows dùng foreign key lineage tương ứng.

## Business APIs

```bash
curl http://localhost:8000/products
curl "http://localhost:8000/products/Oral-B%20Pro%20500"
curl http://localhost:8000/services
curl "http://localhost:8000/services/T%E1%BA%A9y%20tr%E1%BA%AFng%20r%C4%83ng"
```

Các endpoint này luôn filter `status = 'active'`.

## Evaluation

Dataset mặc định `eval_datasets/dental_basic_eval.jsonl` có 30 case cân bằng, ba case cho
mỗi intent. Ground truth dùng stable source key như `product:<name>`, `service:<name>`,
`faq:<question>` và `clinic_info:<key>`; evaluator resolve các key này sang UUID hiện tại
trước mỗi run nên không phụ thuộc UUID thay đổi sau re-ingestion.

Dataset `eval_datasets/dental_retrieval_eval.jsonl` bổ sung các paraphrase và entity rút gọn
để buộc chạy structured fallback, dense, sparse và RRF. Dùng dataset này khi tuning retrieval;
không suy luận chất lượng hybrid chỉ từ dataset direct-SQL mặc định.
Quy trình mở rộng lên 30–50 case nằm tại
[`eval_datasets/README.md`](eval_datasets/README.md).

Baseline local ngày 12/06/2026:

- dataset 30 case: E2E pass `1.0`, router accuracy `1.0`;
- dataset hybrid 6 case: Hit@1/Recall@5/MRR@10/nDCG@10 và E2E pass đều `1.0`;
- wrong asset/missing asset trên hybrid dataset đều `0.0`.

Hybrid baseline trên dùng generation timeout một giây để ép kiểm tra grounded fallback; không
dùng con số fallback/LLM latency của run đó làm benchmark production.

Seed dữ liệu fixture trước lần chạy đầu:

```bash
uv run python scripts/seed_evaluation_fixtures.py
```

Router-only, không cần Ollama nhưng cần PostgreSQL để lưu run:

```bash
curl -X POST http://localhost:8000/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"mode":"router","profile":"deterministic","data_version":"fixtures-v1"}'
```

Comprehensive deterministic tắt LLM router, HyDE và reranker để tạo baseline có thể lặp lại.
Profile `production` dùng nguyên cấu hình runtime trong `.env`.

```bash
curl -X POST http://localhost:8000/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"mode":"all","profile":"deterministic","data_version":"fixtures-v1"}'
```

```bash
curl -X POST http://localhost:8000/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{
    "mode":"all",
    "profile":"deterministic",
    "dataset_path":"eval_datasets/dental_retrieval_eval.jsonl",
    "dataset_name":"dental_retrieval_eval",
    "dataset_version":"1.0",
    "data_version":"fixtures-v1"
  }'
```

```bash
curl http://localhost:8000/evaluation/runs/EVALUATION_RUN_UUID
curl "http://localhost:8000/api/evaluation/results?eval_run_id=EVALUATION_RUN_UUID"
curl http://localhost:8000/api/evaluation/summary
curl http://localhost:8000/api/observability/diagnostics
```

Framework gồm:

- router accuracy, precision/recall/F1, confusion matrix;
- Hit@1, Hit@3, Recall@5/10, MRR@10, nDCG@10;
- reranker before/after top-1, improvement rate, latency;
- JSON/schema pass, answer correctness, faithfulness, price/citation grounding và safety;
- asset resolve/missing/broken/wrong rates;
- E2E success/pass/fallback/no-result/clarification và p50/p95/p99 latency;
- coverage cho retrieval, answer, faithfulness, safety; metric thiếu ground truth trả `N/A`,
  không được tính là pass;
- kết quả từng case gồm trace ID, expected/retrieved IDs, scores, violations và answer;
- diagnostics tự động cho empty retrieval, stage failure, latency cao, fallback cao và case
  failure rate.

Migration `0004_eval_results` thêm dataset hash/version snapshot, config snapshot và
`evaluation_case_results`. Evaluation run luôn lưu cả aggregate metrics và kết quả từng case.

## Traces

Summary:

```bash
curl http://localhost:8000/traces
```

Chi tiết không lộ input/output:

```bash
curl http://localhost:8000/traces/TRACE_UUID
```

Chỉ dùng trong môi trường quản trị:

```bash
curl "http://localhost:8000/traces/TRACE_UUID?include_payloads=true"
```

Production cần đặt auth/proxy policy trước các endpoint admin.

## Tests

```bash
uv run python scripts/preflight_phase1.py
uv run ruff check .
uv run pytest
```

Test PostgreSQL thật:

```bash
TEST_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/dental_rag \
  uv run pytest -m integration tests/test_db_connection.py
```

## Optional Langfuse

Install:

```bash
uv sync --extra observability
```

```env
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://...
```

Internal PostgreSQL tracing luôn hoạt động kể cả khi Langfuse không được cấu hình hoặc export
thất bại.

## Troubleshooting

**`extension "vector" is not available`**

Cài pgvector cho PostgreSQL instance đang dùng, restart PostgreSQL, rồi chạy lại migration.

**HNSW migration lỗi do PostgreSQL/pgvector cũ**

Nâng pgvector lên phiên bản hỗ trợ HNSW. Không bỏ vector indexes trong production.

**Docling import/parsing lỗi**

Dùng Python 3.11/3.12 và chạy `uv sync --extra ingestion`. TXT/CSV fallback không cần Docling.

**Model embedding tải chậm hoặc thiếu RAM**

Đổi sang `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, cập nhật
`EMBEDDING_DIM=384`, rồi tạo lại database/migration phù hợp dimension.

**Application không start vì embedding validation**

Kiểm tra model thực tế có cùng dimension với `EMBEDDING_DIM` và migration hay không. Với môi
trường development không cần embedding, có thể tắt `VALIDATE_EMBEDDING_ON_STARTUP` và gửi
`create_embeddings=false`; không nên tắt strict validation trong production.

**Ollama connection refused**

Kiểm tra `ollama serve`, `ollama list` và `OLLAMA_BASE_URL`. Direct SQL routes vẫn hoạt động.

**Dữ liệu ingest không xuất hiện trong chat**

Kiểm tra `document.status`, `quality_report.smoke_test.blocking_reasons` và status của record
liên quan. Khi auto-approve tắt, gọi endpoint `/documents/{doc_id}/approve`; endpoint sẽ từ
chối nếu dữ liệu chưa đủ điều kiện retrieval.

**Document luôn ở `review_required` vì duplicate**

Mặc định `duplicate_policy=reject`. Dùng `reuse` để lấy run cũ hoặc `replace` để ingest version
mới và archive version cũ sau khi version mới pass. `force` chủ động giữ nhiều version nên
smoke check sẽ chặn auto-approve nếu vẫn có checksum active trùng.

## Production Notes

- Đặt authentication/authorization cho ingestion, evaluation và trace APIs.
- Dùng object storage/CDN thay local assets khi triển khai nhiều replica.
- Không đổi embedding model/dimension mà không version dữ liệu và re-embed.
- Backup PostgreSQL gồm cả business records, vectors, trace và evaluation metadata.
- Medical answers chỉ là định hướng; prompt yêu cầu không chẩn đoán chắc chắn.
