# Dental RAG Control Center Frontend

Web UI vận hành và kiểm thử hệ thống SimplyDent Multimodal RAG.

## Tech Stack

- Vite + React + TypeScript
- TailwindCSS
- React Router
- TanStack Query
- Recharts
- Lucide icons
- Vitest + Testing Library

Không có màn hình đăng nhập. Mỗi page có query/error boundary riêng; Chatbot lỗi không làm
Evaluation, Observability, Trace Explorer hoặc Ingestion Monitor crash.

Chat requests không có timeout phía browser vì demo có thể chạy các model Ollama local chậm.
Frontend giữ trạng thái loading cho đến khi backend trả response hoặc kết nối thực sự thất bại.

## Install

```bash
cd frontend
cp .env.example .env
npm install
```

Hoặc:

```bash
pnpm install
```

## Environment

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_PUBLIC_ASSETS_BASE_URL=http://localhost:8000/assets
```

Backend cần cho phép frontend origin:

```env
CORS_ORIGINS=http://localhost:5173
```

## Run

Terminal backend:

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

Terminal frontend:

```bash
cd frontend
npm run dev
```

Mở `http://127.0.0.1:5173`.

## Build, Lint, Test

```bash
npm run lint
npm run test
npm run build
```

## Implemented Pages

1. Chatbot: user bên phải, assistant card căn giữa, debug panel, assets, sources, tables.
2. Upload Documents: drag/drop, document type và ingestion options.
3. Document Store: list, detail drawer, activate, archive, re-ingest.
4. Ingestion Monitor: summary cards, runs và pipeline timeline.
5. Retrieval Playground: structured/dense/sparse/RRF/reranker/HyDE controls.
6. Evaluation Dashboard: chọn bộ basic/multi-turn/semantic, chạy deterministic/production,
   xem grounded metrics, entity-binding/follow-up/multi-task scores, scenario pass rate,
   diagnostics và case-level violations/trace IDs.
7. Observability: health, request metrics, automatic diagnostic alerts và recent errors.
8. Trace Explorer: search/deep-link theo `trace_id`, recent traces, step timeline, token usage,
   GLiNER spans, context-binding decisions, conversation state và JSON details.
9. Asset Manager: preview, broken-image fallback và usage details.
10. Data Tables: products, services, FAQs, clinic info, tables, rows, chunks.
11. Settings: frontend URLs và read-only backend RAG settings.

## Backend Endpoints

Core:

- `POST /chat`
- `POST /ingest/upload`
- `POST /evaluation/run`

Control Center:

- `GET /api/documents`
- `GET /api/documents/{doc_id}`
- `POST /api/documents/{doc_id}/approve|activate|archive|reingest`
- `GET /api/ingestion/runs`
- `GET /api/ingestion/runs/{run_id}`
- `GET /api/ingestion/summary`
- `POST /api/retrieval/debug`
- `GET /api/evaluation/datasets|runs|cases|results|summary`
- `GET /api/observability/health|metrics|diagnostics|errors`
- `GET /api/traces`
- `GET /api/traces/{trace_id}`
- `GET /api/assets`
- `GET /api/assets/{asset_id}`
- `GET /api/products|services|faqs|clinic-info|tables|table-rows|chunks`
- `GET /api/settings`

## Backend Gaps Handled Gracefully

- Document delete chưa được hỗ trợ nên UI không hiển thị destructive delete action.
- Runtime settings update chưa được hỗ trợ; Settings page là read-only.
- Upload options ngoài file hiện được gửi để tương thích nhưng ingestion backend vẫn dùng
  pipeline đầy đủ theo cấu hình server.
- Metric không có ground truth được hiển thị `N/A`; coverage được hiển thị riêng để tránh hiểu
  nhầm `0%` là retrieval/generation lỗi.
- Evaluation run lưu actual result từng case, expected/retrieved IDs, score, violation và
  `trace_id`; bảng case trên dashboard đọc trực tiếp dữ liệu này.

Endpoint thiếu hoặc trả 404 sẽ hiển thị:

> Endpoint chưa sẵn sàng hoặc backend chưa trả dữ liệu.
