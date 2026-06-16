# AGENTS.md — Dental Multimodal RAG Web UI / RAG Control Center

## 1. Project Goal

Build a web UI for testing and operating the existing Dental Multimodal RAG system.

This web app is not only a chatbot UI. It must work as a **RAG Control Center** for:

1. Testing the chatbot as a real user.
2. Uploading and managing documents.
3. Monitoring ingestion pipeline.
4. Debugging retrieval pipeline.
5. Tracking evaluation metrics.
6. Exploring traces and latency per pipeline step.
7. Managing assets/images created by Asset Masking.
8. Viewing structured data such as products, services, FAQs, clinic info, tables, and table rows.

No login/authentication is required for this version.

The chatbot itself may fail during testing, but the evaluation and observability UI must remain stable and usable. If the chatbot API returns errors, the UI must still display trace_id, error type, failed step, and available debug information.

---

## 2. Preferred Tech Stack

Use one of the following frontend stacks:

Recommended:

- Vite + React + TypeScript
- TailwindCSS
- shadcn/ui or equivalent clean component system
- Recharts or ECharts for charts
- TanStack Query for API fetching/caching
- React Router for navigation
- Zod for frontend schema validation if needed

Do not build the UI as a single huge file. Keep components modular and maintainable.

---

## 3. Backend Assumption

Assume an existing FastAPI backend for the RAG system.

The frontend should call backend APIs through a centralized API client.

Create a `.env.example` for frontend:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_PUBLIC_ASSETS_BASE_URL=http://localhost:8000/assets
```

The UI must handle missing or not-yet-implemented backend endpoints gracefully by showing an empty state or friendly error message, not crashing the full application.

---

## 4. Main Layout

Create a dashboard layout:

```text
┌────────────────────────────────────────────────────────────┐
│ Dental RAG Control Center                    System Health │
├───────────────┬────────────────────────────────────────────┤
│ Sidebar       │ Main Content                               │
│               │                                            │
│ Chatbot       │                                            │
│ Upload Docs   │                                            │
│ Documents     │                                            │
│ Ingestion     │                                            │
│ Retrieval     │                                            │
│ Evaluation    │                                            │
│ Observability │                                            │
│ Traces        │                                            │
│ Assets        │                                            │
│ Data Tables   │                                            │
│ Settings      │                                            │
└───────────────┴────────────────────────────────────────────┘
```

Requirements:

- Sidebar fixed on the left.
- Header fixed or sticky at the top.
- Main content scrollable.
- Responsive enough for laptop/desktop screens.
- No login page.
- Use clean spacing and readable cards/tables.

Header should show high-level system status:

- PostgreSQL status
- pgvector status
- Ollama status
- Embedding model status
- Reranker status
- Assets directory status

If status endpoint is unavailable, show `Unknown` instead of crashing.

---

## 5. Sidebar Tabs

Create these navigation tabs:

1. Chatbot
2. Upload Documents
3. Document Store
4. Ingestion Monitor
5. Retrieval Playground
6. Evaluation Dashboard
7. Observability
8. Trace Explorer
9. Asset Manager
10. Data Tables
11. Settings

Each page should have a clear title, a short description, and the relevant content.

---

## 6. Chatbot Page — Critical Layout Requirements

This page is the most important test screen.

### 6.1 Chatbot Layout

Use a 2-column layout:

```text
┌───────────────────────────────────────┬───────────────────────────┐
│ Chat Area                             │ Debug Panel               │
├───────────────────────────────────────┼───────────────────────────┤
│ Conversation                          │ trace_id                  │
│                                       │ intent                    │
│ User messages: right side             │ confidence                │
│ Assistant messages: centered area      │ latency                   │
│                                       │ retrieval info            │
│ Input box at bottom                   │ assets                    │
└───────────────────────────────────────┴───────────────────────────┘
```

### 6.2 User Message Alignment

When the user types a message and presses Enter or clicks Send:

- The user message must appear on the **right side**.
- The user bubble should be visually distinct.
- The message should be appended immediately before waiting for backend response.
- The chat should auto-scroll to the latest message.

### 6.3 Chatbot Message Alignment

The chatbot/assistant message must always display in the **center of the chat area**, not left-aligned like a normal chat app.

Interpretation:

- The assistant response container should be centered horizontally inside the chat area.
- It can have a max-width such as `720px` or `760px`.
- Text inside the assistant answer can be left-aligned for readability, but the overall assistant card must be centered.
- Assistant messages should support text, images, tables, sources, and error states.

Example:

```text
                                  ┌──────────────┐
                                  │ User message │
                                  └──────────────┘

              ┌────────────────────────────────────────────┐
              │ Assistant answer appears centered here      │
              │ It can contain text, images, tables, source │
              └────────────────────────────────────────────┘
```

### 6.4 Input Box

The input box is already considered visually good and should remain polished.

Requirements:

- Fixed at the bottom of chat area.
- Rounded, clean, spacious.
- Supports Enter to send.
- Supports Shift+Enter for newline.
- Has Send button.
- Shows loading state when waiting for chatbot response.
- Does not move or resize awkwardly when messages are added.
- Prevent empty submission.

### 6.5 Chatbot Response Rendering

The chatbot response should support:

- Text answer
- Images from assets
- Tables if returned
- Source documents
- Trace ID
- Error messages
- Feedback buttons

For assets:

- If response contains assets with URLs, render them as image cards.
- If text contains `[asset:...]`, keep the token visible only if the asset cannot be resolved.
- If asset URL is broken, show a broken asset warning instead of crashing.

For sources:

Show source cards:

```text
Source: bang_san_pham.pdf — page 2
chunk_id: chunk_001
row_id: row_003
```

### 6.6 Debug Panel

The right debug panel must show the latest request metadata:

- trace_id
- detected intent
- confidence
- answer_type
- total_latency_ms
- retrieval_used
- JSON valid true/false
- failed_step if any
- chunks used
- rows used
- assets returned
- model used if available

If there is no current request, show an empty state.

### 6.7 Chatbot Error Handling

If `/api/chat` fails, still render a bot error card centered in the chat area.

The card should show:

- Friendly message: `Chatbot request failed.`
- trace_id if available
- error type
- failed step
- details if available

The page must not crash.

---

## 7. Upload Documents Page

Purpose: upload files into the RAG ingestion pipeline.

Features:

- Drag-and-drop upload area.
- File picker.
- Document type selector:
  - Auto Detect
  - Product Document
  - Service Document
  - FAQ Document
  - Clinic Info
  - Policy
  - Unknown
- Options:
  - Extract tables
  - Extract assets/images
  - Create embeddings
  - Require human review before active
- Upload button.
- Display upload result with doc_id and ingestion_run_id.

Expected API usage:

```text
POST /api/documents/upload
POST /api/ingestion/{doc_id}/run
```

If backend is unavailable, show a clear error.

---

## 8. Document Store Page

Purpose: view and manage ingested documents.

Display a table with columns:

- doc_id
- file_name
- file_type
- status
- version
- chunks count
- tables count
- assets count
- created_at
- updated_at
- actions

Actions:

- View detail
- Approve
- Activate
- Archive
- Re-ingest
- Delete if endpoint exists

Document detail should show:

- metadata
- chunks preview
- extracted tables
- extracted assets
- ingestion quality report
- status/version

Expected APIs:

```text
GET /api/documents
GET /api/documents/{doc_id}
POST /api/documents/{doc_id}/approve
POST /api/documents/{doc_id}/activate
POST /api/documents/{doc_id}/archive
```

---

## 9. Ingestion Monitor Page

Purpose: monitor ingestion health and quality.

Show summary cards:

- Total documents uploaded
- Parse success rate
- Parse failed count
- Total chunks created
- Empty chunk rate
- Tables detected
- Table rows created
- Assets extracted
- Broken asset count
- Embedding success rate
- Average ingestion latency

Show ingestion runs table:

- run_id
- doc_id
- file_name
- status
- started_at
- ended_at
- total latency
- error if any

Clicking a run opens timeline:

```text
upload_file          120ms
parse_document       2300ms
normalize_blocks     310ms
asset_masking        240ms
table_processing     480ms
chunking             130ms
embedding            4200ms
quality_check        80ms
save_to_postgres     250ms
```

Expected APIs:

```text
GET /api/ingestion/runs
GET /api/ingestion/runs/{run_id}
```

---

## 10. Retrieval Playground Page

Purpose: debug retrieval before generation.

UI:

- Query input.
- Checkboxes:
  - Structured SQL
  - Dense Retrieval
  - Sparse Retrieval
  - RRF
  - Reranker
  - HyDE
- Run Retrieval button.

Result tabs:

1. Router
2. Structured
3. Dense
4. Sparse
5. RRF
6. Reranker
7. Final Context

Show retrieved items with:

- id
- type: chunk / row / product / service / faq
- score
- source
- content preview
- metadata

Expected API:

```text
POST /api/retrieval/debug
```

Request example:

```json
{
  "query": "Tôi muốn xem sản phẩm Oral-B",
  "use_dense": true,
  "use_sparse": true,
  "use_structured": true,
  "use_rrf": true,
  "use_reranker": true,
  "use_hyde": false
}
```

---

## 11. Evaluation Dashboard Page

This page must be stable even when chatbot runtime fails.

It should read stored evaluation results and traces from the backend, not depend on live chat calls by default.

Show summary cards:

- End-to-end pass rate
- Router accuracy
- Retrieval Recall@5
- Faithfulness rate
- Answer correctness
- Safety pass rate
- JSON validity rate
- Asset resolve rate
- p95 latency
- Fallback rate
- No-result rate
- Unsupported claim rate

Show ground-truth coverage separately for retrieval, answer correctness, faithfulness, assets,
and safety. Render missing/non-applicable metrics as `N/A`, not `0%`.

Show charts:

- Metrics over time
- Latency by pipeline stage
- Pass, router accuracy, retrieval quality and faithfulness by intent
- Diagnostic alerts for failed/slow stages, empty retrieval, high fallback and high case
  failure rate

Show evaluation runs table:

- eval_run_id
- dataset_name
- pipeline_version
- data_version
- status
- started_at
- ended_at
- key metrics

Show evaluation cases table:

- query
- expected intent
- actual intent
- expected source
- retrieved source
- pass/fail
- trace_id
- per-case scores
- violations
- details

Allow selecting:

- mode: router, end-to-end, or all metrics
- profile: deterministic baseline or production runtime
- data version label

Expected APIs:

```text
GET /api/evaluation/datasets
POST /api/evaluation/run
GET /api/evaluation/runs
GET /api/evaluation/runs/{eval_run_id}
GET /api/evaluation/summary
GET /api/evaluation/results
GET /api/observability/diagnostics
```

If evaluation API is unavailable, show empty dashboard cards with `No evaluation data yet`.

---

## 12. Observability Page

Purpose: show system health and request metrics.

Show system status:

- PostgreSQL connected/disconnected/unknown
- pgvector enabled/disabled/unknown
- Ollama connected/disconnected/unknown
- Embedding model ready/unknown
- Reranker ready/disabled/unknown
- Assets folder available/missing/unknown

Show request metrics:

- Total requests today
- Success rate
- Error rate
- Average latency
- p50/p95/p99 latency
- No-result rate
- Fallback rate
- Clarification rate

Show recent errors table:

- time
- trace_id
- step
- error_type
- message

Expected APIs:

```text
GET /api/observability/health
GET /api/observability/metrics
GET /api/observability/errors
```

---

## 13. Trace Explorer Page

Purpose: inspect a single request or ingestion trace.

Features:

- Search by trace_id.
- List recent traces.
- Show trace summary:
  - query
  - intent
  - confidence
  - total latency
  - status
  - final answer preview
  - error if any
- Show timeline of steps:
  - router_intent
  - query_rewrite_hyde
  - dense_retrieval
  - sparse_retrieval
  - structured_retrieval
  - rrf_fusion
  - reranker
  - context_builder
  - asset_resolver
  - prompt_builder
  - llm_generation
  - json_validation
  - response_rendering

Clicking a step should show:

- input summary
- output summary
- latency_ms
- status
- error message
- retrieved IDs and scores if available

Expected APIs:

```text
GET /api/traces
GET /api/traces/{trace_id}
```

---

## 14. Asset Manager Page

Purpose: manage and debug Asset Masking.

Show assets table:

- asset_id
- asset_token
- asset_type
- doc_id
- chunk_id
- local_path
- public_url
- status
- preview

Asset detail should show:

- asset token
- image preview
- source document
- used in chunks/products/services
- broken URL warning if image cannot load

Expected APIs:

```text
GET /api/assets
GET /api/assets/{asset_id}
```

---

## 15. Data Tables Page

Purpose: inspect normalized structured data.

Create sub-tabs:

1. Products
2. Services
3. FAQs
4. Clinic Info
5. Tables
6. Table Rows
7. Chunks

Products columns:

- product_id
- name
- category
- description
- price
- quantity
- asset
- source_doc
- status
- version

Services columns:

- service_id
- name
- description
- duration_minutes
- price
- source_doc
- status
- version

Table Rows columns:

- row_id
- table_id
- entity_type
- entity_name
- row_text preview
- row_json preview
- embedding_status

Expected APIs:

```text
GET /api/products
GET /api/services
GET /api/faqs
GET /api/clinic-info
GET /api/tables
GET /api/table-rows
GET /api/chunks
```

---

## 16. Settings Page

Purpose: configure frontend-visible settings for testing.

Settings should include:

- API base URL display
- Public assets base URL display
- Router model name display
- Generation model name display
- Embedding model name display
- Reranker enabled/disabled display
- top_k_dense
- top_k_sparse
- top_k_final
- rrf_k
- reranker_enabled
- hyde_enabled
- confidence_threshold

If backend supports update settings, add save button. Otherwise read-only is acceptable.

Expected API:

```text
GET /api/settings
PUT /api/settings
```

Handle missing endpoint gracefully.

---

## 17. API Client Requirements

Create a centralized API client:

```text
src/api/client.ts
```

Responsibilities:

- Read `VITE_API_BASE_URL`.
- Add timeout handling.
- Parse JSON safely.
- Return typed errors.
- Never crash React components due to raw API errors.

Create separated API modules:

```text
src/api/chat.ts
src/api/documents.ts
src/api/ingestion.ts
src/api/retrieval.ts
src/api/evaluation.ts
src/api/observability.ts
src/api/traces.ts
src/api/assets.ts
src/api/dataTables.ts
src/api/settings.ts
```

---

## 18. Frontend Types

Create TypeScript types for:

```text
ChatMessage
ChatResponse
ChatDebugInfo
DocumentRecord
IngestionRun
TraceRecord
TraceStep
EvaluationSummary
EvaluationRun
EvaluationCase
AssetRecord
ProductRecord
ServiceRecord
FAQRecord
ClinicInfoRecord
RetrievalDebugResponse
SystemHealth
```

Place types under:

```text
src/types/
```

---

## 19. Error Boundaries and Stability

Very important: one broken page must not crash the whole web app.

Implement:

- App-level error boundary.
- Page-level error boundary if practical.
- Empty states.
- Loading states.
- API error states.
- Broken image fallback.
- Missing data fallback.

Evaluation Dashboard, Observability, and Trace Explorer must be resilient.

If chatbot fails, the user should still be able to open:

- Evaluation Dashboard
- Observability
- Trace Explorer
- Ingestion Monitor

---

## 20. Suggested Frontend Folder Structure

Use this structure or a clean equivalent:

```text
frontend/
  src/
    app/
      App.tsx
      routes.tsx
    components/
      layout/
        Sidebar.tsx
        Header.tsx
        PageContainer.tsx
      common/
        MetricCard.tsx
        StatusBadge.tsx
        DataTable.tsx
        EmptyState.tsx
        ErrorState.tsx
        LoadingState.tsx
      chat/
        ChatPage.tsx
        ChatWindow.tsx
        ChatInput.tsx
        MessageBubble.tsx
        AssistantMessageCard.tsx
        DebugPanel.tsx
        SourceCards.tsx
        AssetGallery.tsx
      upload/
        UploadDocumentsPage.tsx
        FileUploader.tsx
      documents/
        DocumentStorePage.tsx
        DocumentDetailDrawer.tsx
      ingestion/
        IngestionMonitorPage.tsx
        IngestionTimeline.tsx
      retrieval/
        RetrievalPlaygroundPage.tsx
        RetrievalResultTabs.tsx
      evaluation/
        EvaluationDashboardPage.tsx
        EvaluationCasesTable.tsx
        EvaluationRunsTable.tsx
      observability/
        ObservabilityPage.tsx
        SystemHealthPanel.tsx
        RecentErrorsTable.tsx
      traces/
        TraceExplorerPage.tsx
        TraceTimeline.tsx
        TraceStepDetail.tsx
      assets/
        AssetManagerPage.tsx
        AssetPreview.tsx
      data/
        DataTablesPage.tsx
        ProductsTable.tsx
        ServicesTable.tsx
        FAQsTable.tsx
        ClinicInfoTable.tsx
        TableRowsTable.tsx
      settings/
        SettingsPage.tsx
    api/
      client.ts
      chat.ts
      documents.ts
      ingestion.ts
      retrieval.ts
      evaluation.ts
      observability.ts
      traces.ts
      assets.ts
      dataTables.ts
      settings.ts
    types/
      chat.ts
      document.ts
      ingestion.ts
      trace.ts
      evaluation.ts
      observability.ts
      asset.ts
      data.ts
    hooks/
      useChat.ts
      useEvaluation.ts
      useTrace.ts
    main.tsx
    index.css
```

---

## 21. Chat API Expected Response Handling

The frontend should support this response shape:

```json
{
  "trace_id": "trace_001",
  "message": {
    "text": "Đây là thông tin sản phẩm Oral-B...",
    "assets": [
      {
        "asset_id": "asset_001",
        "url": "/assets/doc_001/img_001.png",
        "type": "product_image"
      }
    ],
    "sources": [
      {
        "doc_id": "doc_001",
        "file_name": "bang_san_pham.pdf",
        "page_number": 2,
        "chunk_id": "chunk_001",
        "row_id": "row_003"
      }
    ]
  },
  "debug": {
    "intent": "PRODUCT_DETAIL",
    "confidence": 0.92,
    "answer_type": "direct_data",
    "latency_ms": 1250,
    "retrieval_used": true,
    "json_valid": true,
    "chunks_used": ["chunk_001"],
    "rows_used": ["row_003"],
    "assets_returned": ["asset_001"]
  }
}
```

Also support error response:

```json
{
  "trace_id": "trace_002",
  "error": {
    "type": "LLM_TIMEOUT",
    "message": "LLM generation timeout",
    "failed_step": "llm_generation"
  },
  "debug": {
    "latency_ms": 30000
  }
}
```

---

## 22. UI Quality Requirements

- Clean, modern dashboard UI.
- Clear typography.
- Tables must be readable.
- Cards must have consistent spacing.
- Avoid cramped layout.
- Use badges for statuses:
  - success
  - failed
  - running
  - pending
  - active
  - archived
  - unknown
- Use skeleton/loading indicators.
- Use toast notifications for actions like upload, approve, activate, archive.
- Avoid browser alert boxes.

---

## 23. Testing Requirements

Add basic tests if the project setup supports it.

At minimum, manually verify:

1. App loads without backend.
2. Sidebar navigation works.
3. Chatbot page renders.
4. User message appears on the right after pressing Enter.
5. Assistant message appears centered.
6. Input box stays fixed and usable.
7. Chat API error does not crash the app.
8. Evaluation Dashboard loads empty state if no data.
9. Trace Explorer handles missing trace.
10. Broken asset image shows fallback.

If adding automated tests, test:

- Message alignment behavior.
- API error handling.
- Sidebar navigation.
- Rendering of evaluation cards.

---

## 24. README Requirements

Create or update `README.md` for the frontend.

Include:

- Project overview
- Tech stack
- How to install dependencies
- How to configure `.env`
- How to run dev server
- How to build production
- API endpoints expected from backend
- Notes about chatbot layout
- Notes about evaluation/observability stability

Example commands:

```bash
npm install
npm run dev
npm run build
```

---

## 25. Final Checklist for Codex

Before finishing, verify:

- [ ] Web app runs.
- [ ] No login screen.
- [ ] Sidebar tabs exist.
- [ ] Chatbot user message aligns right.
- [ ] Chatbot assistant response is centered.
- [ ] Input box is polished and stable.
- [ ] Upload Documents page exists.
- [ ] Document Store page exists.
- [ ] Ingestion Monitor page exists.
- [ ] Retrieval Playground page exists.
- [ ] Evaluation Dashboard page exists and is stable.
- [ ] Observability page exists.
- [ ] Trace Explorer page exists.
- [ ] Asset Manager page exists.
- [ ] Data Tables page exists.
- [ ] Settings page exists.
- [ ] API client handles errors safely.
- [ ] Empty states exist.
- [ ] README.md exists.
- [ ] `.env.example` exists.

After implementation, report clearly:

1. What was implemented.
2. Which pages are complete.
3. Which backend endpoints are expected.
4. Which endpoints are currently mocked or handled as empty state.
5. How to run the frontend.
6. What environment variables the user must provide, especially `VITE_API_BASE_URL` and `VITE_PUBLIC_ASSETS_BASE_URL`.
