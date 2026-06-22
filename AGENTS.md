# AGENTS.md — Build Multimodal RAG Chatbot for Dental Business

## 0. Project Goal

Build a complete, working, production-oriented **Multimodal RAG chatbot for a dental business**.

The system must support:

- Ingesting documents from multiple formats: PDF, DOCX, TXT, CSV, XLSX, images if needed.
- Parsing documents with **Docling**.
- Using **PostgreSQL as the unified database**:
  - document database
  - structured business database
  - vector database through `pgvector`
  - conversation memory
  - tracing/evaluation metadata storage
- Using **Asset Masking** for images:
  - stage extracted images before committing and store promoted files in `assets/<doc_id>/`
  - replace image positions in text/chunks/tables with stable tokens such as `[asset:<stable_hash>]`
  - preserve explicit source tokens when the document provides them
  - resolve these tokens back to real asset URLs when answering the user
- Handling tables properly:
  - store original table as JSON/Markdown
  - split table into row-level retrievable records
  - sync product/service tables into normalized business tables
- Supporting initial intents:
  - `GREETING`
  - `CHITCHAT`
  - `CLINIC_INFO`
  - `FAQ`
  - `PRODUCT_LIST`
  - `PRODUCT_DETAIL`
  - `PRODUCT_COMPARE`
  - `SERVICE_LIST`
  - `SERVICE_DETAIL`
  - `UNKNOWN`
- Using evidence-first orchestration for chat:
  - direct SQL returns authoritative business evidence;
  - FAQ rows and document chunks/table rows provide curated or retrieved evidence;
  - the generation LLM synthesizes the final answer from evidence when enabled.
- Using RAG only when needed. Structured SQL remains the authority for catalog, clinic facts,
  prices, quantities, service durations and exact product/service records.
- Using direct SQL as evidence for structured queries such as product list, service list,
  clinic info, and exact product/service detail.
- Using hybrid retrieval for RAG:
  - dense retrieval using pgvector
  - sparse retrieval using PostgreSQL full-text search
  - structured retrieval using SQL filters
  - RRF fusion
  - optional reranker
- Using local/open-source models by default:
  - LLM through Ollama
  - embeddings through open-source sentence-transformers models
- Providing evaluation and observability from the beginning:
  - ingestion tracing
  - retrieval tracing
  - generation tracing
  - latency per stage
  - component-level evaluation
  - system-level evaluation
  - debugging tables and logs

Important: Build a complete working project, not just a skeleton. Every major module must be implemented with runnable code, clear interfaces, tests where possible, and documented setup steps.

---

## 1. Confirmed Architecture Decisions

The following decisions are final and must be implemented:

1. **PostgreSQL is the unified data layer**.
   - Use PostgreSQL for documents, chunks, tables, assets, products, services, FAQs, traces, and evaluation datasets.
   - Use `pgvector` for vector similarity search.
   - Do not use Qdrant.

2. **Tables must not be stored only as raw tables**.
   - Store the full table in `tables`.
   - Store every row in `table_rows`.
   - If a table represents products, sync it into `products`.
   - If a table represents services, sync it into `services`.

3. **Asset Masking is required**.
   - Do not use a Vision LLM to caption images during ingestion by default.
   - Stage real images in `assets/.staging/<doc_id>/` and promote them to `assets/<doc_id>/` only after the ingestion data is ready to commit.
   - Generate deterministic asset identity from document checksum, source reference, and asset checksum.
   - Replace image positions with stable asset tokens like `[asset:<stable_hash>]`.
   - Preserve a concrete authored token when one exists in the source.
   - Link assets to every chunk that contains their token through `chunk_assets`.
   - At response time, resolve asset tokens to actual URLs/paths.

4. **Chat uses evidence-first orchestration, not a single rigid intent handler**.
   - `ENABLE_MULTI_TASK_PLANNER=true` is the default architecture. A user query is planned into
     one or more tasks, each task executes allowed tools, evidence is merged, and the final
     response is synthesized from evidence when `ENABLE_EVIDENCE_SYNTHESIS=true`.
   - Structured SQL results are authoritative evidence, not a separate final-answer shortcut.
     They must still pass through the shared response schema, validator, renderer and asset
     resolver.
   - `PRODUCT_LIST`, `SERVICE_LIST`, and `CLINIC_INFO` should use direct SQL first as evidence.
   - `PRODUCT_DETAIL` and `SERVICE_DETAIL` use exact/fuzzy SQL to resolve the entity. Ambiguous
     or missing entities require clarification. General RAG may provide supplementary context,
     but must not invent or select a structured business record.
   - `PRODUCT_COMPARE` retrieves structured rows for each entity and uses the LLM only for
     comparison wording grounded in those rows.
   - FAQ may use direct FAQ search, FAQ semantic search, or document RAG evidence. If synthesis
     fails, the safe fallback may return the best curated FAQ answer directly.
   - `GREETING` and `CHITCHAT` are no-RAG LLM paths. They use the generation LLM without SQL,
     retrieval context, or document RAG.

5. **Evaluation and tracing are part of the pipeline**.
   - Do not add observability later as an afterthought.
   - Every ingestion run and query request must produce trace records.
   - Each major stage must record latency, input summary, output summary, status, and error if present.

6. **Conversation memory is explicit application state**.
   - Keep memory in `conversation_sessions`, `conversation_turns`, and
     `conversation_summaries`.
   - Use stable `session_id` values for multi-turn conversations.
   - Do not infer current conversation state from `rag_traces`.
   - Persist canonical post-binding entities and resolved IDs, not raw PlannerLLM proposals.

---

## 2. Recommended Tech Stack

Use this stack unless the existing repository already has equivalent choices.

### Backend

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic v2
- SQLAlchemy 2.x or SQLModel
- Alembic for migrations
- psycopg / asyncpg

### Database

- PostgreSQL
- pgvector extension
- PostgreSQL full-text search using `tsvector` / `tsquery`

### Parsing and ingestion

- Docling for document parsing
- pandas/openpyxl for CSV/XLSX where needed
- Pillow for image saving/validation

### Chunking

- LangChain `RecursiveCharacterTextSplitter`

### Embeddings

Default local embedding model:

- `BAAI/bge-m3` if feasible

Acceptable fallback:

- `intfloat/multilingual-e5-base`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for lightweight demo

Embedding fallback is opt-in for development only. Production defaults must fail fast when the
model cannot load or its dimension does not match `EMBEDDING_DIM` and the PostgreSQL vector
columns.

### Reranker

Preferred:

- `BAAI/bge-reranker-v2-m3`

If too heavy, make reranker optional through config.

### LLM

Use a provider-neutral LLM layer. Ollama is the default demo provider for local pipeline
verification; OpenAI is the intended production provider when `LLM_PROVIDER=openai`.

Recommended models:

- Demo Router: `qwen2.5:7b-instruct` or configured `OLLAMA_ROUTER_MODEL`
- Demo Generation: `qwen2.5:14b-instruct-q4_K_M` or configured `OLLAMA_GENERATION_MODEL`
- Production Router: configured `OPENAI_ROUTER_MODEL`, for example `gpt-4.1-nano`
- Production Generation: configured `OPENAI_GENERATION_MODEL`, for example `gpt-4.1`
- Vision fallback only when explicitly needed: `llava:latest`

Do not use `llava:latest` as the default text router/generator unless no other model is available.

### Evaluation / Observability

Implement local-first observability:

- Internal PostgreSQL trace tables are required.
- Add optional Langfuse integration if environment variables are provided.
- Add optional OpenTelemetry hooks if feasible.
- Ragas integration can be added for offline evaluation, but the project must still include custom evaluation scripts for router, retrieval, asset mapping, and JSON validity.

---

## 3. Current Project Structure and Module Boundaries

The repository is no longer a skeleton. Treat the following as the current architecture and keep
new work aligned with these module boundaries:

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── constants.py
│   ├── taxonomy.py
│   ├── admin/
│   │   └── data_reset.py
│   ├── memory/
│   │   └── conversation_memory.py
│   ├── ner/
│   │   └── entity_span_extractor.py
│   ├── orchestration/
│   │   ├── schemas.py
│   │   ├── task_planner.py
│   │   ├── context_binder.py
│   │   ├── task_resolver.py
│   │   ├── task_canonicalizer.py
│   │   ├── binding_pipeline.py
│   │   ├── intent_registry.py
│   │   ├── consistency_gate.py
│   │   ├── tool_executor.py
│   │   └── evidence_merger.py
│   ├── services/
│   │   └── chat.py
│   ├── db/
│   │   ├── session.py
│   │   ├── models.py
│   │   ├── migrations/
│   │   └── init_db.py
│   ├── ingestion/
│   │   ├── pipeline.py
│   │   ├── docling_parser.py
│   │   ├── document_classifier.py
│   │   ├── normalizer.py
│   │   ├── asset_masker.py
│   │   ├── table_normalizer.py
│   │   ├── table_processor.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── business_dedup.py
│   │   ├── business_validation.py
│   │   ├── quality_checks.py
│   │   ├── smoke_checks.py
│   │   ├── review_policy.py
│   │   └── review.py
│   ├── retrieval/
│   │   ├── router.py
│   │   ├── entity_extractor.py
│   │   ├── entity_resolver.py
│   │   ├── planner.py
│   │   ├── structured_query.py
│   │   ├── normalization.py
│   │   ├── types.py
│   │   ├── query_rewrite.py
│   │   ├── dense_retriever.py
│   │   ├── sparse_retriever.py
│   │   ├── structured_retriever.py
│   │   ├── rrf.py
│   │   ├── reranker.py
│   │   └── context_builder.py
│   ├── generation/
│   │   ├── llm_client.py
│   │   ├── ollama_client.py
│   │   ├── prompts.py
│   │   ├── generator.py
│   │   ├── schemas.py
│   │   ├── validator.py
│   │   └── renderer.py
│   ├── assets/
│   │   ├── resolver.py
│   │   └── storage.py
│   ├── evaluation/
│   │   ├── datasets.py
│   │   ├── diagnostics.py
│   │   ├── eval_router.py
│   │   ├── eval_retrieval.py
│   │   ├── eval_generation.py
│   │   ├── eval_assets.py
│   │   ├── eval_conversation.py
│   │   ├── eval_e2e.py
│   │   ├── faithfulness.py
│   │   ├── ground_truth.py
│   │   ├── runner.py
│   │   ├── source_loader.py
│   │   └── metrics.py
│   ├── observability/
│   │   ├── tracing.py
│   │   ├── logging.py
│   │   ├── langfuse_client.py
│   │   └── metrics.py
│   └── api/
│       ├── routes_ingestion.py
│       ├── routes_chat.py
│       ├── routes_admin.py
│       ├── routes_evaluation.py
│       └── routes_control_center.py
├── assets/
├── uploads/
├── eval_datasets/
├── dental_chatbot_sample_data/
├── scripts/
│   ├── backup_before_reset.py
│   ├── seed_evaluation_fixtures.py
│   ├── verify_catalog_ingestion.py
│   └── verify_retrieval_pipeline.py
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── app/
│   │   ├── components/
│   │   ├── types/
│   │   └── utils/
│   ├── package.json
│   └── README_FRONTEND.md
├── tests/
├── alembic.ini
├── pyproject.toml
├── .env.example
├── README.md
└── AGENTS.md
```

### 3.1 Architectural Layers

The project is one application with several logical data and execution layers. PostgreSQL is the
single physical database, but its tables serve different responsibilities:

1. **Source/document layer**
   - `documents`, `chunks`, `tables`, `table_rows`, `assets`, `chunk_assets`;
   - preserves parsed source content, lineage, embeddings and asset locations.
2. **Authoritative business layer**
   - `products`, `services`, `faqs`, `clinic_info`;
   - taxonomy and alias tables normalize business concepts;
   - this layer is the authority for exact prices, quantities, durations, catalog identity and
     clinic facts.
3. **Conversation layer**
   - `conversation_sessions`, `conversation_turns`, `conversation_summaries`;
   - stores user-facing state needed for follow-up and multi-turn behavior;
   - must not be replaced by trace data.
4. **Observability/evaluation layer**
   - `ingestion_runs`, `rag_traces`, `rag_trace_steps`, `evaluation_*`;
   - explains what happened and measures quality, but is not source-of-truth business data.

`pgvector` is an extension inside PostgreSQL, not a separate vector service. Files under
`assets/` are the binary asset store; PostgreSQL keeps their identity, lineage and URLs.

The backend execution layers are:

```text
API boundary
  -> chat/ingestion service orchestration
  -> immutable task proposals and task-scoped context binding decisions
  -> entity resolution and atomic BoundTask canonicalization
  -> pre-tool consistency gate and registry-owned tool policy
  -> structured/FAQ/hybrid evidence retrieval
  -> evidence merge, evidence consistency gate and context construction
  -> grounded synthesis or deterministic fallback
  -> validation, asset resolution and rendering
  -> memory persistence and trace finalization
```

The frontend is an operational Control Center over these backend contracts. It is not allowed to
reimplement routing, retrieval, binding, validation or business rules.

### 3.2 Default Chat Runtime

The default evidence-first path is:

```text
ChatRequest
  -> memory_load
  -> task_planning
  -> entity_span_extraction
  -> context_binding
  -> entity_resolution
  -> task_canonicalization
  -> bound_task_consistency
  -> tool_execution
  -> evidence_merging
  -> evidence_consistency
  -> context_builder
  -> prompt_builder
  -> synthesis_generation or grounded fallback
  -> json_validation
  -> memory_save
  -> asset_resolver
  -> response_rendering
```

Responsibilities are deliberately separated:

- PlannerLLM decomposes the request into one or more proposed tasks. Its entity names, rewritten
  sub-queries and filters are proposals, not authority.
- GLiNER/catalog matching detects spans explicitly present in the original user message.
- Context Binder emits immutable decisions using explicit spans, same-turn resolved task
  references and conversation state. It never mutates Planner output.
- Database entity resolution converts names into authoritative product/service IDs.
- Task Canonicalizer atomically creates immutable `BoundTask` records with an effective query,
  canonical entity names, resolved IDs and typed filters.
- IntentCapabilityRegistry owns entity scope, inheritance, allowed tools and clarification policy.
- Mandatory consistency gates validate BoundTask state before tools and evidence before synthesis.
- Tool Executor accepts only gated BoundTasks, applies registry-owned tool policy and returns
  evidence only.
- Evidence Merger groups evidence by `task_id`, prioritizes authoritative rows and reports
  missing/conflicting evidence.
- Generation writes natural answer text. The server owns items, sources, entities and assets.
- Validation must reject responses that are structurally valid but conflict with selected
  evidence.

### 3.3 Task Contract and Multi-Intent Rules

One user message may create zero, one or many tasks. Context binding is always task-scoped; there
is no single global entity that may be applied to every task.

Example:

```text
"Tẩy trắng răng giá bao nhiêu, có đau không và cuối tuần mở cửa không?"

t1 SERVICE_DETAIL -> Tẩy trắng răng
t2 FAQ            -> Tẩy trắng răng, inherited from t1
t3 CLINIC_INFO    -> no product/service entity
```

Planner and executable task contracts are deliberately separate:

- `original_query`: exact user message;
- immutable `PlannedTask.planner_query`, `planner_entities` and planner filters, retained only as
  untrusted proposals and diagnostics;
- immutable `BoundTask.effective_query`: server-approved query after binding and resolution;
- `BoundTask.entity_names`: canonical entity names for this task;
- `BoundTask.resolved_ids`: authoritative IDs after database resolution;
- typed filters, sort and limit;
- binding source and reference mode.

`PlannedTask` is frozen immediately after Planner output validation. Entity resolution,
effective queries, server tool policy and authoritative IDs never live on it. `TaskCanonicalizer`
performs one atomic construction of `BoundTask`; downstream tools, evidence merge, synthesis and
memory do not accept `PlannedTask`.

`planner_global_entities` is diagnostic metadata only. It is excluded from BoundTask, retrieval,
tool selection, synthesis payloads and memory updates.

Binding authority order:

```text
explicit span in the original user message
  > authoritative same-turn resolved task reference
  > authoritative conversation-state ID/name
  > PlannerLLM proposal
```

Rules:

- explicit entity switching overrides old memory;
- implicit follow-up may reuse conversation state;
- mixed references may combine memory and explicit entities;
- compare tasks may contain multiple entities without requiring a different schema;
- unrelated tasks such as `CLINIC_INFO` must not inherit product/service entities;
- missing implicit context requires clarification;
- a pre-tool gate must ensure task entity names, IDs, typed filters, effective query, reference
  mode and inheritance source agree;
- a second gate must ensure authoritative evidence IDs and types agree with each BoundTask before
  synthesis.

### 3.4 Context Binding Is Not HyDE

Context binding resolves conversational references using verified state. It does not invent
content and is part of orchestration.

HyDE creates optional hypothetical text to improve dense retrieval. It is part of query rewrite,
may run only after task binding, and its output is never evidence or authoritative entity state.

```text
original query -> task binding -> effective query -> optional HyDE retrieval query
```

HyDE must not overwrite the original query, effective task contract, resolved IDs or conversation
memory.

Module ownership rules:

- `app/main.py` wires FastAPI, CORS, static asset serving, startup validation, reranker warmup,
  GLiNER warmup, and routers. Do not put business logic in `main.py`.
- `app/services/chat.py` is the chat orchestrator. With the default evidence-first settings, it
  owns the runtime order: memory load, task planning, explicit entity-span extraction, task-level
  context binding, database entity resolution, tool execution, evidence merging, context building,
  synthesis generation or grounded fallback, validation, memory save, asset resolution, response
  rendering, and trace finalization.
- The older single-intent path remains available only when `ENABLE_MULTI_TASK_PLANNER=false`.
  Keep it working for debugging, but do not add new behavior there unless the evidence-first
  path also receives the same capability.
- When `ENABLE_MULTI_TASK_PLANNER=true`, `app/services/chat.py` switches to the evidence-first
  orchestration path: conversation memory, task planning/query decomposition, entity-span
  extraction, context binding, entity resolution, tool execution, evidence merging, synthesis
  generation, validation, asset resolution, and memory save.
- `app/memory/*` owns conversation memory. Do not use `rag_traces` as memory; traces are
  observability records. Conversation state is stored server-side in
  `conversation_sessions.metadata.state` and includes active product/service IDs and names,
  active domain/topic, last intents, recent filters, and pending clarification. Summaries are
  persisted in `conversation_summaries`.
- `app/ner/entity_span_extractor.py` owns explicit span detection from the original user query.
  `urchade/gliner_multi-v2.1` is the default multilingual GLiNER model; catalog-name and
  constraint matching remains the deterministic fallback because model recall is not sufficient
  for every Vietnamese dental phrase. Extracted spans are signals, not authoritative database
  entities, and must be traced with provider, confidence, source, degraded state and
  load/inference errors.
- `app/orchestration/*` owns multi-task planning, task-level context binding, tool execution and
  evidence merging. Tools wrap existing retrievers; they must return evidence, not final answers.
  When RouterLLM is enabled, malformed plans receive one JSON repair attempt and one LLM
  plan-review pass. If the plan remains unusable, return a safe `UNKNOWN` clarification instead
  of silently switching to a parallel rule-based router. The planner may propose entities, but
  `context_binder.py` must verify whether they were explicitly mentioned in the original query,
  inherited from another gated/resolved task, or bound from conversation state before entity
  resolution. `task_resolver.py`, `task_canonicalizer.py`, `binding_pipeline.py`,
  `intent_registry.py` and `consistency_gate.py` own the remaining trusted-task transition.
- Context binding is task-scoped. It must support multiple entities, mixed explicit/pronominal
  references, multiple intents in one query, entity switching, list-filter refinements, and
  missing-context clarification. It must never copy one task's entity blindly to every task.
  Follow-up state IDs are authoritative only for implicit references; an explicit new entity must
  continue through database-backed entity resolution. Binder emits decisions without mutating
  `PlannedTask`; only the atomic canonicalizer may create executable `BoundTask` state.
- Tool access is a server-owned policy in `IntentCapabilityRegistry`. Planner output cannot
  broaden `PRODUCT_*`, `SERVICE_*`, or `CLINIC_INFO` tasks into unrelated FAQ/document RAG.
  Tool Executor accepts only BoundTasks that pass the pre-tool gate.
- `app/api/*` should stay thin. API modules should validate request/response boundaries and call
  services or pipeline classes; they should not duplicate retrieval or ingestion logic.
- `app/api/routes_control_center.py` exposes operational `/api/*` endpoints for the frontend:
  documents, ingestion runs, retrieval debug, evaluation, observability, traces, assets, data
  tables, and read-only settings.
- `app/ingestion/pipeline.py` owns ingestion orchestration. Classification, table normalization,
  dedup, business validation, smoke checks, review policy, and embedding behavior live in their
  dedicated ingestion modules.
- `app/retrieval/router.py` performs intent routing; `entity_resolver.py` performs database-backed
  entity matching; `planner.py` chooses the retrieval mode; retrievers should not make routing
  decisions.
- `app/retrieval/structured_query.py` parses and normalizes typed product/service constraints
  extracted by PlannerLLM or deterministic parsing. Product filters include taxonomy/product
  aliases, product names, brand terms, feature terms, price range, stock/quantity and whitelisted
  sort fields. Service filters include service names, category aliases such as
  implant/tẩy trắng/niềng răng, feature/symptom terms, price range, duration range and whitelisted
  sort fields. Keep query parsing separate from execution in `structured_retriever.py`.
- `app/generation/llm_client.py` owns the provider-neutral LLM interface and provider factory.
  `ollama_client.py` is a compatibility wrapper for the local demo provider. Chat orchestration,
  routing, planning and generation must call the shared LLM client rather than depending directly
  on Ollama.
- `app/generation/*` owns prompt construction, LLM calls, schema validation, fallback response
  construction, and response rendering. Evidence synthesis asks the LLM only for answer text,
  used source IDs and safety flags. The server deterministically builds entities, result items,
  sources and asset IDs from validated evidence. Every planned task must have at least one cited
  evidence source. Fallback responses should remain natural enough for demos, but they are still
  fallback paths and must not add facts not present in evidence.
- `app/assets/*` owns asset storage and resolution. Do not resolve product/service assets through
  chunk links only; direct SQL responses must resolve explicit `asset_id` values.
- `app/evaluation/*` owns datasets, offline runners, metric calculation, diagnostics, and result
  persistence. `eval_conversation.py` scores post-binding entities, follow-up memory use,
  multi-task decomposition and whole-scenario outcomes. Evaluation scripts should reuse
  production pipeline components where practical.
- `frontend/` is part of the project architecture. It is a Vite + React + TypeScript Control
  Center that consumes both core endpoints and `/api/*` operational endpoints.
- `scripts/` contains operational verification and fixture utilities. Keep scripts idempotent and
  explicit about whether they require PostgreSQL, Ollama, or seeded data.

---

## 4. Database Schema Requirements

Create migrations for all required tables.

### 4.1 `documents`

Stores uploaded source documents.

Fields:

- `doc_id UUID PRIMARY KEY`
- `file_name TEXT NOT NULL`
- `file_type TEXT`
- `source_path TEXT`
- `uploaded_at TIMESTAMP`
- `checksum TEXT`
- `status TEXT`
- `version INT DEFAULT 1`
- `detected_document_type TEXT`
- `document_type_confidence FLOAT`
- `metadata JSONB`

Statuses:

- `draft`
- `parsed`
- `review_required`
- `approved`
- `active`
- `archived`
- `failed`

### 4.2 `chunks`

Stores retrievable text chunks.

Fields:

- `chunk_id UUID PRIMARY KEY`
- `doc_id UUID REFERENCES documents(doc_id)`
- `chunk_index INT`
- `content TEXT NOT NULL`
- `content_tsv TSVECTOR`
- `embedding VECTOR(<embedding_dim>)`
- `content_type TEXT`
- `page_number INT`
- `section_title TEXT`
- `status TEXT DEFAULT 'active'`
- `metadata JSONB`

Add indexes:

- vector index on `embedding`
- GIN index on `content_tsv`
- index on `doc_id`
- index on `status`

### 4.3 `assets`

Stores extracted images and other assets.

Fields:

- `asset_id UUID PRIMARY KEY`
- `doc_id UUID REFERENCES documents(doc_id)`
- `chunk_id UUID NULL REFERENCES chunks(chunk_id)`
- `asset_token TEXT NOT NULL`
- `stable_asset_key TEXT NOT NULL`
- `asset_type TEXT`
- `local_path TEXT`
- `public_url TEXT`
- `page_number INT`
- `bbox JSONB`
- `status TEXT DEFAULT 'active'`
- `metadata JSONB`

Constraints and indexes:

- unique constraint on `(doc_id, stable_asset_key)`
- non-unique index on `asset_token`
- index on `doc_id`
- index on `status`

`chunk_id` is retained as the primary/first chunk compatibility link. It is not the complete
asset-to-chunk relationship.

### 4.3.1 `chunk_assets`

Stores the many-to-many relationship between assets and chunks.

Fields:

- `chunk_asset_id UUID PRIMARY KEY`
- `chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE`
- `asset_id UUID REFERENCES assets(asset_id) ON DELETE CASCADE`
- `occurrence_count INT DEFAULT 1`
- `metadata JSONB`

Add a unique constraint on `(chunk_id, asset_id)` and indexes on both foreign keys.

### 4.4 `tables`

Stores full original parsed tables.

Fields:

- `table_id UUID PRIMARY KEY`
- `doc_id UUID REFERENCES documents(doc_id)`
- `page_number INT`
- `table_name TEXT`
- `table_markdown TEXT`
- `table_json JSONB`
- `status TEXT DEFAULT 'active'`
- `metadata JSONB`

### 4.5 `table_rows`

Stores row-level retrievable table records.

Fields:

- `row_id UUID PRIMARY KEY`
- `table_id UUID REFERENCES tables(table_id)`
- `doc_id UUID REFERENCES documents(doc_id)`
- `row_index INT`
- `entity_type TEXT`
- `entity_name TEXT`
- `row_text TEXT`
- `row_json JSONB`
- `row_tsv TSVECTOR`
- `embedding VECTOR(<embedding_dim>)`
- `status TEXT DEFAULT 'active'`
- `metadata JSONB`

Add indexes:

- vector index on `embedding`
- GIN index on `row_tsv`
- index on `entity_type`
- index on `entity_name`
- index on `status`

### 4.6 `products`

Normalized business table for product queries.

Fields:

- `product_id UUID PRIMARY KEY`
- `name TEXT NOT NULL`
- `category TEXT`
- `category_code TEXT NULL REFERENCES product_categories(code)`
- `source_category TEXT`
- `brand TEXT`
- `model TEXT`
- `description TEXT`
- `price NUMERIC`
- `currency TEXT DEFAULT 'VND'`
- `quantity INT`
- `link TEXT`
- `image_reference TEXT`
- `asset_id UUID NULL REFERENCES assets(asset_id)`
- `source_doc_id UUID REFERENCES documents(doc_id)`
- `source_row_id UUID NULL REFERENCES table_rows(row_id)`
- `version INT DEFAULT 1`
- `status TEXT DEFAULT 'active'`
- `valid_from TIMESTAMP`
- `valid_to TIMESTAMP`
- `metadata JSONB`

### 4.7 `services`

Normalized business table for service queries.

Fields:

- `service_id UUID PRIMARY KEY`
- `name TEXT NOT NULL`
- `category_code TEXT NULL REFERENCES service_categories(code)`
- `source_category TEXT`
- `description TEXT`
- `duration_minutes INT`
- `price NUMERIC`
- `currency TEXT DEFAULT 'VND'`
- `symptoms TEXT[]`
- `indications TEXT[]`
- `contraindications TEXT[]`
- `image_reference TEXT`
- `asset_id UUID NULL REFERENCES assets(asset_id)`
- `source_doc_id UUID REFERENCES documents(doc_id)`
- `source_row_id UUID NULL REFERENCES table_rows(row_id)`
- `version INT DEFAULT 1`
- `status TEXT DEFAULT 'active'`
- `valid_from TIMESTAMP`
- `valid_to TIMESTAMP`
- `metadata JSONB`

### 4.8 `clinic_info`

Stores structured clinic facts.

Fields:

- `id UUID PRIMARY KEY`
- `key TEXT NOT NULL`
- `value TEXT NOT NULL`
- `status TEXT DEFAULT 'active'`
- `source_doc_id UUID NULL REFERENCES documents(doc_id)`
- `metadata JSONB`

### 4.9 `faqs`

Stores FAQ data.

Fields:

- `faq_id UUID PRIMARY KEY`
- `question TEXT NOT NULL`
- `answer TEXT NOT NULL`
- `category TEXT`
- `category_code TEXT NULL REFERENCES faq_categories(code)`
- `keywords TEXT[]`
- `is_active BOOLEAN DEFAULT TRUE`
- `question_tsv TSVECTOR`
- `embedding VECTOR(<embedding_dim>)`
- `source_doc_id UUID NULL REFERENCES documents(doc_id)`
- `source_row_id UUID NULL REFERENCES table_rows(row_id)`
- `metadata JSONB`

### 4.9.1 Business taxonomies and aliases

Implement `product_categories`, `service_categories`, `faq_categories`, and
`category_aliases`. Business records keep both the normalized category code and the original
source category. Implement `product_aliases`, `service_aliases`, and `faq_aliases` for
query/entity variants.

### 4.10 Conversation Memory Tables

Conversation memory is normalized into sessions, turns and summaries. It is separate from
`rag_traces`.

#### `conversation_sessions`

- `session_id TEXT PRIMARY KEY`
- `created_at TIMESTAMP`
- `updated_at TIMESTAMP`
- `metadata JSONB`

`metadata.state` stores the current canonical conversation state:

- active product/service IDs and names;
- active domain and topic;
- recent intents and typed filters;
- pending clarification.

#### `conversation_turns`

- `turn_id UUID PRIMARY KEY`
- `session_id TEXT REFERENCES conversation_sessions(session_id)`
- `role TEXT`
- `content TEXT`
- `detected_intents TEXT[]`
- `entities JSONB`
- `resolved_ids JSONB`
- `trace_id UUID NULL REFERENCES rag_traces(trace_id)`
- `created_at TIMESTAMP`

User and assistant turns for one exchange share the final post-binding/post-resolution context.
Do not persist rejected planner entities.

#### `conversation_summaries`

- `summary_id UUID PRIMARY KEY`
- `session_id TEXT REFERENCES conversation_sessions(session_id)`
- `summary TEXT`
- `last_turn_id UUID NULL REFERENCES conversation_turns(turn_id)`
- `metadata JSONB`
- `updated_at TIMESTAMP`

Summaries are compact navigation state for future prompts, not evidence for factual answers.

### 4.11 Trace and Evaluation Tables

Implement these tables:

#### `ingestion_runs`

- `run_id UUID PRIMARY KEY`
- `doc_id UUID`
- `started_at TIMESTAMP`
- `ended_at TIMESTAMP`
- `status TEXT`
- `parser_name TEXT`
- `parser_version TEXT`
- `total_chunks INT`
- `total_tables INT`
- `total_table_rows INT`
- `total_assets INT`
- `total_embeddings INT`
- `error_message TEXT`
- `quality_report JSONB`

#### `rag_traces`

- `trace_id UUID PRIMARY KEY`
- `session_id TEXT`
- `user_query TEXT`
- `detected_intent TEXT`
- `confidence FLOAT`
- `total_latency_ms INT`
- `status TEXT`
- `final_answer JSONB`
- `created_at TIMESTAMP`

#### `rag_trace_steps`

- `step_id UUID PRIMARY KEY`
- `trace_id UUID REFERENCES rag_traces(trace_id)`
- `step_name TEXT`
- `input JSONB`
- `output JSONB`
- `latency_ms INT`
- `status TEXT`
- `error_message TEXT`
- `created_at TIMESTAMP`

#### `evaluation_datasets`

- `dataset_id UUID PRIMARY KEY`
- `name TEXT`
- `version TEXT`
- `description TEXT`
- `content_hash TEXT`
- `metadata JSONB`
- `created_at TIMESTAMP`

#### `evaluation_cases`

- `case_id UUID PRIMARY KEY`
- `dataset_id UUID REFERENCES evaluation_datasets(dataset_id)`
- `case_key TEXT`
- `query TEXT`
- `expected_intent TEXT`
- `expected_answer_type TEXT`
- `expected_doc_ids UUID[]`
- `expected_chunk_ids UUID[]`
- `expected_row_ids UUID[]`
- `expected_asset_ids UUID[]`
- `expected_entities JSONB`
- `expected_source_keys JSONB`
- `expected_answer_contains JSONB`
- `forbidden_answer_contains JSONB`
- `expected_answer JSONB`
- `metadata JSONB`

#### `evaluation_runs`

- `eval_run_id UUID PRIMARY KEY`
- `dataset_id UUID`
- `pipeline_version TEXT`
- `data_version TEXT`
- `started_at TIMESTAMP`
- `ended_at TIMESTAMP`
- `metrics JSONB`
- `config_snapshot JSONB`
- `status TEXT`

#### `evaluation_case_results`

- `result_id UUID PRIMARY KEY`
- `eval_run_id UUID REFERENCES evaluation_runs(eval_run_id)`
- `case_id UUID REFERENCES evaluation_cases(case_id)`
- `trace_id UUID NULL REFERENCES rag_traces(trace_id)`
- `query TEXT`
- `expected_intent TEXT`
- `actual_intent TEXT`
- `status TEXT`
- `passed BOOLEAN`
- `latency_ms FLOAT`
- `expected_ids JSONB`
- `retrieved_ids JSONB`
- `answer_text TEXT`
- `scores JSONB`
- `violations JSONB`
- `details JSONB`
- `error_message TEXT`

---

## 5. Ingestion Pipeline Requirements

Implement the ingestion flow in `app/ingestion/pipeline.py`.

### 5.1 Flow

The ingestion pipeline must do the following:

1. Accept uploaded file path and calculate SHA-256 checksum.
2. Apply an explicit duplicate policy: `reject`, `reuse`, `replace`, or `force`.
3. Create `doc_id` and `ingestion_run_id`.
4. Insert a record into `documents` with status `draft`.
5. Insert a record into `ingestion_runs` with status `running`.
6. Parse the file using Docling when supported.
7. Normalize parsed output into separate collections:
   - text blocks
   - table blocks
   - image/asset blocks
   - metadata
8. Auto-detect document type from table schemas and text signals; preserve the explicit upload
   type as an override and require review when it conflicts with inferred content.
9. Preserve page number, section title, source reference, bbox, and content type where available.
10. Accept optional companion image uploads and match their basenames against
    `image_reference` columns.
11. Stage images/assets in `assets/.staging/<doc_id>/`.
12. Generate deterministic identity and replace image positions with stable tokens:
    - default format: `[asset:<stable_hash>]`
    - preserve concrete source tokens where available
    - ignore template placeholders such as `[asset:product_XX]`
13. Store asset records in `assets`.
14. Process tables without sending them to the text chunker:
    - store full table in `tables`
    - convert each table row into `table_rows`
    - create `row_text` for embedding
    - detect product/service/FAQ/clinic_info using normalized multilingual column aliases
    - store classification confidence, reasons, column mapping, and review requirement
    - sync product rows to `products`
    - sync service rows to `services`
    - sync question/answer rows to `faqs`
    - sync key/value and schedule rows to `clinic_info`
    - deduplicate business rows by normalized product/service name, FAQ question, or clinic key
    - distinguish product/service schemas using entity-specific signals before shared columns:
      `duration`, `symptoms`, `indications`, and `contraindications` are service signals;
      `brand`, `model`, `quantity`, and product links are product signals
    - never assign high product confidence from generic `name + price/category` alone
15. Chunk only normalized text blocks with `RecursiveCharacterTextSplitter`.
16. Embed chunks, table rows, and FAQ questions.
17. Store embeddings in pgvector without padding or truncating vectors.
18. Link each asset to all chunks containing its token in `chunk_assets`; set `assets.chunk_id`
    to the first matching chunk for compatibility.
19. Link companion images directly to product/service records when an image reference exists.
20. Run quality checks and blocking ingestion smoke checks.
21. Set document status to `active` only when auto-approval is enabled and there are no review
    or smoke-check reasons; otherwise set `review_required`.
22. Promote staged assets immediately before the final database commit.
23. For `replace`, archive previous active versions only after the replacement passes.
24. Update `ingestion_runs` with metrics, stage latency, classification details, and smoke report.
25. On an exception, rollback content records, remove staged/final files for the failed run, and
    mark the document and ingestion run as `failed`.

### 5.2 Important Rules

- Do not use LLM summary during ingestion.
- Do not call Vision LLM for image captioning by default.
- Image captioning may exist as an optional future feature, disabled by default.
- Do not split tables like normal text.
- Store both full table and row-level records.
- If a table row contains a product image token, connect the product to the asset.
- Prefer concrete asset tokens found in table rows when linking product/service records.
- A source asset may belong to multiple chunks; do not rely only on `assets.chunk_id`.
- Validate the embedding model dimension against `EMBEDDING_DIM` and PostgreSQL `vector(n)`
  columns at application startup.
- Strict embedding is the default. Never silently pad, truncate, or skip a failed vector insert.
- A low-confidence or unknown table classification must require review.
- Keep `review_reasons`, `review_only_reasons`, and `approval_blocking_reasons` separate.
  Review-only reasons may be acknowledged by an explicit manual approval. Integrity blockers
  may not be waived.
- Approval must rerun smoke checks, recompute business-row validation from current persisted
  rows, reject unresolved integrity blockers, and update every related status in one shared
  function. Do not rely only on a possibly stale `quality_report`.
- Do not activate two documents with the same checksum unless explicitly supported by a future
  versioning policy.
- Document checksum dedup and business-record dedup are separate. Within one ingestion, sync
  only one business record per normalized key. On successful activation, archive/disable older
  active business records with that key before activating the new version.
- PostgreSQL partial unique indexes must enforce one active normalized product name, service
  name, FAQ question, and clinic-info key.
- If parsing fails, update document and ingestion run status to `failed`.

### 5.3 Ingestion Smoke Checks

Before auto-approval and before manual approval, verify:

- at least one retrievable chunk, table row, or FAQ exists;
- every required chunk and table row has an embedding;
- stored table row count matches the normalized full-table row count;
- no table row remains unclassified;
- every asset is linked to retrievable content through `chunk_assets` or a normalized
  product/service `asset_id`;
- every concrete token in chunks resolves to an asset record;
- every asset file exists in staging or final storage;
- every product/service keeps its source row lineage;
- no other active document has the same checksum, except versions being replaced.

Any failed smoke check prevents activation and is written to
`quality_report.smoke_test.blocking_reasons`.

### 5.4 Ingestion Quality Report

Generate a JSON report like this:

```json
{
  "total_chunks": 24,
  "empty_chunks": 0,
  "tables_found": 2,
  "table_rows_created": 20,
  "assets_found": 10,
  "assets_resolved": 10,
  "products_created": 10,
  "services_created": 0,
  "embedding_success": 54,
  "embedding_failed": 0,
  "embedding_backend": "BAAI/bge-m3",
  "table_classifications": [],
  "review_reasons": [],
  "review_only_reasons": [],
  "approval_blocking_reasons": [],
  "smoke_test": {
    "passed": true,
    "checks": {},
    "blocking_reasons": [],
    "warnings": []
  },
  "stage_traces": [],
  "warnings": []
}
```

---

## 6. Asset Masking and Asset Resolver Requirements

### 6.1 Asset Storage

Implement `app/assets/storage.py`.

Responsibilities:

- Stage extracted images in `assets/.staging/<doc_id>/`.
- Validate image files where applicable.
- Promote a completed document directory to `assets/<doc_id>/`.
- Remove staged/final files when ingestion rolls back.
- Generate stable `asset_id`.
- Generate deterministic `stable_asset_key`.
- Generate stable `asset_token`.
- Return local path and public URL.

Token format:

```text
[asset:<stable_hash>]
```

Example:

```text
[asset:8d7f2d03d53e411fa58ddbc61e058de1]
```

An authored token such as `[asset:test_product_01]` may be retained. Placeholder examples such
as `[asset:test_product_XX]` must not be treated as concrete assets.

### 6.2 Asset Resolver

Implement `app/assets/resolver.py`.

Responsibilities:

- Detect asset tokens inside retrieved text.
- Accept explicit asset UUIDs from structured result items.
- Query `assets` table.
- Return resolved assets.
- Report missing assets.
- Record latency and trace step.
- Resolve from both response text tokens and `result.items[].asset_ids`; direct SQL list/detail
  responses must not depend on `chunk_assets`.
- Deduplicate the final asset list by `asset_id`.
- Filter both the asset and its owning document to active status.

Output example:

```json
{
  "text": "Bàn chải Oral-B. Ảnh: [asset:8d7f2d03d53e411fa58ddbc61e058de1]",
  "assets": [
    {
      "asset_id": "3e924226-4fa6-4d68-8981-8cce7768a933",
      "token": "[asset:8d7f2d03d53e411fa58ddbc61e058de1]",
      "url": "/assets/8a7a3b4e-894c-4ca7-a222-6b86f350f13a/8d7f2d03d53e411fa58ddbc6.png",
      "type": "product_image"
    }
  ],
  "missing_assets": []
}
```

---

## 7. Retrieval Pipeline Requirements

Implement retrieval in `app/retrieval/`.

### 7.1 Router

Implement `app/retrieval/router.py`.

There are two routing surfaces:

- In the default evidence-first path, `app/orchestration/task_planner.py` is the primary
  RouterLLM/task decomposer. It may emit multiple tasks for one message.
- `app/retrieval/router.py` remains the single-intent router used by the legacy path and by
  deterministic helper logic. It must not become a second competing authority after a
  multi-task plan has been accepted.

Supported intents:

```python
GREETING
CHITCHAT
CLINIC_INFO
FAQ
PRODUCT_LIST
PRODUCT_DETAIL
PRODUCT_COMPARE
SERVICE_LIST
SERVICE_DETAIL
UNKNOWN
```

Router output schema:

```json
{
  "intent": "PRODUCT_DETAIL",
  "confidence": 0.91,
  "entities": ["Oral-B"],
  "needs_rag": false,
  "needs_clarification": false
}
```

Routing rules:

- RouterLLM is the primary router when `ENABLE_LLM_ROUTER=true`.
- Do not skip RouterLLM only because a deterministic rule has high confidence.
- Route by the user's task, not only by entity text. A product/service entity plus risk,
  safety, usage, post-treatment, or "có ... không" wording should normally be FAQ.
- `GREETING` and `CHITCHAT` use no-RAG LLM generation; they must not use retrieval, SQL, or
  RAG context. Safe templates are allowed only as error fallbacks.
- Natural list/filter wording must route to list intents when no exact entity is being asked
  about. Examples: "sản phẩm nào dưới 2 triệu", "dịch vụ nào liên quan đến implant",
  "dịch vụ thời gian ngắn nhất".
- If RouterLLM confidence is below the global confidence threshold or asks for clarification,
  ask the user to clarify.
- RouterLLM output validation must be tolerant of realistic model output: normalize unsupported
  entity types to `unknown`, accept null/empty entity names, and filter null names before entity
  resolution.
- If RouterLLM returns malformed JSON or schema-invalid JSON, retry once with a JSON-fix prompt
  before falling back.
- With `ENABLE_PLAN_REVIEW=true`, run a second RouterLLM review over the proposed task plan. The
  reviewer must correct list/detail confusion, preserve every clause in a multi-intent query and
  classify pain, risk, safety, usage and post-treatment questions as FAQ tasks.
- RouterLLM prompts should include only query-relevant active product/service/category names,
  not the full catalog by default.
- Always log router latency, prompt size, raw model output, validated output, reason code,
  answer strategy, and selected entities.

Router implementation:

1. Ollama RouterLLM, enabled by config, is the normal path.
2. RouterLLM request timeout is disabled for local pipeline measurement; keep latency in trace.
3. Deterministic fallback is retained only as a minimal safe fallback when RouterLLM fails or
   the circuit breaker is open. It may handle obvious greetings, chitchat, clinic info, explicit
   product/service list requests, and clear dental FAQ patterns. It must not behave as a full
   parallel rule-based router for product/service detail decisions; otherwise return `UNKNOWN`
   with clarification.

### 7.2 Entity Extractor

Implement `app/retrieval/entity_extractor.py`.

Extract product/service names from the user query.

Entity processing has three distinct stages:

1. `app/ner/entity_span_extractor.py` finds text spans explicitly present in the original query.
2. `app/orchestration/context_binder.py` decides which task each explicit, inherited or remembered
   entity belongs to.
3. `app/retrieval/entity_resolver.py` maps the resulting mentions to authoritative database IDs.

`app/retrieval/entity_extractor.py` and fuzzy catalog matching support legacy/single-intent
retrieval. They do not replace task-level binding in the evidence-first path.

For initial version:

- Use simple regex/fuzzy matching against active `products.name` and `services.name`.
- Structured matching must combine normalized containment, meaningful-token coverage and fuzzy
  similarity. Do not reject the correct longer service name merely because several unrelated
  names share a generic token such as `răng`.
- Support multiple entities for compare queries.
- Trace all considered candidates with entity ID, name, score, and match type, not only the
  selected or ambiguous candidates.
- Tune `ENTITY_MATCH_THRESHOLD` and `ENTITY_AMBIGUITY_MARGIN` from traces/evaluation data.
  Increasing the ambiguity margin creates more ambiguous matches, not fewer.

### 7.3 Task-Level Context Binding

Implement `app/orchestration/context_binder.py`.

Inputs:

- original user query;
- PlannerLLM task plan;
- explicit spans from GLiNER/catalog/constraint extraction;
- normalized conversation state;
- entities resolved by earlier tasks in the same user message.

Outputs:

- one immutable binding decision per task;
- candidate entity names/verified inherited IDs and rejected Planner entities;
- binding source: explicit span, same-turn task, conversation state, planner, or none;
- reference mode: explicit, implicit, mixed, compare, filter refinement, or no entity;
- clarification state when required.

The binder must reject PlannerLLM-invented entities that are absent from both the original query
and verified context. It must not mutate `PlannedTask`. Database resolution and
`TaskCanonicalizer` consume the decision and atomically construct a `BoundTask`, removing rejected
entities from the effective query and typed filters before execution.

Binding must preserve multi-intent structure. It may link two tasks to the same entity, but it
must not collapse them into one task or attach planner-global entities to every task. Same-turn
inheritance requires a source BoundTask that is resolved and passes the pre-tool consistency
contract.

### 7.4 Query Rewrite / HyDE

Implement `app/retrieval/query_rewrite.py`.

Rules:

- HyDE is optional and config-controlled.
- Do not use HyDE for obvious structured queries such as product list, service list, exact product detail, or clinic info.
- Use HyDE only for FAQ or unstructured RAG queries.
- Run HyDE from the post-binding effective task query, never from an unverified PlannerLLM entity.
- Hypothetical output is a retrieval aid only. Do not save it into conversation memory or expose
  it as evidence.
- Trace original query, rewritten query, latency, and whether HyDE was used.

### 7.5 Dense Retriever

Implement `app/retrieval/dense_retriever.py`.

Responsibilities:

- Embed user query.
- Search `chunks.embedding`, `table_rows.embedding`, and `faqs.embedding` depending on intent.
- Return top-k results with scores and source type.

### 7.6 Sparse Retriever

Implement `app/retrieval/sparse_retriever.py`.

Responsibilities:

- Use PostgreSQL full-text search.
- Search `chunks.content_tsv`, `table_rows.row_tsv`, and `faqs.question_tsv` depending on intent.
- Return top-k results with scores and source type.

### 7.7 Structured Retriever

Implement `app/retrieval/structured_retriever.py`.

Responsibilities:

- `CLINIC_INFO`: query `clinic_info`.
- `PRODUCT_LIST`: parse safe typed filters/sorts and query matching active products.
- `PRODUCT_DETAIL`: exact/fuzzy match product name.
- `PRODUCT_COMPARE`: retrieve each product entity separately.
- `SERVICE_LIST`: parse safe typed filters/sorts and query matching active services.
- `SERVICE_DETAIL`: exact/fuzzy match service name.
- `FAQ`: exact/fuzzy FAQ search before semantic fallback.

Rules:

- Always filter `status = 'active'` or `is_active = true`.
- Never return archived/stale records.
- Product/service list filters must be typed and whitelisted. Do not let the LLM generate SQL.
- Service filtering must use `category_code`, `source_category`, service name and normalized
  service category aliases. A query about implant must not return cosmetic services unless the
  actual service evidence matches implant-related fields.

### 7.8 RRF Fusion

Implement `app/retrieval/rrf.py`.

Use Reciprocal Rank Fusion to combine dense, sparse, and structured results.

Formula:

```text
score = sum(1 / (k + rank_i))
```

Default `k = 60`.

Trace:

- input result IDs
- fused ranking
- latency

### 7.9 Reranker

Implement `app/retrieval/reranker.py`.

Requirements:

- Reranker must be optional through config.
- If model is not available, fallback without crashing.
- Rerank top N fused results to final top K.
- Trace before/after ranking and latency.

### 7.10 Context Builder

Implement `app/retrieval/context_builder.py`.

Responsibilities:

- Convert retrieved chunks/rows/products/services/FAQs into clean context.
- Deduplicate repeated context.
- Include source metadata.
- Include `row_json` when the source is table row or business object.
- Keep asset tokens unchanged.
- Limit context length.

Context object example:

```json
{
  "items": [
    {
      "source_type": "product",
      "source_id": "product_001",
      "text": "Sản phẩm: Bàn chải điện Oral-B. Giá: 850000. Ảnh: [asset:8d7f2d03d53e411fa58ddbc61e058de1]",
      "raw_json": {},
      "source": {
        "doc_id": "doc_001",
        "file_name": "bang_san_pham.pdf",
        "page_number": 2
      }
    }
  ]
}
```

---

## 8. Generation Pipeline Requirements

Implement generation in `app/generation/`.

### 8.1 LLM Client

Implement `app/generation/llm_client.py` and keep `app/generation/ollama_client.py` as a
compatibility wrapper for older imports.

Responsibilities:

- Select provider through `LLM_PROVIDER`.
- Support `ollama` for local demo and `openai` for production.
- Support configurable router/generation model names.
- Support provider-specific token/context options without leaking them into orchestration.
- Return text and latency.
- Fail gracefully with useful error messages.

### 8.2 Prompt Builder

Implement `app/generation/prompts.py`.

System prompt:

```text
Bạn là chatbot hỗ trợ khách hàng cho phòng khám nha khoa.
Bạn chỉ được trả lời dựa trên dữ liệu được cung cấp trong retrieved_context.
Không tự bịa giá, dịch vụ, sản phẩm, lịch làm việc, chính sách hoặc thông tin y tế.
Nếu dữ liệu không có, hãy nói: "Hiện tại tôi chưa có đủ thông tin trong dữ liệu của phòng khám."
Với câu hỏi về sản phẩm/dịch vụ, ưu tiên trả lời ngắn gọn, có cấu trúc.
Với câu hỏi so sánh, chỉ so sánh các thuộc tính có trong dữ liệu.
Với câu hỏi y tế, chỉ tư vấn định hướng, không chẩn đoán chắc chắn, và khuyến nghị gặp nha sĩ nếu có dấu hiệu nguy hiểm.
Giữ nguyên asset token dạng [asset:...] trong phần text.
Trả về JSON hợp lệ theo schema được yêu cầu.
Không viết markdown bên ngoài JSON.
```

Do not ask the model to reveal chain-of-thought. Use structured output only.

### 8.3 Response Schema

Implement Pydantic schemas in `app/generation/schemas.py`.

Required output shape:

```json
{
  "intent": "PRODUCT_DETAIL",
  "confidence": 0.93,
  "answer_type": "direct_data",
  "entities": [
    {
      "type": "product",
      "name": "Bàn chải điện Oral-B",
      "matched_id": "product_001"
    }
  ],
  "result": {
    "text": "Bàn chải điện Oral-B có giá 850.000đ. Ảnh: [asset:8d7f2d03d53e411fa58ddbc61e058de1]",
    "items": [
      {
        "type": "product",
        "id": "product_001",
        "name": "Bàn chải điện Oral-B",
        "chunk_id": "chunk_001",
        "asset_ids": ["3e924226-4fa6-4d68-8981-8cce7768a933"]
      }
    ],
    "assets": [],
    "sources": []
  },
  "safety": {
    "medical_disclaimer_required": false,
    "needs_human_support": false
  }
}
```

### 8.4 Validator

Implement `app/generation/validator.py`.

Validation requirements:

- JSON must parse.
- Intent must be in enum.
- Confidence must be between 0 and 1.
- Referenced `chunk_id`, `asset_id`, `product_id`, `service_id`, `doc_id`, or `row_id` must exist if provided.
- If response contains a price, that price must come from retrieved context or structured record.
- Normalize model-provided `result.items`, `result.sources`, and `entities[].matched_id` against
  the current context. If the model uses a product/service/FAQ name where a source ID is
  expected, map it to the matching context source ID instead of failing the whole answer.
- Accept common Vietnamese price wording such as `2,5 triệu` only when it maps to a numeric
  value present in context.
- Server-managed asset fields must be normalized before validation. The model may reference
  asset tokens, but final `result.assets` and missing asset records are resolved by the server.
- In the evidence-first path, the LLM must not construct result items, entities, sources or
  assets. Validate `used_source_ids` against current context and require source coverage for
  every planned task before the server builds the shared response schema.
- Validation must be semantic as well as structural. Reject an answer when it names a
  product/service that is not represented by the selected evidence for that task.
- Validate structured facts against authoritative evidence where applicable, including price,
  duration, quantity/stock, opening hours and explicit availability statements. Do not allow a
  valid source ID to excuse answer text about a different entity.
- Before generation, reject a task/evidence pack when its executable query, bound entities,
  resolved IDs, typed entity filters and authoritative evidence refer to different records.
- If JSON fails, retry once with a JSON-fix prompt.
- If retry fails, return safe fallback response.

### 8.5 Renderer

Implement `app/generation/renderer.py`.

Responsibilities:

- Resolve asset tokens.
- Attach final asset URLs.
- Return final API response.
- Include trace ID in response for debugging.

---

## 9. Chat API Requirements

Implement in `app/api/routes_chat.py`.

Endpoint:

```http
POST /chat
```

Request:

```json
{
  "message": "Tôi muốn xem bàn chải điện Oral-B",
  "session_id": "optional-session-id",
  "history": []
}
```

Response:

```json
{
  "trace_id": "trace_xxx",
  "intent": "PRODUCT_DETAIL",
  "answer_type": "rag",
  "degraded": false,
  "answer": {
    "text": "... [asset:8d7f2d03d53e411fa58ddbc61e058de1]",
    "assets": [
      {
        "asset_id": "3e924226-4fa6-4d68-8981-8cce7768a933",
        "url": "/assets/8a7a3b4e-894c-4ca7-a222-6b86f350f13a/8d7f2d03d53e411fa58ddbc6.png",
        "type": "product_image"
      }
    ],
    "items": [],
    "sources": []
  },
  "debug": {
    "enabled": false
  }
}
```

Add optional debug mode through config or request flag. Debug output must not expose sensitive data by default.

Current architecture notes:

- `routes_chat.py` must delegate the pipeline to `app/services/chat.py`; do not reimplement
  router, retriever, generator, or renderer logic inside the route.
- Every chat request must create a `rag_traces` record and per-stage `rag_trace_steps`.
- The default `ChatService` path is evidence-first when `ENABLE_MULTI_TASK_PLANNER=true`:
  memory load -> task planning -> entity span extraction -> context binding -> entity resolution
  -> task canonicalization -> BoundTask consistency -> tool execution -> evidence merging ->
  evidence consistency -> context builder -> no-RAG social generation or evidence synthesis ->
  semantic JSON validation/fallback -> memory save -> asset resolver -> response rendering.
- The legacy single-intent path is used only when `ENABLE_MULTI_TASK_PLANNER=false`:
  router -> entity extraction/resolution -> structured retrieval -> retrieval planning ->
  optional rewrite -> dense/sparse -> RRF -> optional reranker -> context builder -> prompt
  builder/generation or direct response -> JSON validation -> asset resolver -> rendering.
- Stages that are intentionally not run must still be recorded as `skipped` for debugging.
- `GREETING` and `CHITCHAT` must use `NO_RAG_LLM`: no retrieval context, no SQL data lookup,
  but still generated through the LLM and validated through the shared response schema.
- Direct SQL responses and LLM-generated responses must both pass through the shared renderer
  and asset resolver before returning to the API.
- In the evidence-first path, SQL rows, FAQ rows, chunks and table rows are all evidence. The
  executor must not return final user-facing answers.
- `ENABLE_EVIDENCE_SYNTHESIS=true` makes the LLM synthesize the final answer from the evidence
  pack. The prompt receives a compact evidence payload rather than full ORM/source metadata.
  Keep answer validation, per-task source validation and asset resolution in place.
- If synthesis generation or validation fails, use a grounded fallback built from focused
  evidence. The fallback may format product/service/list/detail/compare naturally, but it must
  never add prices, durations, availability or policy facts not present in evidence.
- Generation failures set `degraded=true`, `answer_type=fallback`, a `generation_fallback` trace
  step and overall trace status `degraded`. A valid empty list result is not a generation error.
- Conversation memory is session-scoped. Follow-up questions can reuse recent entities,
  resolved product/service IDs, active domain/topic, filters and intents when `session_id` is
  stable, but traces are not memory and must not be used as chat state.
- Persist only resolved, registry-eligible BoundTasks that passed the evidence gate. Planner-only
  entities, list result rows and failed/stale task state must not be written into conversation
  state, turns or summaries.
- Synthesis receives task-scoped executable queries and evidence. It must not receive a stale
  planner entity as if it were an instruction. `planner_global_entities` is trace-only metadata.
- `entity_span_extraction`, `context_binding`, `entity_resolution`, `task_canonicalization`,
  `bound_task_consistency` and `evidence_consistency` are separate trace steps.
  Trace Explorer must make it possible to compare the PlannerLLM proposal with explicit spans,
  each binding decision, rejected planner entities, executable task state, and the final database
  resolution result.
- Do not ask the LLM for chain-of-thought. If debugging is needed, expose evidence, sources,
  missing_info and conflicts through trace/debug output.

---

## 10. Ingestion API Requirements

Implement in `app/api/routes_ingestion.py`.

Endpoints:

```http
POST /ingest/upload
POST /ingest/run
GET /ingest/runs/{run_id}
GET /documents/{doc_id}
POST /documents/{doc_id}/approve
POST /documents/{doc_id}/archive
POST /documents/{doc_id}/tables/{table_id}/classify
```

Upload options must include:

- `document_type`
- `extract_tables`
- `extract_assets`
- `create_embeddings`
- `require_review`
- `duplicate_policy`

Approval is validation-gated:

- upload document
- parse document
- status becomes `active` only when auto-approval is enabled and all checks pass
- otherwise status becomes `review_required`
- approve endpoint reruns smoke checks
- approve endpoint recomputes current business validation and merges it with smoke blockers
- review-only reasons can be acknowledged by manual approval; integrity blockers cannot
- failed approval returns HTTP 409 with the combined approval report
- successful approval updates `documents`, `chunks`, `tables`, `table_rows`, `assets`,
  `products`, `services`, `clinic_info`, and source-linked FAQs

Also support config:

```env
AUTO_APPROVE_INGESTION=true
```

When enabled, parsed records can become active automatically for demo.
It must not bypass classification review, duplicate protection, embedding validation, or smoke
checks.

### 10.1 Control Center API

`app/api/routes_control_center.py` exposes the operational API consumed by `frontend/`.
These endpoints are part of the project architecture and should stay compatible with the
frontend types in `frontend/src/types/` and API clients in `frontend/src/api/`.

Control Center endpoint groups:

- `GET /api/documents`, `GET /api/documents/{doc_id}`;
- `POST /api/documents/{doc_id}/approve`, `/activate`, `/archive`, `/reingest`;
- `GET /api/ingestion/runs`, `GET /api/ingestion/runs/{run_id}`,
  `GET /api/ingestion/summary`;
- `POST /api/retrieval/debug`;
- `GET /api/evaluation/datasets`, `/runs`, `/cases`, `/results`, `/summary`;
- `GET /api/observability/health`, `/metrics`, `/diagnostics`, `/errors`;
- `GET /api/traces`, `GET /api/traces/{trace_id}`;
- `GET /api/assets`, `GET /api/assets/{asset_id}`;
- `GET /api/products`, `/services`, `/faqs`, `/clinic-info`, `/tables`, `/table-rows`,
  `/chunks`;
- `GET /api/settings`.

Control Center rules:

- These endpoints are operational/debugging surfaces, not separate business logic paths.
- Retrieval debug must reuse `IntentRouter`, `DatabaseEntityResolver`, `RetrievalPlanner`,
  `StructuredRetriever`, dense/sparse retrievers, RRF, reranker, and context builder.
- Document activation from Control Center must call the same review/approval functions as the
  ingestion API.
- Read-only settings should reflect backend config; do not add runtime mutation unless the
  backend has an explicit safe update mechanism.
- Do not add destructive delete actions unless the backend supports safe deletion and asset
  cleanup semantics.

---

## 11. Evaluation Requirements

Implement evaluation scripts and endpoints.

### 11.1 Component-Level Evaluation

Implement these modules:

- `eval_router.py`
- `eval_retrieval.py`
- `eval_generation.py`
- `eval_assets.py`
- `eval_e2e.py`

Metrics required:

#### Router

- accuracy
- precision/recall/F1 if enough labels
- confusion matrix
- low confidence rate
- clarification rate

#### Retrieval

- Hit@1
- Hit@3
- Recall@5
- Recall@10
- MRR@10
- nDCG@10 if possible

#### RRF / Reranker

- top-1 accuracy before rerank
- top-1 accuracy after rerank
- rerank improvement rate
- latency added

#### Generation

- JSON validity rate
- schema pass rate
- unsupported claim rate using simple source matching
- answer correctness where expected answer exists
- safety pass rate for medical-style queries

#### Asset Resolver

- asset resolve success rate
- missing asset rate
- broken local file rate
- wrong asset rate if expected asset is provided

#### End-to-End

- success rate
- pass rate based on required checks, not on every reporting metric being exactly 1.0
- fallback rate
- no-result rate
- clarification rate
- average latency
- p50 latency
- p95 latency
- p99 latency

Metrics that require ground truth must return `null`/`N/A` when ground truth is absent.
They must never treat an empty expected set as a perfect score. Report ground-truth coverage
beside each metric family.

### 11.2 Evaluation Dataset

Create `eval_datasets/dental_basic_eval.jsonl` with balanced, grounded cases.

Include at least these groups:

- clinic info
- product list
- product detail
- product compare
- service list
- service detail
- FAQ
- unknown/negative queries

Each grounded case should use stable source keys rather than hard-coded UUIDs where possible:

- `product:<name>`
- `service:<name>`
- `faq:<question>`
- `clinic_info:<key>`
- `asset:<stable_asset_key-or-token>`
- `document:<checksum>`
- `chunk:<document_checksum>:<chunk_index>`
- `table_row:<document_checksum>:<entity_name>`

Include `expected_entities`, `expected_answer_contains`, safety metadata and asset expectations
where applicable. Keep at least three cases per supported intent.

Example line:

```json
{"case_key":"product_detail_aquajet","query":"Cho tôi thông tin AquaJet Mini Water Flosser","expected_intent":"PRODUCT_DETAIL","expected_answer_type":"direct_data","expected_entities":["AquaJet Mini Water Flosser"],"expected_source_keys":["product:AquaJet Mini Water Flosser"],"expected_answer_contains":["AquaJet Mini Water Flosser"],"metadata":{"group":"product_detail","expect_assets":true}}
```

The default dataset contains 30 balanced cases. Fixture CSV files live under
`eval_datasets/fixtures/` and are seeded through `scripts/seed_evaluation_fixtures.py`.

Create conversation-level evaluation data in
`eval_datasets/dental_conversation_scenarios.json`. It must contain 10-20 scenarios, each with
4-6 user turns. All turns in one scenario must run with the same evaluation `session_id` so the
pipeline tests conversation memory, follow-up resolution, multi-task planning and intent shifts,
not only isolated single-query behavior.

### 11.3 Evaluation API

Implement in `app/api/routes_evaluation.py`:

```http
POST /evaluation/run
GET /evaluation/runs/{eval_run_id}
```

Supported profiles:

- `deterministic`: disable LLM router, HyDE and reranker; cap Ollama timeout for repeatable
  baseline evaluation.
- `production`: use the runtime configuration exactly as deployed.

The evaluation runner must store aggregate metrics in `evaluation_runs`, config in
`config_snapshot`, and one record per case in `evaluation_case_results`. It must also generate
diagnostic alerts from case results and trace-stage failures/latencies.
When a case has `metadata.conversation_session_key` or `metadata.scenario_key`, the runner must
reuse one `session_id` for all turns in that scenario.
Conversation evaluation must score expected entities against the post-binding task plan, verify
that cases marked `requires_follow_up_memory` actually use conversation state or inherited task
context, and verify that cases marked `expects_multi_task` produce at least two planned tasks.
Expose aggregate entity-binding accuracy, follow-up success, multi-task success, GLiNER degraded
rate and whole-scenario pass rate in the Evaluation Dashboard. Each case result should retain the
span provider/output, binding decisions, memory state and planned tasks needed to explain a
failure, with a deep link to its trace.

---

## 12. Observability and Tracing Requirements

Implement in `app/observability/`.

### 12.1 Required Internal Tracing

Every chat request must create:

- one row in `rag_traces`
- multiple rows in `rag_trace_steps`

Required steps:

```text
memory_load
task_planning
entity_span_extraction
context_binding
entity_resolution
task_canonicalization
bound_task_consistency
tool_execution
evidence_merging
evidence_consistency
context_builder
prompt_builder
synthesis_generation
json_validation
generation_fallback
memory_save
asset_resolver
response_rendering
```

The legacy single-intent path additionally uses:

```text
router_intent
entity_extraction
retrieval_planning
query_rewrite_hyde
structured_retrieval
dense_retrieval
sparse_retrieval
rrf_fusion
reranker
llm_generation
```

If a step is skipped, record it with status `skipped` and a short reason.

### 12.2 Required Latency Tracking

Every step must record:

- start time
- end time
- latency_ms
- status
- error_message if failed

### 12.3 Optional Langfuse Integration

If these env vars are present:

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

then send traces to Langfuse.

If not present, the app must continue working using only internal PostgreSQL traces.

### 12.4 Optional OpenTelemetry

If feasible, add OpenTelemetry instrumentation for FastAPI and database calls.

If not feasible in the first implementation, leave clear TODOs and keep internal tracing fully working.

---

## 13. Business Logic by Intent

These sections define task behavior, not mutually exclusive whole-request routes. A multi-intent
message may execute several sections in one request. Each task retrieves its own evidence and the
final synthesis combines the task outputs without changing their entity bindings.

### 13.1 GREETING

No RAG required.

Use no-RAG LLM generation through the shared JSON response schema. Return a short helpful
greeting. A static greeting is allowed only as an error fallback.

### 13.2 CHITCHAT

No RAG required.

Use no-RAG LLM generation through the same structured response schema. Keep the response short
and steer the user back to dental support. A safe template is allowed only as an error fallback
when LLM generation or JSON validation fails.

### 13.3 CLINIC_INFO

Use direct SQL from `clinic_info`.

Examples:

- phone
- email
- address
- opening hours
- Facebook/Zalo links

SQL remains authoritative. Evidence synthesis may use the generation LLM for natural wording,
but it must not modify clinic facts.

### 13.4 FAQ

Order:

1. Exact/fuzzy FAQ match.
2. FAQ-only semantic/sparse search when direct matching is insufficient.
3. Add document/table evidence only when the planner/tool executor requests broader RAG context.
4. With evidence synthesis enabled, use the LLM to answer naturally from the curated FAQ and
   retrieved evidence.
5. If synthesis fails, return a grounded fallback from the best FAQ evidence.

FAQ must stay grounded. It may paraphrase and combine evidence, but it must not invent clinical
claims beyond FAQ/document evidence.

### 13.5 PRODUCT_LIST

Use direct SQL as authoritative evidence:

```sql
SELECT * FROM products WHERE status = 'active';
```

Parse safe typed filters for category/product, price range and sorting by price, name, category
or quantity. If the user requests a filter/sort operation without enough detail, ask for category
and sort direction instead of generating SQL with an LLM. When evidence synthesis is enabled,
the final answer may be natural language plus structured `result.items`; the SQL rows remain the
source of truth.

### 13.6 PRODUCT_DETAIL

Order:

1. Extract entity.
2. Exact/fuzzy SQL match on `products.name`.
3. Return product details and assets when one authoritative record is resolved.
4. If evidence synthesis is enabled, use the LLM to phrase the answer from the product evidence.
5. If no unique record is resolved, ask the user to clarify. Do not use general RAG to invent a
   product match.

### 13.7 PRODUCT_COMPARE

Order:

1. Extract multiple product entities.
2. Retrieve each product separately.
3. If both/multiple products found, use LLM to produce comparison based only on retrieved structured rows.
4. If one product is missing, ask the user to clarify or say the product is not available in current data.

### 13.8 SERVICE_LIST

Use direct SQL as authoritative evidence:

```sql
SELECT * FROM services WHERE status = 'active';
```

Parse safe typed filters for service category aliases, service name, price range, duration range
and sorting by price, duration, category or name. For example, "Dịch vụ nào liên quan đến
implant?" must filter to implant-related services, not return every active service.

### 13.9 SERVICE_DETAIL

Order:

1. Extract entity.
2. Exact/fuzzy SQL match on `services.name`.
3. Return service details when one authoritative record is resolved.
4. If evidence synthesis is enabled, use the LLM to phrase the answer from the service evidence.
5. If no unique record is resolved, ask the user to clarify. Do not use general RAG to invent a
   service match.

### 13.10 UNKNOWN

If router confidence is low, ask the user to clarify.

Example response:

```text
Mình chưa hiểu rõ bạn muốn hỏi về sản phẩm, dịch vụ, FAQ hay thông tin phòng khám. Bạn có thể hỏi cụ thể hơn không?
```

---

## 14. Environment Configuration

Create `.env.example` with at least:

```env
# App
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
AUTO_APPROVE_INGESTION=true
DUPLICATE_INGESTION_POLICY=reject
TABLE_CLASSIFICATION_THRESHOLD=0.85
ENABLE_LLM_ROUTER=true
ROUTER_TIMEOUT_SECONDS=0
ROUTER_FAILURE_THRESHOLD=2
ROUTER_CIRCUIT_BREAKER_SECONDS=60
ENABLE_MULTI_TASK_PLANNER=true
ENABLE_PLAN_REVIEW=true
ENABLE_EVIDENCE_SYNTHESIS=true
ENABLE_GLINER_NER=true
GLINER_MODEL=urchade/gliner_multi-v2.1
GLINER_THRESHOLD=0.4
GLINER_DEVICE=cpu
PRELOAD_GLINER_ON_STARTUP=true
ENABLE_CONTEXT_BINDER=true
CONTEXT_BINDER_STRICT_FOLLOW_UP=true
CONTEXT_BINDER_TRACE_DECISIONS=true
MAX_SUB_QUERIES=3
MAX_EVIDENCE_ITEMS=12
CONVERSATION_HISTORY_TURNS=8
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# PostgreSQL
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/dental_rag
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=dental_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# pgvector / embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_DEVICE=cpu
VALIDATE_EMBEDDING_ON_STARTUP=true
STRICT_EMBEDDING=true
ALLOW_EMBEDDING_FALLBACK=false

# LLM provider
LLM_PROVIDER=ollama

# Ollama demo provider
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GENERATION_MODEL=qwen2.5:14b-instruct-q4_K_M
OLLAMA_ROUTER_MODEL=qwen2.5:7b-instruct
OLLAMA_VISION_MODEL=llava:latest
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_ROUTER_TIMEOUT_SECONDS=0
OLLAMA_GENERATION_TIMEOUT_SECONDS=0
OLLAMA_KEEP_ALIVE=30m
OLLAMA_NUM_PREDICT=2048
OLLAMA_ROUTER_NUM_PREDICT=512
OLLAMA_GENERATION_NUM_PREDICT=2048
OLLAMA_ROUTER_NUM_CTX=8192
OLLAMA_GENERATION_NUM_CTX=16384

# OpenAI production provider
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_ROUTER_MODEL=gpt-4.1-nano
OPENAI_GENERATION_MODEL=gpt-4.1
OPENAI_TIMEOUT_SECONDS=0
OPENAI_ROUTER_TIMEOUT_SECONDS=0
OPENAI_GENERATION_TIMEOUT_SECONDS=0
OPENAI_MAX_TOKENS=2048
OPENAI_ROUTER_MAX_TOKENS=512
OPENAI_GENERATION_MAX_TOKENS=2048

# Retrieval
DENSE_TOP_K=20
DENSE_MIN_SCORE=0.25
SPARSE_TOP_K=20
SPARSE_TRIGRAM_THRESHOLD=0.2
SPARSE_MIN_FTS_RANK=0.001
SPARSE_MAX_PER_SOURCE=10
RRF_K=60
RRF_MAX_PER_SOURCE=4
STRUCTURED_RRF_WEIGHT=1.5
DENSE_RRF_WEIGHT=1.0
SPARSE_RRF_WEIGHT=1.0
RERANK_TOP_N=20
FINAL_TOP_K=5
ENABLE_HYDE=false
HYDE_TIMEOUT_SECONDS=15
ENABLE_RERANKER=false
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
PRELOAD_RERANKER_ON_STARTUP=true
MAX_CONTEXT_CHARS=16000
MAX_CONTEXT_ITEMS_PER_SOURCE=4
CONFIDENCE_THRESHOLD=0.65
STRUCTURED_DIRECT_THRESHOLD=0.9
FAQ_DIRECT_THRESHOLD=0.9
ENTITY_MATCH_THRESHOLD=0.72
ENTITY_AMBIGUITY_MARGIN=0.05
JSON_RETRY_COUNT=1

# Assets
ASSET_STORAGE_DIR=assets
ASSETS_DIR=assets
ASSET_PUBLIC_BASE_URL=/assets
PUBLIC_ASSETS_BASE_URL=/assets
UPLOAD_DIR=uploads

# Observability
ENABLE_LANGFUSE=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

# Evaluation
EVAL_DATASET_PATH=eval_datasets/dental_basic_eval.jsonl
```

The checked-in `.env.example` uses `qwen2.5:7b-instruct` for task planning/routing and
`qwen2.5:14b-instruct-q4_K_M` for evidence synthesis. Production or OpenAI-backed deployments
should change model settings through environment variables, not hardcoded code.

Important: after implementing the project, ask the user to provide the required environment variables, especially PostgreSQL connection values, before trying to run the program.

---

## 15. README Requirements

Update `README.md` with:

1. Project overview.
2. Architecture summary.
3. Setup instructions.
4. PostgreSQL + pgvector setup.
5. Ollama setup.
6. Environment variables.
7. How to run migrations.
8. How to start server.
9. How to ingest a document.
10. How to approve an ingested document.
11. How to chat.
12. How to run evaluation.
13. How to view traces.
14. Troubleshooting.

Include commands such as:

```bash
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

Also include example curl calls for:

```bash
POST /ingest/upload
POST /chat
POST /evaluation/run
```

### 15.1 Frontend Control Center Requirements

The frontend in `frontend/` is an operational Control Center for the backend. Keep it aligned
with `app/api/routes_control_center.py` and the core backend routes.

Frontend stack:

- Vite + React + TypeScript;
- React Router;
- TanStack Query;
- TailwindCSS;
- Recharts;
- Vitest + Testing Library.

Implemented pages:

- Chatbot with assistant cards, debug panel, assets and sources;
- Upload Documents;
- Document Store;
- Ingestion Monitor;
- Retrieval Playground;
- Evaluation Dashboard with basic, semantic and multi-turn dataset presets, conversation/entity
  binding metrics, scenario outcomes and trace deep links;
- Observability;
- Trace Explorer with task/token summaries, GLiNER spans, context-binding decisions,
  conversation state, pipeline timeline and raw JSON details;
- Asset Manager;
- Data Tables;
- Settings.

Frontend rules:

- API client code belongs in `frontend/src/api/`.
- Shared frontend types belong in `frontend/src/types/` and must match backend JSON contracts.
- Page-level UI should live under `frontend/src/components/<domain>/`.
- Do not add UI-only assumptions that contradict backend behavior. If the backend returns
  `N/A`, skipped trace steps, missing ground truth, or unavailable endpoints, render that state
  explicitly instead of treating it as a crash.
- Settings are read-only unless a backend runtime update endpoint is deliberately implemented.
- Do not add destructive document deletion controls until backend deletion and asset cleanup
  semantics exist.

Frontend verification commands:

```bash
cd frontend
npm run lint
npm run test
npm run build
```

---

## 16. Testing Requirements

Maintain tests for:

- asset token detection and resolution
- structured item asset-ID resolution when summary text has no token
- template asset token rejection
- stable asset identity and staging/promotion behavior
- asset-to-multiple-chunk linking
- Docling token unescaping and separated normalization
- table row serialization
- multilingual table classification and business-table synchronization
- generic `name` service catalogs with duration/symptom columns
- rejection of ambiguous `name + price/category` schemas
- embedding dimension mismatch failure
- ingestion smoke checks
- approval status cascade
- manual approval integrity-blocker enforcement
- duplicate ingestion policies
- RRF fusion
- router basic classification
- response schema validation
- direct SQL product/service retrieval
- trace step creation
- RouterLLM-first routing, tolerant entity parsing, JSON repair retry, and safe fallback
- multi-task planning, tool execution contracts and evidence merging
- task-scoped context binding for explicit, implicit, mixed, multi-entity and multi-intent cases
- rejection of stale PlannerLLM entities after authoritative memory/database binding
- consistency among effective task query, entity filters, resolved IDs and evidence
- semantic generation validation for entity and structured-fact grounding
- conversation memory persistence for session turns
- retrieval planning modes, including `NO_RAG_LLM` for greeting and chitchat
- active-status filters for dense/sparse/structured retrieval
- structured product-list parsing for category, price range, sort and ambiguity
- context deduplication and canonical-key fusion
- generation fallback and response rendering
- optional Langfuse behavior when disabled or partially configured
- frontend API clients, chat components, navigation/sidebar behavior, and rendering of missing
  backend data

Backend verification:

```bash
.venv/bin/ruff check app tests
.venv/bin/pytest
```

Frontend verification when changing `frontend/`, `/api/*` contracts, response schemas, or
Control Center payloads:

```bash
cd frontend
npm run lint
npm run test
npm run build
```

Current backend test coverage includes:

```text
tests/test_asset_resolver.py
tests/test_asset_storage.py
tests/test_business_dedup.py
tests/test_context_builder.py
tests/test_db_connection.py
tests/test_document_classifier.py
tests/test_entity_resolver.py
tests/test_entity_span_extractor.py
tests/test_evaluation.py
tests/test_generation_fallback.py
tests/test_ingestion_pipeline.py
tests/test_langfuse_client.py
tests/test_normalization.py
tests/test_orchestration.py
tests/test_query_rewrite.py
tests/test_retrieval_planner.py
tests/test_retriever_status_filters.py
tests/test_review_policy.py
tests/test_table_processor.py
tests/test_rrf.py
tests/test_router.py
tests/test_structured_query.py
tests/test_structured_retriever.py
tests/test_validator.py
tests/test_tracing.py
```

---

## 17. Implementation Priorities

The core project is implemented. For new work, extend it in this order:

1. Identify the owning layer first: ingestion, retrieval, generation, assets, evaluation,
   observability, API, frontend, or scripts.
2. Update the shared schema/contract before updating callers:
   database migration/model, Pydantic schema, API payload, frontend type, and tests.
3. Keep orchestration centralized:
   - chat orchestration in `app/services/chat.py`;
   - ingestion orchestration in `app/ingestion/pipeline.py`;
   - task planning, binding and evidence contracts in `app/orchestration/`;
   - approval/status cascade in `app/ingestion/review.py`;
   - operational API wiring in `app/api/routes_control_center.py`.
4. Prefer extending existing helpers over adding a parallel path:
   `RetrievalPlanner`, `StructuredRetriever`, `ContextBuilder`, `ResponseRenderer`,
   `AssetResolver`, `TraceRecorder`, and evaluation runners should remain the shared path.
5. Add or update tests at the same layer as the change. Broaden tests when a change crosses
   backend/frontend contracts or shared response schemas.
6. Update `README.md`, `AGENTS.md`, `.env.example`, and `frontend/README_FRONTEND.md` when
   behavior, configuration, or architecture changes.
7. Run backend checks and, when relevant, frontend checks before handing work back.

At every step, keep the application runnable and avoid introducing a second implementation of
an existing pipeline stage.

---

## 18. Critical Acceptance Criteria

The final implementation is accepted only if all of these are true:

1. The app starts with FastAPI.
2. Database migrations create all required tables.
3. The app can connect to PostgreSQL with pgvector enabled.
4. A document can be uploaded and ingested.
5. Ingestion creates document/chunk/table/row/asset records where applicable.
6. Normalization produces separate text, table, and image collections while preserving page and
   section metadata.
7. Tables are excluded from normal text chunking and remain available as full tables and rows.
8. Asset Masking works: text contains a concrete `[asset:...]` token, every token resolves to a
   real URL/path, and template placeholders are ignored.
9. Assets are staged, promoted only for completed runs, and linked to every matching chunk.
10. Product/service table rows can be stored, queried, and linked to source rows/assets.
11. Embedding dimension is validated against both model output and PostgreSQL columns.
12. Embedding failures cannot silently activate a document.
13. Smoke checks prevent activation of incomplete or inconsistent ingestion data.
14. Manual approval cascades status to all related records and returns HTTP 409 when validation
    fails. Manual approval must not waive current business validation or smoke blockers.
15. Duplicate policies `reject`, `reuse`, `replace`, and `force` behave explicitly; replacement
    archives prior active versions only after the new version passes.
16. Product list query returns active products through SQL.
17. Service list query returns active services through SQL.
18. Product detail query can retrieve a product by name.
19. Product compare retrieves multiple products and compares only from source data.
20. FAQ query works through exact/fuzzy or semantic search.
21. Chat endpoint returns structured JSON.
22. LLM output is validated.
23. Every chat request creates `rag_traces` and `rag_trace_steps`.
24. Latency is recorded per stage.
25. Evaluation dataset can be run.
26. Evaluation results are stored in `evaluation_runs`.
27. Tests pass.
28. README and AGENTS.md explain the implemented ingestion lifecycle.
29. Structured SQL list/detail responses resolve assets from item UUIDs even when summary text
    contains no asset token.
30. Generic service schemas cannot be synchronized into `products` merely because they contain
    `name`, `category`, and `price`.
31. One user message can produce multiple tasks and each task keeps its own intent, entities,
    filters, resolved IDs and evidence.
32. Implicit follow-up binding cannot leave rejected PlannerLLM entities in the executable query,
    entity filters, global summary or memory state.
33. Explicit entity switching overrides previous conversation state and is resolved against the
    database before execution.
34. Synthesis cannot return an entity or structured fact that conflicts with its cited evidence.
35. Conversation evaluation measures entity binding, follow-up memory, multi-task decomposition
    and complete scenario pass rate.

---

### 18.1 Implemented Ingestion Baseline

As of June 12, 2026, the repository implementation has been verified with a real product PDF:

- Docling produced separated text, table, and image blocks.
- The product table created 12 `table_rows` and 12 normalized `products`.
- All 12 products linked to extracted assets.
- 12 assets produced 23 `chunk_assets` links.
- Chunk and table-row embeddings were present with dimension 1024.
- The ingestion smoke report passed with no unresolved tokens, missing asset files, orphan
  business records, unclassified rows, or duplicate active checksum.
- `duplicate_policy=replace` activated the new version and archived three legacy duplicate
  versions.
- PostgreSQL migration revision `0009_service_aliases` is the current head.
- Ingestion now skips duplicate business rows within one run. Activation follows
  latest-approved-wins semantics: older matching product/service/clinic records are archived,
  matching FAQs are disabled, and product/service versions increase transactionally.
- Migration `0006_business_record_dedup` cleaned the existing active service duplicate and
  added four normalized partial unique indexes. PostgreSQL verification reported zero active
  duplicate groups for products, services, FAQs, and clinic information.

This section records the verified baseline, not a permanent exemption from running tests and
smoke checks after future ingestion changes.

### 18.1.1 June 15, 2026 Ingestion Remediation

The repository audit and remediation are documented in
`WALKTHROUGH_REMEDIATION.md`.

- Generic service catalogs using `name`, `duration_minutes`, and `symptoms` are classified as
  services before shared product/service columns are considered.
- Generic product catalogs require product-specific evidence such as brand, model, quantity, or
  product link. `name + price/category` alone is not a high-confidence product schema.
- Quality reports separate review-only reasons from approval integrity blockers.
- Manual approval recomputes business validation from current table rows and merges it with smoke
  failures. Stale row-validation messages are not trusted, but unresolved current failures still
  return HTTP 409.
- The upload UI exposes duplicate policy and reports whether ingestion became active or remained
  review-required.
- The affected `services.csv` was re-ingested with `duplicate_policy=replace`. Version 2 is active,
  the incorrectly classified version 1 is archived, and active business data now contains 20
  products and 15 services.
- Runtime verification classified the service table at confidence `0.96`, returned one exact
  structured service-detail match, and resolved all 20 assets in a 20-item direct product-list
  response with no missing references.

### 18.2 Implemented Retrieval Baseline

As of June 19, 2026, retrieval has been verified against the local PostgreSQL dataset:

- Router uses Ollama JSON routing as the primary path with Pydantic validation, structured
  entity metadata, reason codes, JSON repair retry, safe fallback, and circuit breaker.
  High-confidence deterministic rules no longer skip RouterLLM.
- RouterLLM entity parsing accepts unsupported entity types and null names, normalizes them
  before entity resolution, and filters null names out of `RouterResult.entities`.
- RouterLLM prompts include query-relevant active product/service/category names instead of the
  full active catalog by default.
- RouterLLM requests are sent without a request timeout so local traces show the full router
  latency for each query.
- The evidence-first task planner performs an LLM review pass. Runtime verification corrected
  `dịch vụ nào ... dưới 2 triệu` to `SERVICE_LIST` and decomposed `giá ... và có đau không`
  into `SERVICE_DETAIL` plus `FAQ`.
- `GREETING` and `CHITCHAT` are planned as `NO_RAG_LLM` and use generation LLM without
  retrieval context. Static templates are only error fallbacks.
- PostgreSQL entity resolution uses active product/service names, unaccented trigram matching,
  ambiguity detection, source IDs, and traced candidate scores instead of a hardcoded entity list.
- Retrieval planning chooses template, direct SQL, structured-only, structured-then-hybrid,
  hybrid, or clarification based on intent, confidence, entity resolution, and SQL match.
- Product/service detail uses structured SQL as the authoritative entity-resolution step when
  one resolved entity ID matches one authoritative SQL result. Missing or ambiguous entities
  return clarification. Supplementary RAG evidence may be used for synthesis, but it must not
  invent or choose a product/service record.
- Product list uses a typed SQL query specification for taxonomy/product filters, price ranges,
  and whitelisted sort columns/directions.
- Service list uses a typed SQL query specification for service category aliases, service names,
  price ranges, duration ranges, and whitelisted sort columns/directions. Implant-related
  service queries are filtered to implant evidence instead of returning all services.
- HyDE remains optional and is limited to FAQ/unstructured retrieval. The original query is
  always retained and traced.
- FAQ tasks combine their inherited product/service entity with the sub-query before FAQ and
  hybrid retrieval. This prevents generic text such as `có đau không` from ranking an unrelated
  treatment FAQ above the entity-specific FAQ.
- FAQ evidence may come from curated FAQ rows plus active document chunks/table rows. RRF and
  the optional reranker operate inside `document_rag_tool`; structured catalog tasks do not
  invoke this tool.
- Dense and sparse retrieval search collections according to intent.
- Migration `0005_vietnamese_retrieval` enables `unaccent`, rebuilds generated `tsvector`
  columns with the `simple` configuration, and adds normalized trigram indexes.
- Vietnamese sparse normalization removes non-discriminative question phrases, expands a
  small dental synonym set, and filters table rows by entity type.
- Product/service/FAQ row lineage is converted to canonical keys. Weighted RRF merges the same
  entity across structured, dense, and sparse result lists while preserving the authoritative
  structured representation.
- Context building deduplicates canonical entities, preserves `raw_json` and asset tokens,
  enforces per-source limits, and skips oversized items without aborting the context.
- Startup validation loads the embedding model on the same cached `ChatService` instance used by
  requests. Optional reranker preload moves its model-load cost out of the first hybrid request.
- Ollama traces retain per-attempt load, prompt-evaluation, generation duration, and token counts.
  Asset resolution fields are server-managed so malformed LLM asset objects cannot trigger a
  second generation attempt.
- Evidence synthesis uses a compact JSON contract: `answer`, `used_source_ids` and safety flags.
  The server builds entities, items, sources and assets from context, requires source coverage
  for every planned task, accepts grounded Vietnamese price units such as `2,5 triệu`, and
  records failed generation as a degraded fallback instead of a successful normal response.
- The evidence-first planner records RouterLLM/planner fallback metadata and can reuse recent
  session entities for follow-up questions such as "mất bao lâu?" when a stable `session_id`
  is provided.
- Response asset resolution consumes both concrete text tokens and structured item asset UUIDs,
  so direct SQL list/detail responses do not depend on chunk text.
- Dense and sparse retrieval join owning documents and exclude archived documents in addition
  to filtering active chunks/table rows and active FAQs.
- Runtime verification returned the correct whitening service at score `0.97`, price
  `2.500.000 VND`, duration `90` minutes; the multi-task response cited both the service row and
  the `Răng ê buốt sau tẩy trắng...` FAQ; the implant-under-2-million list returned no match
  instead of selecting the 18-million implant detail.
- Real verification ranked the expected AquaJet product, implant service, and sensitive-teeth
  FAQ at top-1. A FAQ request recorded successful dense, sparse, RRF, context, asset, and
  rendering trace steps.
- `scripts/verify_retrieval_pipeline.py` reproduces structured and hybrid trace checks.
- `eval_datasets/dental_retrieval_eval.jsonl` contains grounded paraphrase/entity-shortening
  cases that exercise hybrid retrieval instead of only direct SQL paths.
- `eval_datasets/README.md` defines the 40-case expansion matrix, authoring rules, dataset
  splits, `expected_retrieval_mode`, and acceptance thresholds.
- The deterministic 30-case baseline completed with 100% E2E pass rate. The grounded
  six-case hybrid baseline completed with Router accuracy, Hit@1, Recall@5, MRR@10, nDCG@10,
  faithfulness, and E2E pass rate all equal to `1.0`; wrong/missing asset rates were `0.0`.
- Hybrid verification used a one-second generation timeout to exercise grounded fallback, so
  its 100% fallback rate is not a production generation benchmark.

### 18.2.1 June 20, 2026 Context-Binding Audit

The current pipeline has separate GLiNER extraction, task-level context binding and database
entity resolution. Runtime traces confirmed that an implicit follow-up can recover the correct
whitening service ID and retrieve authoritative evidence with `duration_minutes=90`.

The same audit identified a remaining contract defect: Context Binder can replace
`task.entities`, `selection.mentions` and `selection.resolved_ids` while stale PlannerLLM text
remains in `task.query`, typed entity filters and `global_entities`. Because the synthesis prompt
includes those task fields, the generation model may answer about the stale entity even when
cited items and sources point to the correct database row. Price-only validation does not catch
that semantic mismatch.

The audit required:

- canonicalize every entity-bearing task field after binding;
- rebuild `global_entities` from post-binding tasks;
- add a pre-synthesis task/evidence consistency gate;
- validate entity names and structured facts such as duration, stock and availability;
- persist only canonical post-binding filters into conversation memory.

### 18.2.2 June 22, 2026 Context-Binding Remediation

The stale-field defect was remediated by replacing mutable planner/execution state with explicit
trust boundaries:

- `PlannedTask` is a frozen, untrusted proposal with separate `planner_query`,
  `planner_entities` and planner filters. It has no `resolved_ids` or `effective_query`.
- Context Binder returns immutable `BindingDecision` records and does not mutate the plan.
- Database-backed resolution runs before `TaskCanonicalizer`, which atomically creates frozen
  `BoundTask` records with canonical names, IDs, typed filters, reference provenance and a
  rewritten `effective_query`.
- `IntentCapabilityRegistry` owns entity cardinality, inheritance, allowed tools, clarification
  and evidence contracts for every intent. Startup validation requires complete registry
  coverage.
- A pre-tool gate validates BoundTask internal/cross-task consistency. Same-turn FAQ inheritance
  is allowed only from a resolved source task that passes this gate.
- A post-retrieval evidence gate validates source types, authority and resolved-ID coverage before
  synthesis. Failed tasks are clarified or blocked and are never included in the LLM payload.
- Synthesis receives only effective task queries, canonical entity names and validated evidence.
  Planner queries, planner-global entities, rejected proposals and resolution candidates remain
  trace-only diagnostics.
- Semantic response validation now checks canonical entity identity, duration, stock/availability
  and explicit opening-hour values in addition to price grounding.
- Conversation memory persists only resolved, registry-eligible BoundTasks that passed the
  evidence gate. List results, planner proposals and failed tasks do not become active entities.

Backend Ruff and the complete backend test suite pass after the remediation. Frontend lint,
Vitest and production build also pass with the updated Trace Explorer contract.

Migration `0007_catalog_taxonomy` adds document auto-detection, normalized product/service/FAQ
taxonomies, entity/FAQ aliases, FAQ lineage, companion `image_reference` linking, and
confirmation-gated document/content deletion. Migration `0008_conversation_memory` adds
conversation sessions, turns and summaries for the evidence-first pipeline. The Web UI exposes
companion image upload, detected document type, permanent delete/reset controls, and separate
Chat/FAQ tabs.
Migration `0009_service_aliases` adds service aliases used by explicit span matching, task
binding and database-backed service resolution.

---

## 19. Important Safety and Correctness Rules

- Do not hallucinate product prices, service prices, opening hours, or clinic policies.
- If data is missing, say that current clinic data does not contain enough information.
- For dental/medical questions, do not provide definitive diagnosis.
- Keep medical answers as general guidance and recommend seeing a dentist for severe symptoms.
- Do not expose internal traces to normal users unless debug mode is explicitly enabled.
- Do not expose database credentials.
- Do not return archived/stale records.
- Do not use LLM-generated summaries as source of truth.
- Source of truth must be PostgreSQL records derived from ingested or manually approved data.

---

## 20. Final Instruction to Codex

After completing or changing the implementation:

1. Run formatting and tests.
2. Verify imports and startup.
3. Check that migrations are present.
4. Check that `.env.example` is complete.
5. Update README with exact setup commands.
6. Run ingestion smoke checks against a disposable or explicitly approved source document.
7. Verify that any test document and staged asset directory are cleaned afterward.

Before the first full run in a new environment, ask for any configuration values that are not
already present in `.env` or otherwise confirmed:

- PostgreSQL host
- PostgreSQL port
- PostgreSQL database name
- PostgreSQL username
- PostgreSQL password
- whether pgvector is already enabled
- Ollama base URL
- available Ollama model names
- asset storage directory preference
- whether Langfuse should be enabled

Do not overwrite confirmed local values. Do not ask for credentials again when the existing
environment has already been validated and the requested work can be completed safely with it.
