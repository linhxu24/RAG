# SimplyDent RAG Remediation Walkthrough

## 1. Scope

This walkthrough records the June 15, 2026 investigation of ingestion, asset
resolution, retrieval, generation, and runtime configuration. It separates:

- defects reproduced in the current PostgreSQL data;
- risks already mitigated by the implementation;
- fixes applied to code and tests;
- the safe procedure for repairing affected catalog data.

## 2. Reproduced Defects

### 2.1 Service catalog classified as products

The active `services.csv` document was detected as `product_catalog` with
confidence `0.96`. Its 15 rows were synchronized into `products`, leaving:

```text
products active: 35
services active: 0
```

Root cause:

- both catalogs use a generic `name` column;
- `price` and `category` were treated as product-specific signals;
- service-specific columns such as `duration_minutes` and `symptoms` were not
  considered before the product rule.

Expected fix:

- treat duration, symptoms, indications, and contraindications as strong
  service signals;
- treat brand, model, quantity, and product links as strong product signals;
- never classify `name + price/category` alone as a high-confidence product.

### 2.2 Manual approval bypassed ingestion review blockers

The affected service document retained category and image-reference review
reasons but was manually activated because approval only reran smoke checks.

Expected fix:

- manual approval may acknowledge review-only reasons such as an explicit
  `require_review` option or a low-confidence but classified table;
- manual approval must not waive integrity blockers such as invalid business
  rows, missing embeddings, unresolved concrete asset tokens, or failed smoke
  checks;
- business-row validation must be recomputed from current persisted rows so
  stale quality-report messages do not control approval.

### 2.3 SQL list responses omitted resolved asset objects

`PRODUCT_DETAIL` correctly resolved the asset token generated from
`products.asset_id`. `PRODUCT_LIST` returned item-level `asset_ids`, but its
summary text contained no tokens, so `result.assets` was empty.

Expected fix:

- resolve assets from both response text tokens and
  `result.items[].asset_ids`;
- deduplicate the final asset list by UUID;
- report missing token or UUID references.

## 3. Confirmed Existing Protections

The following suspected defects were not present in the current implementation:

- dense and sparse retrieval filter active chunks/table rows and active FAQs;
- FAQ filtering correctly uses `is_active`;
- pgvector indexes use HNSW, not IVFFlat;
- Docling markdown export excludes table labels before text chunking;
- direct SQL and LLM output share `GeneratedResponse` and validation;
- runtime generation timeout is 120 seconds; the one-second timeout is scoped
  to a verification script;
- authored and hash asset tokens use one detector, and `XX`-style template
  segments are ignored.

## 4. Configuration Guidance

- Keep strict embeddings enabled in production. A partial document with
  missing vectors must not become active.
- Keep table threshold changes separate from schema-signal changes. Lowering
  the threshold would not have fixed the reproduced service misclassification.
- Do not increase `ENTITY_AMBIGUITY_MARGIN` to reduce clarifications. A larger
  margin marks more close candidate pairs as ambiguous.
- Tune entity thresholds and reranking only from recorded candidate scores and
  evaluation datasets.

## 5. Implementation Sequence

1. Add product/service schema-specific classification rules.
2. Add approval policy that merges smoke failures with recomputed integrity
   blockers.
3. Resolve response assets by token and UUID.
4. Expose duplicate policy and review reasons in the upload UI.
5. Record all entity candidates in trace output.
6. Add regression tests for each reproduced defect and active-only retrieval.
7. Run backend tests, frontend tests, lint, and startup/import checks.
8. Re-ingest the affected service catalog with `duplicate_policy=replace`.
9. Approve only after the new report has no integrity blockers.
10. Verify active counts and run `SERVICE_LIST`/`SERVICE_DETAIL`.

## 6. Data Repair Procedure

Use the original uploaded service CSV. Because its checksum already exists,
replacement must be explicit:

```bash
curl -X POST http://localhost:8000/ingest/run \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "uploads/93daa48012ad4705a647c3e7b9cbd2fb.csv",
    "document_type": "auto",
    "duplicate_policy": "replace"
  }'
```

If the result is `review_required`, inspect:

```text
quality_report.approval_blocking_reasons
quality_report.review_only_reasons
quality_report.smoke_test.blocking_reasons
```

Only call approval when `approval_blocking_reasons` is empty. Successful
replacement archives the old document and its incorrectly synchronized product
records before activating the corrected service records.

## 7. Verification Checklist

- generic service schema is classified as `service`;
- generic product schema remains classified as `product`;
- ambiguous generic business schema requires review;
- approval rejects current business validation failures;
- approval permits explicit human-review flags when integrity checks pass;
- direct product detail resolves its image;
- product/service list assets resolve from item UUIDs;
- archived chunks, rows, and FAQs are excluded from dense/sparse retrieval;
- all backend and frontend tests pass;
- active PostgreSQL data contains the expected product/service counts.

## 8. Executed Repair Result

The service catalog was re-ingested on June 15, 2026 with
`duplicate_policy=replace` after the code fixes were applied.

- the corrected `services.csv` document is active as version 2;
- the incorrectly classified version 1 document is archived;
- active business data contains 20 products and 15 services;
- the new table classification is `service` with confidence `0.96`;
- unresolved optional companion image references remain review-only;
- `approval_blocking_reasons` is empty;
- exact structured service detail returned one service record;
- a direct product-list response returned 20 items and resolved 20 assets with
  no missing asset references.
