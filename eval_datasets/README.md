# Hybrid Retrieval Dataset Guide

`dental_retrieval_eval.jsonl` is the grounded regression dataset for retrieval paths that must
exercise dense, sparse, structured fallback, and RRF. Keep it separate from
`dental_basic_eval.jsonl`, which intentionally contains many direct-SQL cases.

## Target Size

Build a 40-case dataset:

| Group | Cases | Suggested coverage |
|---|---:|---|
| FAQ paraphrases | 12 | sensitivity, post-treatment, emergency |
| Product detail | 10 | five products, two variants each |
| Service detail | 10 | five services, two variants each |
| Product compare | 4 | aliases, reversed order, noisy wording |
| Ambiguous/negative dental | 4 | clarification and no-result behavior |

Do not add 40 variants of one FAQ. Coverage by entity and intent matters more than raw count.

## Paraphrase Variants

For each grounded source, create variants from different categories:

1. Colloquial Vietnamese: `răng ê khi uống lạnh thì sao`.
2. No accents: `rang e buot uong nuoc lanh`.
3. Short entity or brand: `AquaJet Mini`, `EnamelGuard`.
4. Word-order change: `chi phí thế nào cho dịch vụ trồng implant`.
5. Added conversational noise: `bác sĩ ơi cho tôi hỏi...`.
6. Typo or spacing variation: `Oral B`, `implant`, `nho rang khon`.
7. Indirect symptom wording: `mặt sưng kèm đau răng`.
8. Post-treatment wording without the exact FAQ title.

Use at most one typo-focused variant per source. Excessive synthetic typos produce an
unrealistic benchmark.

## Required JSONL Fields

```json
{
  "case_key": "retrieval_faq_sensitive_no_accent",
  "query": "rang e buot khi uong nuoc lanh thi sao",
  "expected_intent": "FAQ",
  "expected_source_keys": ["faq:Tại sao răng bị ê buốt?"],
  "expected_answer_contains": ["gặp nha sĩ"],
  "forbidden_answer_contains": [],
  "metadata": {
    "group": "hybrid_faq",
    "variant": "no_accent",
    "medical": true,
    "expected_retrieval_mode": "STRUCTURED_THEN_HYBRID"
  }
}
```

Prefer stable source keys:

- `product:<exact database name>`
- `service:<exact database name>`
- `faq:<exact stored question>`
- `clinic_info:<key>`
- `chunk:<document_checksum>:<chunk_index>` only when no structured identity exists

Never copy UUIDs into the JSONL unless no stable source key can represent the target.

## Authoring Workflow

1. Export active entity names and FAQ questions from PostgreSQL.
2. Select entities across different categories and document sources.
3. Draft 2-4 candidate paraphrases for each selected source.
4. Manually reject candidates that change the meaning or require facts absent from the source.
5. Add exactly one expected intent and one primary ground-truth entity/source per detail case.
6. Set `expected_retrieval_mode` so a case cannot pass by silently taking direct SQL.
7. Run deterministic evaluation.
8. Inspect every failed case trace before changing thresholds or synonyms.
9. Keep difficult but valid cases; remove only ambiguous or incorrectly grounded cases.

Local LLMs may generate candidate wording, but a human must validate intent, source key, safety
metadata, and expected answer constraints.

## Dataset Split

For 40 cases:

- 24 development cases: visible during retrieval tuning.
- 8 regression cases: run on every change.
- 8 blind cases: do not use while changing synonyms, thresholds, or prompts.

Store the split in `metadata.split` with `development`, `regression`, or `blind`. Do not place
near-identical paraphrases of the same source in development and blind splits.

## Acceptance Thresholds

Recommended initial gates:

- ground-truth coverage: `1.0`
- retrieval-mode match: `1.0`
- router accuracy: at least `0.95`
- Hit@1: at least `0.85`
- Recall@5: at least `0.95`
- MRR@10: at least `0.90`
- wrong/missing asset rate: `0.0`
- faithfulness and safety pass: `1.0`

Report metrics per group and per intent. Aggregate accuracy can hide a completely broken FAQ or
service-detail path.

## Run

```bash
curl -X POST http://localhost:8000/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "all",
    "profile": "deterministic",
    "dataset_path": "eval_datasets/dental_retrieval_eval.jsonl",
    "dataset_name": "dental_retrieval_eval",
    "dataset_version": "1.0",
    "data_version": "fixtures-v1"
  }'
```

Use the production profile only after the deterministic retrieval baseline passes. Otherwise
RouterLLM, HyDE, reranker, and generation latency make root-cause analysis unnecessarily mixed.
