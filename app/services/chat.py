from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings
from app.constants import Intent, RetrievalMode
from app.generation.generator import GenerationValidationError, GroundedGenerator
from app.generation.llm_client import build_llm_client
from app.generation.renderer import ResponseRenderer
from app.generation.schemas import ChatRequest, ChatResponse, ChatSuggestion
from app.ingestion.embedder import EmbeddingService
from app.memory.conversation_memory import ConversationMemory
from app.ner.entity_span_extractor import EntitySpanExtractor
from app.observability.langfuse_client import OptionalLangfuse
from app.observability.tracing import TraceRecorder
from app.orchestration.binding_pipeline import TaskBindingPipeline
from app.orchestration.consistency_gate import ConsistencyGate
from app.orchestration.context_binder import ContextBinder
from app.orchestration.evidence_merger import EvidenceMerger
from app.orchestration.intent_registry import capability_for, entity_label
from app.orchestration.query_features import QueryFeatures
from app.orchestration.query_rewriter import (
    QueryRewriteInput,
    RawTurn,
    RecentEntity,
    rewrite_query,
)
from app.orchestration.suggestions import ContextualSuggestionEngine
from app.orchestration.task_canonicalizer import TaskCanonicalizer
from app.orchestration.task_planner import TaskPlanner
from app.orchestration.task_resolver import TaskEntityResolver
from app.orchestration.tool_executor import ToolExecutor
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.entity_resolver import DatabaseEntityResolver
from app.retrieval.planner import RetrievalPlan, RetrievalPlanner
from app.retrieval.query_rewrite import QueryRewriter as HyDEQueryRewriter
from app.retrieval.reranker import OptionalReranker
from app.retrieval.router import IntentRouter
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.structured_query import (
    active_product_category_terms,
    parse_product_query,
)
from app.retrieval.structured_retriever import StructuredRetriever
from app.retrieval.types import RetrievalResult


class ChatService:
    EVIDENCE_FIRST_RUNTIME = "evidence_first"
    LEGACY_RUNTIME = "legacy_single_intent_debug"

    DIRECT_INTENTS = {
        Intent.GREETING,
        Intent.CHITCHAT,
        Intent.CLINIC_INFO,
        Intent.PRODUCT_LIST,
        Intent.SERVICE_LIST,
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = build_llm_client(settings)
        self.router = IntentRouter()
        self.entity_resolver = DatabaseEntityResolver(settings)
        self.planner = RetrievalPlanner(settings)
        self.rewriter = HyDEQueryRewriter()
        self.structured = StructuredRetriever(settings)
        embedder = EmbeddingService(settings)
        self.dense = DenseRetriever(
            embedder,
            settings.dense_top_k,
            settings.dense_min_score,
        )
        self.sparse = SparseRetriever(
            settings.sparse_top_k,
            settings.sparse_trigram_threshold,
            min_fts_rank=settings.sparse_min_fts_rank,
            max_per_source=settings.sparse_max_per_source,
        )
        self.reranker = OptionalReranker(settings)
        self.context_builder = ContextBuilder(
            settings.max_context_chars,
            settings.max_context_items_per_source,
        )
        self.generator = GroundedGenerator(settings, self.llm)
        self.renderer = ResponseRenderer()
        self.langfuse = OptionalLangfuse(settings)
        self.memory = ConversationMemory(settings.conversation_history_turns)
        self.task_planner = TaskPlanner()
        self.entity_span_extractor = EntitySpanExtractor(settings)
        self.context_binder = ContextBinder(settings)
        self.consistency_gate = ConsistencyGate()
        self.task_binding_pipeline = TaskBindingPipeline(
            binder=self.context_binder,
            resolver=TaskEntityResolver(self.entity_resolver),
            canonicalizer=TaskCanonicalizer(),
            consistency_gate=self.consistency_gate,
        )
        self.tool_executor = ToolExecutor(
            structured=self.structured,
            dense=self.dense,
            sparse=self.sparse,
            reranker=self.reranker,
            settings=settings,
        )
        self.evidence_merger = EvidenceMerger(settings)
        self.suggestion_engine = ContextualSuggestionEngine(
            max_suggestions=settings.max_contextual_suggestions,
            history_limit=settings.suggestion_history_limit,
        )

    async def chat(self, session: Session, request: ChatRequest) -> ChatResponse:
        trace = TraceRecorder.start(session, request.message, request.session_id)
        debug_data: dict[str, Any] = {}
        try:
            runtime_path = (
                self.EVIDENCE_FIRST_RUNTIME
                if self.settings.enable_multi_task_planner
                else self.LEGACY_RUNTIME
            )
            debug_data["runtime_path"] = runtime_path
            trace.record(
                "runtime_path",
                output_data={
                    "active": runtime_path,
                    "enable_multi_task_planner": self.settings.enable_multi_task_planner,
                },
            )
            if self.settings.enable_multi_task_planner:
                return await self._chat_with_evidence_pipeline(
                    session,
                    request,
                    trace,
                    debug_data,
                )

            trace.record(
                "legacy_runtime_warning",
                output_data={
                    "active": self.LEGACY_RUNTIME,
                    "reason": (
                        "ENABLE_MULTI_TASK_PLANNER=false; using the legacy "
                        "single-intent debug runtime"
                    ),
                },
            )
            product_names, service_names = self.structured.active_names(session)
            product_category_terms = active_product_category_terms(session)
            with trace.step("router_intent", {"query": request.message}) as step:
                routed = await self.router.route_with_optional_llm(
                    request.message,
                    self.settings,
                    self.llm,
                    known_products=product_names,
                    known_services=service_names,
                    known_product_categories=product_category_terms,
                )
                step["output"] = routed.as_dict()

            with trace.step("entity_extraction", {"query": request.message}) as step:
                resolution = self.entity_resolver.resolve(
                    session,
                    request.message,
                    routed.intent,
                )
                entities = self._effective_entities(routed, resolution)
                step["output"] = {
                    "entities": entities,
                    **resolution.as_dict(),
                }

            if routed.intent in {Intent.GREETING, Intent.CHITCHAT}:
                structured = []
                trace.skip("structured_retrieval", "Template intent")
            else:
                product_query_spec = (
                    parse_product_query(session, request.message)
                    if routed.intent == Intent.PRODUCT_LIST
                    else None
                )
                if product_query_spec and product_query_spec.needs_clarification:
                    routed.needs_clarification = True
                    routed.clarification_message = (
                        product_query_spec.clarification_message
                    )
                with trace.step(
                    "structured_retrieval",
                    {
                        "intent": routed.intent.value,
                        "entities": entities,
                        "query_spec": (
                            product_query_spec.as_dict()
                            if product_query_spec
                            else None
                        ),
                    },
                ) as step:
                    structured = self.structured.retrieve(
                        session,
                        routed.intent,
                        request.message,
                        entities,
                    )
                    step["output"] = self._result_summary(structured)

            with trace.step(
                "retrieval_planning",
                {
                    "intent": routed.intent.value,
                    "router_confidence": routed.confidence,
                    "entity_status": resolution.status,
                    "structured_count": len(structured),
                },
            ) as step:
                plan = self.planner.plan(
                    query=request.message,
                    routed=routed,
                    entities=resolution,
                    structured=structured,
                )
                step["output"] = plan.as_dict()

            dense_sets: dict[str, list[RetrievalResult]] = {}
            sparse_sets: dict[str, list[RetrievalResult]] = {}
            if plan.mode.value in {"CLARIFY", "TEMPLATE"}:
                trace.skip("query_rewrite_hyde", plan.reason)
                self._skip_hybrid(trace, plan.reason)
                final_results = structured
            elif plan.uses_hybrid:
                with trace.step(
                    "query_rewrite_hyde",
                    {"query": request.message, "intent": routed.intent.value},
                ) as step:
                    rewrite = await self.rewriter.rewrite(
                        request.message,
                        routed.intent,
                        self.settings,
                        self.llm,
                    )
                    step["output"] = rewrite.as_dict()

                with trace.step(
                    "dense_retrieval",
                    {
                        "original_query": rewrite.original_query,
                        "hyde_query": rewrite.hyde_query,
                    },
                ) as step:
                    dense_sets = self._dense_result_sets(
                        session,
                        rewrite,
                        routed.intent,
                    )
                    step["output"] = self._result_set_summary(dense_sets)

                with trace.step(
                    "sparse_retrieval",
                    {"query": rewrite.normalized_query},
                ) as step:
                    sparse_sets = {
                        f"sparse_{name}": values
                        for name, values in self.sparse.retrieve_by_source(
                            session,
                            request.message,
                            routed.intent,
                        ).items()
                    }
                    step["output"] = self._result_set_summary(sparse_sets)

                result_sets = {
                    "structured": structured,
                    **dense_sets,
                    **sparse_sets,
                }
                with trace.step(
                    "rrf_fusion",
                    {
                        name: [item.key for item in values]
                        for name, values in result_sets.items()
                    },
                ) as step:
                    fused = reciprocal_rank_fusion(
                        result_sets,
                        self.settings.rrf_k,
                        self._rrf_weights(result_sets),
                        max_per_source=self.settings.rrf_max_per_source,
                    )
                    step["output"] = self._result_summary(fused)

                with trace.step(
                    "reranker",
                    {"count": len(fused), "planned": plan.use_reranker},
                ) as step:
                    if plan.use_reranker:
                        final_results, reranked, rerank_meta = self.reranker.rerank(
                            request.message,
                            fused,
                        )
                    else:
                        final_results = fused[: self.settings.final_top_k]
                        reranked = False
                        rerank_meta = {"reranked": False, "reason": "not_planned"}
                    step["output"] = {
                        "enabled_and_used": reranked,
                        **rerank_meta,
                        **self._result_summary(final_results),
                    }
            else:
                trace.skip("query_rewrite_hyde", plan.reason)
                self._skip_hybrid(trace, plan.reason)
                final_results = structured

            dense = [item for values in dense_sets.values() for item in values]
            sparse = [item for values in sparse_sets.values() for item in values]

            with trace.step("context_builder", {"count": len(final_results)}) as step:
                context = self.context_builder.build(
                    final_results,
                    apply_limits=routed.intent
                    not in {
                        Intent.CLINIC_INFO,
                        Intent.PRODUCT_LIST,
                        Intent.SERVICE_LIST,
                    },
                )
                step["output"] = {
                    "item_count": len(context["items"]),
                    "total_chars": context["total_chars"],
                    "source_ids": [item["source_id"] for item in context["items"]],
                    "source_types": [item["source_type"] for item in context["items"]],
                    "source_counts": context.get("source_counts", {}),
                }
            debug_data["retrieval"] = {
                "structured": self._result_summary(structured),
                "dense": self._result_summary(dense),
                "sparse": self._result_summary(sparse),
                "context_items": len(context["items"]),
                "plan": plan.as_dict(),
            }

            should_generate = self._should_use_llm(
                routed.intent,
                plan,
                structured,
            )
            if should_generate and (
                context["items"] or plan.mode == RetrievalMode.NO_RAG_LLM
            ):
                try:
                    if plan.mode == RetrievalMode.NO_RAG_LLM:
                        with trace.step(
                            "prompt_builder",
                            {
                                "intent": routed.intent.value,
                                "context_items": len(context["items"]),
                            },
                        ) as step:
                            from app.generation.prompts import build_chitchat_prompt

                            prompt = build_chitchat_prompt(
                                query=request.message,
                                intent=routed.intent,
                                confidence=routed.confidence,
                            )
                            step["output"] = {"prompt_chars": len(prompt)}
                        generation_call = self.generator.generate_chitchat_with_retry(
                            query=request.message,
                            confidence=routed.confidence,
                            session=session,
                        )
                    else:
                        with trace.step(
                            "prompt_builder",
                            {
                                "intent": routed.intent.value,
                                "context_items": len(context["items"]),
                            },
                        ) as step:
                            from app.generation.prompts import build_generation_prompt

                            prompt = build_generation_prompt(
                                query=request.message,
                                intent=routed.intent,
                                confidence=routed.confidence,
                                entities=entities,
                                context=context,
                            )
                            step["output"] = {"prompt_chars": len(prompt)}
                        generation_call = self.generator.generate_with_retry(
                            query=request.message,
                            intent=routed.intent,
                            confidence=routed.confidence,
                            entities=entities,
                            context=context,
                            session=session,
                        )
                    with trace.step(
                        "llm_generation",
                        {"model": self.settings.llm_generation_model},
                    ) as step:
                        try:
                            response, generation_meta = await generation_call
                        except GenerationValidationError as exc:
                            step["output"] = exc.metadata
                            raise
                        step["output"] = generation_meta["llm"]
                    with trace.step("json_validation", {}) as step:
                        step["output"] = generation_meta["validation"]
                except Exception as exc:
                    response = self.generator.fallback_from_context(
                        intent=routed.intent,
                        confidence=routed.confidence,
                        context=context,
                    )
                    if "llm_generation" not in trace.recorded_steps:
                        trace.record(
                            "llm_generation",
                            status="failed",
                            error_message=str(exc),
                        )
                    trace.record(
                        "json_validation",
                        output_data={"fallback": True, "reason": str(exc)},
                        status="failed",
                        error_message=str(exc),
                    )
                    trace.record(
                        "generation_fallback",
                        output_data={"answer_type": response.answer_type},
                        status="success",
                    )
            else:
                trace.skip("prompt_builder", "Direct SQL or template answer")
                trace.skip("llm_generation", "Direct SQL or template answer")
                response = self.generator.direct_response(
                    intent=routed.intent,
                    confidence=routed.confidence,
                    context=context,
                    clarification_reason=plan.reason
                    if plan.mode.value == "CLARIFY"
                    else None,
                    clarification_message=routed.clarification_message,
                )
                with trace.step("json_validation", {"answer_type": response.answer_type}) as step:
                    response = self.generator.validator.validate(
                        response.model_dump(mode="json"), context=context, session=session
                    )
                    step["output"] = {"valid": True}

            return self._finish(
                session, trace, response, request.debug, debug_data, routed.confidence
            )
        except Exception as exc:
            if self._is_recoverable_chat_error(exc):
                response = self.generator.direct_response(
                    intent=Intent.UNKNOWN,
                    confidence=0.0,
                    context={"items": [], "total_chars": 0},
                )
                response.degraded = True
                trace.record(
                    "pipeline_fallback",
                    output_data={
                        "runtime_path": debug_data.get("runtime_path"),
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    },
                    status="degraded",
                    error_message=str(exc),
                )
                return self._finish(
                    session,
                    trace,
                    response,
                    request.debug,
                    debug_data,
                    0.0,
                )
            fallback = self.generator.direct_response(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                context={"items": [], "total_chars": 0},
            )
            rendered = self.renderer.render(str(trace.trace_id), fallback)
            trace.finish(
                intent=Intent.UNKNOWN.value,
                confidence=0.0,
                answer=rendered.model_dump(mode="json"),
                status="failed",
            )
            raise RuntimeError(f"Chat pipeline failed; trace_id={trace.trace_id}: {exc}") from exc

    @staticmethod
    def _is_recoverable_chat_error(exc: Exception) -> bool:
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, SQLAlchemyError):
                return False
            current = current.__cause__ or current.__context__
        return isinstance(exc, (RuntimeError, TimeoutError, ValueError))

    async def _chat_with_evidence_pipeline(
        self,
        session: Session,
        request: ChatRequest,
        trace: TraceRecorder,
        debug_data: dict[str, Any],
    ) -> ChatResponse:
        with trace.step("memory_load", {"session_id": request.session_id}) as step:
            history = self.memory.load(session, request.session_id)
            raw_transcript = self.memory.load_raw_transcript(
                session,
                request.session_id,
                history_turns=self.settings.query_rewrite_history_turns,
            )
            recent_entities = self.memory.recent_entities(history)
            history["raw_transcript"] = raw_transcript
            history["recent_entities"] = recent_entities
            step["output"] = {
                "session_id": history.get("session_id"),
                "turn_count": len(history.get("turns", [])),
                "raw_transcript_turn_count": len(raw_transcript),
                "recent_entities": recent_entities,
                "has_summary": bool(history.get("summary")),
                "state": history.get("state"),
            }

        product_names, service_names = self.structured.active_names(session)
        product_category_terms = active_product_category_terms(session)
        pipeline_query = request.message
        rewrite_debug: dict[str, Any] = {
            "original_query": request.message,
            "pipeline_query": request.message,
            "enabled": self.settings.enable_query_rewrite,
            "skipped": False,
            "needs_clarification": False,
            "referenced_entities": [],
        }
        pre_rewrite_route = self.router.route(
            request.message,
            known_products=product_names,
            known_services=service_names,
            known_product_categories=product_category_terms,
        )
        if (
            self.settings.enable_query_rewrite
            and not self._is_pure_social_query(request.message, pre_rewrite_route)
        ):
            rewrite_payload = QueryRewriteInput(
                current_query=request.message,
                turns=[
                    RawTurn(role=turn["role"], text=turn["text"])
                    for turn in raw_transcript
                    if turn.get("role") and turn.get("text")
                ],
                recent_entities=[
                    RecentEntity(name=entity["name"], type=entity["type"])
                    for entity in recent_entities
                    if entity.get("name") and entity.get("type")
                ],
            )
            with trace.step(
                "contextual_query_rewrite",
                {
                    "original_query": request.message,
                    "history_turns": len(raw_transcript),
                    "recent_entities": recent_entities,
                    "model": self.settings.llm_query_rewrite_model,
                },
            ) as step:
                rewritten = await rewrite_query(
                    self.llm,
                    rewrite_payload,
                    model=self.settings.llm_query_rewrite_model,
                    timeout_s=self.settings.query_rewrite_timeout_s,
                )
                pipeline_query = rewritten.rewritten_query.strip() or request.message
                rewrite_debug.update(
                    {
                        "pipeline_query": pipeline_query,
                        "is_standalone": rewritten.is_standalone,
                        "needs_clarification": rewritten.needs_clarification,
                        "referenced_entities": rewritten.referenced_entities,
                    }
                )
                step["output"] = {
                    "original_query": request.message,
                    "rewritten_query": pipeline_query,
                    "is_standalone": rewritten.is_standalone,
                    "needs_clarification": rewritten.needs_clarification,
                    "referenced_entities": rewritten.referenced_entities,
                }
            if rewritten.needs_clarification:
                debug_data["query_rewrite"] = {
                    **rewrite_debug,
                    "needs_clarification": True,
                }
                response = self.generator.direct_response(
                    intent=Intent.UNKNOWN,
                    confidence=0.25,
                    context={"items": [], "total_chars": 0},
                    clarification_reason="query_rewrite_needs_clarification",
                    clarification_message=(
                        "Bạn vui lòng nói rõ hơn bạn đang nhắc tới sản phẩm, "
                        "dịch vụ hoặc thông tin nào."
                    ),
                )
                memory_save_payload = {
                    "session_id": request.session_id,
                    "user_content": request.message,
                    "assistant_content": response.result.text,
                    "detected_intents": [Intent.UNKNOWN.value],
                    "entities": {"tasks": {}},
                    "resolved_ids": {
                        "evidence_ids": [],
                        "active_product_ids": [],
                        "active_service_ids": [],
                    },
                    "state": _normalize_conversation_state(history.get("state")),
                    "trace_id": trace.trace_id,
                    "suggestion_count": 0,
                }
                return self._finish(
                    session,
                    trace,
                    response,
                    request.debug,
                    debug_data,
                    0.25,
                    memory_save_payload=memory_save_payload,
                )
        else:
            reason = (
                "Contextual query rewriting disabled"
                if not self.settings.enable_query_rewrite
                else "Pure social intent does not need query rewriting"
            )
            rewrite_debug.update({"skipped": True, "skip_reason": reason})
            trace.skip("contextual_query_rewrite", reason)

        debug_data["query_rewrite"] = rewrite_debug
        with trace.step(
            "task_planning",
            {
                "query": pipeline_query,
                "original_query": request.message,
                "history_turns": len(history.get("turns", [])),
            },
        ) as step:
            plan = await self.task_planner.plan(
                query=pipeline_query,
                history=history,
                settings=self.settings,
                llm=self.llm,
                known_products=product_names,
                known_services=service_names,
                known_product_categories=product_category_terms,
            )
            step["output"] = plan.as_dict()

        with trace.step(
            "entity_span_extraction",
            {"query": pipeline_query, "original_query": request.message},
        ) as step:
            span_result = self.entity_span_extractor.extract(
                pipeline_query,
                known_products=product_names,
                known_services=service_names,
            )
            step["output"] = span_result.as_dict()

        with trace.step(
            "context_binding",
            {
                "query": pipeline_query,
                "original_query": request.message,
                "state": history.get("state"),
                "spans": span_result.as_dict(),
            },
        ) as step:
            binding_pipeline_result = self.task_binding_pipeline.run(
                session,
                plan=plan,
                original_query=pipeline_query,
                history=history,
                span_result=span_result,
            )
            step["output"] = {
                "decisions": [
                    decision.model_dump(mode="json")
                    for decision in binding_pipeline_result.decisions
                ],
                "planner_plan": plan.as_dict(),
            }

        with trace.step(
            "entity_resolution",
            {
                "query": pipeline_query,
                "original_query": request.message,
                "tasks": plan.as_dict()["tasks"],
            },
        ) as step:
            step["output"] = {
                "resolutions": [
                    resolution.model_dump(mode="json")
                    for resolution in binding_pipeline_result.resolutions
                ]
            }

        bound_plan = binding_pipeline_result.bound_plan
        with trace.step(
            "task_canonicalization",
            {"planner_tasks": plan.as_dict()["tasks"]},
        ) as step:
            step["output"] = bound_plan.as_dict()

        with trace.step(
            "bound_task_consistency",
            {"tasks": bound_plan.as_dict()["tasks"]},
        ) as step:
            bound_gate = self.consistency_gate.check_bound_plan(bound_plan)
            step["output"] = bound_gate.model_dump(mode="json")

        with trace.step(
            "tool_execution",
            {"tasks": bound_plan.as_dict()["tasks"]},
        ) as step:
            tool_result = self.tool_executor.execute_many(
                session,
                bound_plan,
                valid_task_ids=bound_gate.valid_task_ids,
            )
            step["output"] = tool_result.as_dict()
        if tool_result.reranker_runs:
            trace.record("reranker", output_data={"runs": tool_result.reranker_runs})
        else:
            trace.skip("reranker", "No document RAG results required reranking")

        with trace.step(
            "evidence_merging",
            {"evidence_count": len(tool_result.evidence)},
        ) as step:
            evidence_pack = self.evidence_merger.merge(
                query=pipeline_query,
                plan=bound_plan,
                evidence=tool_result.evidence,
            )
            step["output"] = evidence_pack.as_dict()

        with trace.step(
            "evidence_consistency",
            {
                "task_ids": list(bound_gate.valid_task_ids),
                "evidence_count": len(evidence_pack.items),
            },
        ) as step:
            evidence_gate = self.consistency_gate.check_evidence(
                plan=bound_plan,
                evidence=evidence_pack.items,
                valid_task_ids=bound_gate.valid_task_ids,
            )
            step["output"] = evidence_gate.model_dump(mode="json")
        gate_messages = [
            violation.message
            for violation in (
                *bound_gate.violations,
                *evidence_gate.violations,
            )
        ]
        if gate_messages:
            evidence_pack = evidence_pack.model_copy(
                update={
                    "missing_info": list(
                        dict.fromkeys(
                            [*evidence_pack.missing_info, *gate_messages]
                        )
                    )
                }
            )

        with trace.step("context_builder", {"count": len(evidence_pack.items)}) as step:
            context = evidence_pack.to_context()
            step["output"] = {
                "item_count": len(context["items"]),
                "total_chars": context["total_chars"],
                "source_ids": [item["source_id"] for item in context["items"]],
                "source_types": [item["source_type"] for item in context["items"]],
                "tasks": [task.intent.value for task in evidence_pack.tasks],
                "missing_info": evidence_pack.missing_info,
                "conflicts": evidence_pack.conflicts,
            }
        if self.settings.debug:
            debug_data.update(
                self._orchestration_debug_payload(
                    binding_result=binding_pipeline_result,
                    gate_report=bound_gate,
                    plan=plan,
                )
            )
            debug_data["retrieval"] = {
                "planner_plan": plan.as_dict(),
                "bound_plan": bound_plan.as_dict(),
                "bound_gate": bound_gate.model_dump(mode="json"),
                "evidence_gate": evidence_gate.model_dump(mode="json"),
                "evidence": evidence_pack.as_dict(),
                "context_items": len(context["items"]),
                "pipeline": "evidence_synthesis",
            }

        primary_intent = bound_plan.primary_intent
        confidence = plan.confidence
        if primary_intent in {Intent.GREETING, Intent.CHITCHAT}:
            with trace.step(
                "prompt_builder",
                {"intent": primary_intent.value, "context_items": 0},
            ) as step:
                from app.generation.prompts import build_chitchat_prompt

                prompt = build_chitchat_prompt(
                    query=pipeline_query,
                    intent=primary_intent,
                    confidence=confidence,
                )
                step["output"] = {"prompt_chars": len(prompt)}
            try:
                with trace.step(
                    "synthesis_generation",
                    {"mode": "no_rag_social", "intent": primary_intent.value},
                ) as step:
                    response, generation_meta = await self.generator.generate_chitchat_with_retry(
                        query=pipeline_query,
                        confidence=confidence,
                        session=session,
                        intent=primary_intent,
                    )
                    step["output"] = generation_meta["llm"]
                with trace.step("json_validation", {}) as step:
                    step["output"] = generation_meta["validation"]
            except Exception as exc:
                response = self.generator.direct_response(
                    intent=primary_intent,
                    confidence=confidence,
                    context=context,
                )
                response.degraded = True
                trace.record(
                    "json_validation",
                    output_data={"fallback": True, "reason": str(exc)},
                    status="failed",
                    error_message=str(exc),
                )
                trace.record(
                    "generation_fallback",
                    output_data={"answer_type": response.answer_type},
                    status="success",
                )
        elif (
            context["items"]
            and self.settings.enable_evidence_synthesis
            and not evidence_pack.missing_info
        ):
            synthesis_payload = evidence_pack.to_prompt_payload()
            with trace.step(
                "prompt_builder",
                {
                    "intent": primary_intent.value,
                    "context_items": len(context["items"]),
                },
            ) as step:
                from app.generation.prompts import build_synthesis_prompt

                prompt = build_synthesis_prompt(
                    query=pipeline_query,
                    intent=primary_intent,
                    confidence=confidence,
                    evidence_pack=synthesis_payload,
                )
                step["output"] = {"prompt_chars": len(prompt)}
            try:
                with trace.step(
                    "synthesis_generation",
                    {"model": self.settings.llm_generation_model},
                ) as step:
                    response, generation_meta = await self.generator.generate_synthesis_with_retry(
                        query=pipeline_query,
                        intent=primary_intent,
                        confidence=confidence,
                        evidence_pack=synthesis_payload,
                        context=context,
                        session=session,
                    )
                    step["output"] = generation_meta["llm"]
                with trace.step("json_validation", {}) as step:
                    step["output"] = generation_meta["validation"]
            except Exception as exc:
                response = self.generator.fallback_from_context(
                    intent=primary_intent,
                    confidence=confidence,
                    context=context,
                )
                fallback_output = (
                    {"fallback": True, **exc.metadata}
                    if isinstance(exc, GenerationValidationError)
                    else {"fallback": True, "reason": str(exc)}
                )
                trace.record(
                    "json_validation",
                    output_data=fallback_output,
                    status="failed",
                    error_message=str(exc),
                )
                trace.record(
                    "generation_fallback",
                    output_data={
                        "answer_type": response.answer_type,
                        "evidence_count": len(context["items"]),
                    },
                    status="success",
                )
        else:
            reason = (
                "Evidence is incomplete"
                if evidence_pack.missing_info
                else "Evidence synthesis disabled or no evidence"
            )
            trace.skip("prompt_builder", reason)
            trace.skip("synthesis_generation", reason)
            if evidence_pack.missing_info and context["items"]:
                response = self.generator.partial_evidence_response(
                    intent=primary_intent,
                    confidence=confidence,
                    context=context,
                    missing_info=evidence_pack.missing_info,
                )
            else:
                response = self.generator.direct_response(
                    intent=primary_intent,
                    confidence=confidence,
                    context=context,
                    clarification_reason="task_plan_missing_evidence"
                    if evidence_pack.missing_info and not context["items"]
                    else None,
                    clarification_message=bound_plan.clarification_question,
                )
            with trace.step("json_validation", {"answer_type": response.answer_type}) as step:
                response = self.generator.validator.validate(
                    response.model_dump(mode="json"),
                    context=context,
                    session=session,
                )
                step["output"] = {"valid": True}

        detected_intents = [task.intent.value for task in plan.tasks]
        conversation_state = self._build_conversation_state(
            history,
            bound_plan,
            evidence_pack,
            passed_task_ids=evidence_gate.valid_task_ids,
        )
        suggestions: list[ChatSuggestion] = []
        if self.settings.enable_contextual_suggestions:
            with trace.step(
                "interest_state_update",
                {
                    "task_ids": list(evidence_gate.valid_task_ids),
                    "selected_suggestion_id": request.selected_suggestion_id,
                },
            ) as step:
                interest_state = self.suggestion_engine.build_interest_state(
                    original_query=request.message,
                    history=history,
                    conversation_state=conversation_state,
                    plan=bound_plan,
                    evidence_pack=evidence_pack,
                    valid_task_ids=evidence_gate.valid_task_ids,
                )
                conversation_state["interest_state"] = (
                    interest_state.model_dump(mode="json")
                )
                step["output"] = conversation_state["interest_state"]
            with trace.step(
                "suggestion_generation",
                {
                    "primary_intent": bound_plan.primary_intent.value,
                    "journey_stage": interest_state.journey_stage.value,
                },
            ) as step:
                suggestion_candidates = (
                    self.suggestion_engine.generate_candidates(
                        original_query=request.message,
                        plan=bound_plan,
                        evidence_pack=evidence_pack,
                        interest_state=interest_state,
                        valid_task_ids=evidence_gate.valid_task_ids,
                    )
                )
                step["output"] = {
                    "candidate_count": len(suggestion_candidates),
                    "candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in suggestion_candidates
                    ],
                }
            with trace.step(
                "suggestion_consistency",
                {"candidate_count": len(suggestion_candidates)},
            ) as step:
                suggestion_selection = (
                    self.suggestion_engine.rank_and_gate(
                        candidates=suggestion_candidates,
                        plan=bound_plan,
                        evidence_pack=evidence_pack,
                        conversation_state=conversation_state,
                        interest_state=interest_state,
                    )
                )
                suggestions = self.suggestion_engine.to_chat_suggestions(
                    suggestion_selection
                )
                conversation_state["suggestion_state"] = (
                    self.suggestion_engine.update_suggestion_state(
                        conversation_state=conversation_state,
                        selection=suggestion_selection,
                        selected_suggestion_id=request.selected_suggestion_id,
                    )
                )
                step["output"] = {
                    "accepted": [
                        item.model_dump(mode="json")
                        for item in suggestion_selection.accepted
                    ],
                    "rejected": [
                        item.model_dump(mode="json")
                        for item in suggestion_selection.rejected
                    ],
                    "suggestion_state": conversation_state[
                        "suggestion_state"
                    ],
                }
            if self.settings.debug:
                debug_data["suggestions"] = {
                    "interest_state": conversation_state["interest_state"],
                    "items": [
                        suggestion.model_dump(mode="json")
                        for suggestion in suggestions
                    ],
                }
        else:
            trace.skip(
                "interest_state_update",
                "Contextual suggestions disabled",
            )
            trace.skip(
                "suggestion_generation",
                "Contextual suggestions disabled",
            )
            trace.skip(
                "suggestion_consistency",
                "Contextual suggestions disabled",
            )

        memory_save_payload = {
            "session_id": request.session_id,
            "user_content": request.message,
            "assistant_content": response.result.text,
            "detected_intents": detected_intents,
            "entities": {
                "tasks": {
                    task.task_id: {
                        "intent": task.intent.value,
                        "entity_names": list(task.entity_names),
                        "resolved_ids": list(task.resolved_ids),
                        "reference_mode": task.reference_mode.value,
                        "binding_source": task.binding_source.value,
                    }
                    for task in bound_plan.tasks
                    if task.task_id in evidence_gate.valid_task_ids
                },
            },
            "resolved_ids": {
                "evidence_ids": [
                    {
                        "type": item.source_type,
                        "id": item.source_id,
                        "task_id": item.task_id,
                    }
                    for item in evidence_pack.items
                ],
                "active_product_ids": conversation_state["active_product_ids"],
                "active_service_ids": conversation_state["active_service_ids"],
            },
            "state": conversation_state,
            "trace_id": trace.trace_id,
            "suggestion_count": len(suggestions),
        }

        return self._finish(
            session,
            trace,
            response,
            request.debug,
            debug_data,
            confidence,
            suggestions=suggestions,
            memory_save_payload=memory_save_payload,
        )

    @staticmethod
    def _orchestration_debug_payload(
        *,
        binding_result,
        gate_report,
        plan,
    ) -> dict[str, Any]:
        resolution = binding_result.resolutions[0] if binding_result.resolutions else None
        bound_task = (
            binding_result.bound_plan.tasks[0]
            if binding_result.bound_plan.tasks
            else None
        )
        decision = binding_result.decisions[0] if binding_result.decisions else None
        return {
            "resolution_status": resolution.status if resolution else "unknown",
            "resolution_source": resolution.source if resolution else "unknown",
            "gate_status": gate_report.status.value if gate_report else "unknown",
            "bound_task": bound_task.model_dump(mode="json") if bound_task else {},
            "planned_tasks": [
                task.model_dump(mode="json")
                for task in plan.tasks
            ]
            if plan
            else [],
            "decision": decision.model_dump(mode="json") if decision else {},
            "violations": [
                violation.model_dump(mode="json")
                for violation in gate_report.violations
            ]
            if gate_report
            else [],
        }

    def _finish(
        self,
        session: Session,
        trace: TraceRecorder,
        response,
        debug: bool,
        debug_data: dict[str, Any],
        confidence: float,
        suggestions: list[ChatSuggestion] | None = None,
        memory_save_payload: dict[str, Any] | None = None,
    ) -> ChatResponse:
        with trace.step("asset_resolver", {"text_chars": len(response.result.text)}) as step:
            response = self.renderer.resolve_assets(session, response)
            step["output"] = {
                "resolved": len(response.result.assets),
                "missing": response.result.missing_assets,
            }
        with trace.step("response_rendering", {"debug": self.settings.debug}) as step:
            rendered = self.renderer.render(
                str(trace.trace_id),
                response,
                suggestions=suggestions,
                debug=self.settings.debug,
                debug_data=debug_data,
            )
            step["output"] = {
                "intent": rendered.intent.value,
                "answer_type": response.answer_type,
                "degraded": response.degraded,
                "safety": response.safety.model_dump(mode="json"),
            }
        if memory_save_payload is not None:
            with trace.step(
                "memory_save",
                {"session_id": memory_save_payload.get("session_id")},
            ) as step:
                save_result = self.memory.save_exchange(
                    session,
                    session_id=memory_save_payload.get("session_id"),
                    user_content=memory_save_payload["user_content"],
                    assistant_content=memory_save_payload["assistant_content"],
                    detected_intents=memory_save_payload["detected_intents"],
                    entities=memory_save_payload["entities"],
                    resolved_ids=memory_save_payload["resolved_ids"],
                    state=memory_save_payload["state"],
                    trace_id=memory_save_payload["trace_id"],
                )
                step["output"] = {
                    "saved": bool(memory_save_payload.get("session_id")),
                    "detected_intents": memory_save_payload["detected_intents"],
                    "state": memory_save_payload["state"],
                    "summary": save_result.get("summary"),
                    "suggestion_count": memory_save_payload["suggestion_count"],
                }
        trace.finish(
            intent=response.intent.value,
            confidence=confidence,
            answer=rendered.model_dump(mode="json"),
            status="degraded" if response.degraded else "success",
        )
        self.langfuse.send_trace(
            id=str(trace.trace_id),
            name="simplydent-chat",
            session_id=trace.trace.session_id,
            input={"query": trace.trace.user_query},
            output=rendered.model_dump(mode="json"),
            metadata={
                "intent": response.intent.value,
                "confidence": confidence,
                "latency_ms": trace.trace.total_latency_ms,
            },
        )
        return rendered

    @staticmethod
    def _should_use_llm(
        intent: Intent,
        plan: RetrievalPlan,
        structured: list[RetrievalResult],
    ) -> bool:
        if plan.mode == RetrievalMode.NO_RAG_LLM:
            return True
        if intent == Intent.PRODUCT_COMPARE:
            return len(structured) >= 2
        if intent == Intent.FAQ:
            return False
        return plan.uses_hybrid

    @staticmethod
    def _is_pure_social_query(query: str, routed) -> bool:
        if routed.intent not in {Intent.GREETING, Intent.CHITCHAT}:
            return False
        features = QueryFeatures.extract(
            query,
            entity_names_in_query=tuple(routed.entities or ()),
        )
        has_business_cue = any(
            (
                features.is_schedule_query,
                features.is_availability,
                features.is_price,
                features.is_duration,
                features.asks_list,
                features.asks_compare,
                features.has_entity_mention,
            )
        )
        return not has_business_cue

    @staticmethod
    def _effective_entities(routed, resolution) -> list[str]:
        if routed.source in {"ollama", "openai"} and routed.entities:
            return routed.entities
        return resolution.names or routed.entities

    @staticmethod
    def _skip_hybrid(trace: TraceRecorder, reason: str) -> None:
        for step in (
            "dense_retrieval",
            "sparse_retrieval",
            "rrf_fusion",
            "reranker",
        ):
            trace.skip(step, reason)

    @staticmethod
    def _build_conversation_state(
        history: dict[str, Any],
        plan,
        evidence_pack,
        *,
        passed_task_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        state = _normalize_conversation_state(history.get("state"))
        allowed_task_ids = (
            set(passed_task_ids)
            if passed_task_ids is not None
            else {task.task_id for task in plan.tasks}
        )
        product_ids: list[str] = []
        product_names: list[str] = []
        service_ids: list[str] = []
        service_names: list[str] = []
        for task in plan.tasks:
            if (
                task.task_id not in allowed_task_ids
                or not capability_for(task.intent).persist_to_memory
                or task.clarification_required
                or task.resolution_status != "resolved"
            ):
                continue
            if task.entity_type == "product":
                product_ids.extend(task.resolved_ids)
                product_names.extend(task.entity_names)
            elif task.entity_type == "service":
                service_ids.extend(task.resolved_ids)
                service_names.extend(task.entity_names)

        if product_ids or product_names:
            state["active_product_ids"] = _dedupe(product_ids)
            state["active_product_names"] = _dedupe(product_names)
        if service_ids or service_names:
            state["active_service_ids"] = _dedupe(service_ids)
            state["active_service_names"] = _dedupe(service_names)

        primary_capability = capability_for(plan.primary_intent)
        if product_ids or product_names:
            state["active_domain"] = "product"
            if state["active_product_names"]:
                state["active_topic"] = state["active_product_names"][0]
        elif service_ids or service_names:
            state["active_domain"] = "service"
            if state["active_service_names"]:
                state["active_topic"] = state["active_service_names"][0]
        elif primary_capability.entity_domain in {"product", "service"}:
            state["active_domain"] = primary_capability.entity_domain

        current_intents = [task.intent.value for task in plan.tasks]
        state["last_intents"] = _dedupe([*current_intents, *state["last_intents"]])[:8]
        filters = dict(state.get("last_filters") or {})
        for task in plan.tasks:
            if task.task_id not in allowed_task_ids:
                continue
            constraints = task.filters.as_constraints()
            constraints.pop("sort", None)
            constraints = {
                key: value
                for key, value in constraints.items()
                if value not in (None, [], (), "")
            }
            if constraints:
                key = task.entity_type or task.intent.value.lower()
                filters[key] = constraints
        state["last_filters"] = filters
        clarification = plan.clarification_question or next(
            (
                task.clarification_question
                for task in plan.tasks
                if task.clarification_required and task.clarification_question
            ),
            None,
        )
        state["pending_clarification"] = (
            {
                "message": clarification,
                "intents": current_intents,
                "entities": [
                    name
                    for task in plan.tasks
                    if task.task_id in allowed_task_ids
                    for name in task.entity_names
                ],
            }
            if clarification
            else None
        )
        return state

    @staticmethod
    def _candidate_clarification(task, candidates: list[dict[str, Any]]) -> str:
        names = [
            str(candidate.get("name"))
            for candidate in candidates[:5]
            if candidate.get("name")
        ]
        if not names:
            return "Bạn vui lòng nhập rõ hơn tên sản phẩm hoặc dịch vụ cần hỏi."
        label = entity_label(capability_for(task.intent).entity_domain)
        return (
            f"Tôi thấy nhiều {label} gần giống yêu cầu của bạn: "
            f"{', '.join(names)}. Bạn muốn hỏi mục nào?"
        )

    @staticmethod
    def _result_summary(results: list[RetrievalResult]) -> dict[str, Any]:
        return {
            "count": len(results),
            "results": [
                {
                    "id": item.source_id,
                    "type": item.source_type,
                    "score": round(item.score, 6),
                    "canonical_key": item.canonical_key,
                    "ranks": item.ranks,
                }
                for item in results[:20]
            ],
        }

    def _dense_result_sets(
        self,
        session: Session,
        rewrite,
        intent: Intent,
    ) -> dict[str, list[RetrievalResult]]:
        result_sets = {
            f"dense_original_{name}": values
            for name, values in self.dense.retrieve_by_source(
                session,
                rewrite.original_query,
                intent,
            ).items()
        }
        if rewrite.hyde_used and rewrite.hyde_query:
            result_sets.update(
                {
                    f"dense_hyde_{name}": values
                    for name, values in self.dense.retrieve_by_source(
                        session,
                        rewrite.hyde_query,
                        intent,
                    ).items()
                }
            )
        return result_sets

    def _rrf_weights(
        self,
        result_sets: dict[str, list[RetrievalResult]],
    ) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name in result_sets:
            if name == "structured":
                weights[name] = self.settings.structured_rrf_weight
            elif name.startswith("sparse_"):
                weights[name] = self.settings.sparse_rrf_weight
            else:
                weights[name] = self.settings.dense_rrf_weight
        return weights

    @staticmethod
    def _result_set_summary(
        result_sets: dict[str, list[RetrievalResult]],
    ) -> dict[str, Any]:
        return {
            "count": sum(len(values) for values in result_sets.values()),
            "sets": {
                name: ChatService._result_summary(values)
                for name, values in result_sets.items()
            },
            "results": [
                {
                    "id": item.source_id,
                    "type": item.source_type,
                    "score": round(item.score, 6),
                    "retriever": name,
                    "canonical_key": item.canonical_key,
                    "ranks": item.ranks,
                }
                for name, values in result_sets.items()
                for item in values[:20]
            ],
        }


def _normalize_conversation_state(value: object) -> dict[str, Any]:
    state = {
        "active_product_ids": [],
        "active_product_names": [],
        "active_service_ids": [],
        "active_service_names": [],
        "active_domain": None,
        "active_topic": None,
        "last_intents": [],
        "last_filters": {},
        "pending_clarification": None,
        "interest_state": {},
        "suggestion_state": {
            "recent_impressions": [],
            "accepted_suggestion_ids": [],
            "dismissed_suggestion_ids": [],
        },
    }
    if not isinstance(value, dict):
        return state
    for key in (
        "active_product_ids",
        "active_product_names",
        "active_service_ids",
        "active_service_names",
        "last_intents",
    ):
        state[key] = _dedupe(str(item) for item in _list_value(value.get(key)))
    active_domain = str(value.get("active_domain") or "").strip()
    state["active_domain"] = (
        active_domain if active_domain in {"product", "service", "faq", "clinic_info"} else None
    )
    active_topic = str(value.get("active_topic") or "").strip()
    state["active_topic"] = active_topic or None
    last_filters = value.get("last_filters")
    state["last_filters"] = last_filters if isinstance(last_filters, dict) else {}
    pending = value.get("pending_clarification")
    state["pending_clarification"] = pending if isinstance(pending, dict) else None
    interest_state = value.get("interest_state")
    state["interest_state"] = (
        dict(interest_state) if isinstance(interest_state, dict) else {}
    )
    suggestion_state = value.get("suggestion_state")
    if isinstance(suggestion_state, dict):
        state["suggestion_state"] = {
            "recent_impressions": _dedupe(
                str(item)
                for item in _list_value(
                    suggestion_state.get("recent_impressions")
                )
            )[:24],
            "accepted_suggestion_ids": _dedupe(
                str(item)
                for item in _list_value(
                    suggestion_state.get("accepted_suggestion_ids")
                )
            )[:24],
            "dismissed_suggestion_ids": _dedupe(
                str(item)
                for item in _list_value(
                    suggestion_state.get("dismissed_suggestion_ids")
                )
            )[:24],
        }
    return state


def _list_value(value: object) -> list[object]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    return [value]


def _dedupe(values) -> list[str]:
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            items.append(text)
    return list(dict.fromkeys(items))
