from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings
from app.constants import Intent, RetrievalMode
from app.generation.generator import GenerationValidationError, GroundedGenerator
from app.generation.ollama_client import OllamaClient
from app.generation.renderer import ResponseRenderer
from app.generation.schemas import ChatRequest, ChatResponse
from app.ingestion.embedder import EmbeddingService
from app.observability.langfuse_client import OptionalLangfuse
from app.observability.tracing import TraceRecorder
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.entity_resolver import DatabaseEntityResolver
from app.retrieval.planner import RetrievalPlan, RetrievalPlanner
from app.retrieval.query_rewrite import QueryRewriter
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
    DIRECT_INTENTS = {
        Intent.GREETING,
        Intent.CHITCHAT,
        Intent.CLINIC_INFO,
        Intent.PRODUCT_LIST,
        Intent.SERVICE_LIST,
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ollama = OllamaClient(settings)
        self.router = IntentRouter()
        self.entity_resolver = DatabaseEntityResolver(settings)
        self.planner = RetrievalPlanner(settings)
        self.rewriter = QueryRewriter()
        self.structured = StructuredRetriever()
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
        self.generator = GroundedGenerator(settings, self.ollama)
        self.renderer = ResponseRenderer()
        self.langfuse = OptionalLangfuse(settings)

    async def chat(self, session: Session, request: ChatRequest) -> ChatResponse:
        trace = TraceRecorder.start(session, request.message, request.session_id)
        debug_data: dict[str, Any] = {}
        try:
            product_names, service_names = self.structured.active_names(session)
            product_category_terms = active_product_category_terms(session)
            with trace.step("router_intent", {"query": request.message}) as step:
                routed = await self.router.route_with_optional_llm(
                    request.message,
                    self.settings,
                    self.ollama,
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
                        self.ollama,
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
                        {"model": self.settings.ollama_generation_model},
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

    def _finish(
        self,
        session: Session,
        trace: TraceRecorder,
        response,
        debug: bool,
        debug_data: dict[str, Any],
        confidence: float,
    ) -> ChatResponse:
        with trace.step("asset_resolver", {"text_chars": len(response.result.text)}) as step:
            response = self.renderer.resolve_assets(session, response)
            step["output"] = {
                "resolved": len(response.result.assets),
                "missing": response.result.missing_assets,
            }
        with trace.step("response_rendering", {"debug": debug}) as step:
            rendered = self.renderer.render(
                str(trace.trace_id),
                response,
                debug=debug and self.settings.debug,
                debug_data=debug_data,
            )
            step["output"] = {
                "intent": rendered.intent.value,
                "answer_type": response.answer_type,
                "safety": response.safety.model_dump(mode="json"),
            }
        trace.finish(
            intent=response.intent.value,
            confidence=confidence,
            answer=rendered.model_dump(mode="json"),
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
    def _effective_entities(routed, resolution) -> list[str]:
        if routed.source == "ollama" and routed.entities:
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
