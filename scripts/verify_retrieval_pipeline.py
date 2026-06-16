import argparse
import asyncio
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import RagTraceStep
from app.db.session import get_engine
from app.generation.schemas import ChatRequest
from app.services.chat import ChatService

DEFAULT_QUERIES = [
    "Thông tin AquaJet Mini Water Flosser",
    "Răng nhạy cảm do đâu và xử lý thế nào?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real chat requests and print retrieval trace steps.",
    )
    parser.add_argument("queries", nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--router-timeout", type=int, default=2)
    parser.add_argument("--generation-timeout", type=int, default=1)
    parser.add_argument("--enable-hyde", action="store_true")
    parser.add_argument("--enable-reranker", action="store_true")
    parser.add_argument("--disable-llm-router", action="store_true")
    return parser.parse_args()


async def verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = get_settings().model_copy(
        update={
            "enable_llm_router": (
                False if args.disable_llm_router else get_settings().enable_llm_router
            ),
            "enable_hyde": args.enable_hyde,
            "enable_reranker": args.enable_reranker,
            "router_timeout_seconds": args.router_timeout,
            "ollama_timeout_seconds": args.generation_timeout,
        }
    )
    output: list[dict[str, Any]] = []
    with Session(get_engine()) as session:
        service = ChatService(settings)
        for query in args.queries:
            response = await service.chat(
                session,
                ChatRequest(
                    message=query,
                    session_id="retrieval-verification",
                    debug=False,
                ),
            )
            steps = session.scalars(
                select(RagTraceStep)
                .where(RagTraceStep.trace_id == response.trace_id)
                .order_by(RagTraceStep.created_at)
            ).all()
            output.append(
                {
                    "query": query,
                    "trace_id": str(response.trace_id),
                    "intent": response.intent.value,
                    "answer": response.answer.text,
                    "steps": [
                        {
                            "name": step.step_name,
                            "status": step.status,
                            "latency_ms": step.latency_ms,
                            "output": step.output_json,
                            "error": step.error_message,
                        }
                        for step in steps
                    ],
                }
            )
    return output


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            asyncio.run(verify(args)),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
