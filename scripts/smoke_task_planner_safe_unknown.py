"""Smoke-check PlannerLLM failures become safe UNKNOWN plans."""

import asyncio

from app.config import get_settings
from app.constants import Intent
from app.orchestration.task_planner import TaskPlanner


class FailingLLM:
    async def generate(self, **kwargs):
        raise RuntimeError("planner unavailable")


async def run() -> None:
    settings = get_settings().model_copy(update={"enable_llm_router": True})
    plan = await TaskPlanner().plan(
        query="abc",
        history={},
        settings=settings,
        llm=FailingLLM(),
    )
    assert plan.source == "safe_unknown"
    assert plan.tasks[0].intent == Intent.UNKNOWN
    assert plan.tasks[0].planner_needs_clarification is True
    assert plan.metadata["planner_fallback"] == "safe_unknown_after_llm_error"
    print(plan.as_dict())


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
