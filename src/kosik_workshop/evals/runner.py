"""Glue between the agent, the dataset, and the evaluators.

`build_target` adapts the LangGraph agent to the `(inputs) -> outputs` callable
expected by `langsmith.evaluation.evaluate`. The agent is invoked with a fresh
`thread_id` per example so checkpointer state never leaks between rows.

`run_evaluation` is a thin convenience wrapper that wires the target, the four
default evaluators, and a sensible experiment prefix.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langsmith.evaluation import evaluate

from kosik_workshop.evals.dataset import DATASET_NAME
from kosik_workshop.evals.evaluators import (
    allergen_flagged_explicitly,
    cites_product_id,
    no_hallucinated_products,
    tool_called_correctly,
)


def build_target(agent: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Adapt the LangGraph agent for `evaluate()`.

    Returns a callable that takes the example's `inputs` dict (with `question`)
    and returns `{"answer": str, "messages": [...]}`. The full message list is
    kept on the run so evaluators can inspect tool calls and outputs.
    """

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "session_id": thread_id,
                "user_id": "eval-synthetic",
                "eval_run": True,
            },
            "run_name": "kosik-agent:eval",
            "tags": ["feature:eval"],
        }
        result = agent.invoke(
            {"messages": [HumanMessage(content=inputs["question"])]},
            config=config,
        )
        last = result["messages"][-1]
        return {
            "answer": last.content if isinstance(last.content, str) else str(last.content),
            "messages": result["messages"],
        }

    return target


DEFAULT_EVALUATORS = [
    tool_called_correctly,
    cites_product_id,
    allergen_flagged_explicitly,
    no_hallucinated_products,
]


def run_evaluation(
    agent: Any,
    *,
    dataset_name: str = DATASET_NAME,
    experiment_prefix: str = "kosik-eval",
    evaluators: list[Callable[..., Any]] | None = None,
    max_concurrency: int = 1,
) -> Any:
    """Run the golden eval against `agent` and return the experiment results.

    `max_concurrency` defaults to 1 because the Chroma client backing
    `search_products` is not thread-safe with the default in-process settings.
    Bump it after switching to a server-backed vector store.
    """
    return evaluate(
        build_target(agent),
        data=dataset_name,
        evaluators=evaluators or DEFAULT_EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )
