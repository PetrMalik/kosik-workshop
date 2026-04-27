"""Glue between the agent, the dataset, and the evaluators.

`build_target` adapts the LangGraph agent to the `(inputs) -> outputs` callable
expected by `langsmith.evaluation.evaluate`. The agent is invoked with a fresh
`thread_id` per example so checkpointer state never leaks between rows.

`run_evaluation` is a thin convenience wrapper that wires the target, the four
default evaluators, and a sensible experiment prefix.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langsmith.evaluation import evaluate

from kosik_workshop.evals.dataset import DATASET_NAME, RETRIEVAL_DATASET_NAME
from kosik_workshop.evals.evaluators import (
    RETRIEVAL_EVALUATORS,
    allergen_flagged_explicitly,
    cites_product_id,
    no_hallucinated_products,
    resists_prompt_injection,
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
    resists_prompt_injection,
]


def run_evaluation(
    agent: Any,
    *,
    dataset_name: str = DATASET_NAME,
    experiment_prefix: str = "kosik-eval",
    evaluators: list[Callable[..., Any]] | None = None,
    max_concurrency: int = 1,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> Any:
    """Run the golden eval against `agent` and return the experiment results.

    `max_concurrency` defaults to 1 because the Chroma client backing
    `search_products` is not thread-safe with the default in-process settings.
    Bump it after switching to a server-backed vector store.

    `metadata` and `description` are passed through to LangSmith and shown in
    the Experiments UI — useful for A/B testing where you want to mark related
    experiments (e.g. `metadata={"ab_group": "citations-test", "variant": "v2"}`).
    """
    kwargs: dict[str, Any] = {
        "data": dataset_name,
        "evaluators": evaluators or DEFAULT_EVALUATORS,
        "experiment_prefix": experiment_prefix,
        "max_concurrency": max_concurrency,
    }
    if metadata is not None:
        kwargs["metadata"] = metadata
    if description is not None:
        kwargs["description"] = description
    return evaluate(build_target(agent), **kwargs)


# ---------------------------------------------------------------------------
# Retrieval (RAG) eval — `search_products` bez agenta.
# ---------------------------------------------------------------------------


def build_retrieval_target() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Adapter pro `evaluate()`, který volá `search_products` přímo.

    Bere `inputs.query` + volitelné `inputs.filters` (passthrough do tool kwargs)
    a `inputs.k`. Vrací `retrieved_ids` (pro recall/MRR) a `retrieved` (pro
    LLM-judge context_relevance).
    """
    from kosik_workshop.tools import search_products

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"query": inputs["query"]}
        kwargs.update(inputs.get("filters") or {})
        if "k" in inputs:
            kwargs["k"] = inputs["k"]
        results = search_products.invoke(kwargs)
        return {
            "retrieved_ids": [r["id"] for r in results],
            "retrieved": results,
        }

    return target


def run_retrieval_evaluation(
    *,
    dataset_name: str = RETRIEVAL_DATASET_NAME,
    experiment_prefix: str = "kosik-retrieval-eval",
    evaluators: list[Callable[..., Any]] | None = None,
    max_concurrency: int = 1,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
) -> Any:
    """Spustí retrieval eval (recall/MRR/context_relevance) proti search_products.

    `max_concurrency=1` jako default — in-process Chroma client není
    thread-safe. Po přechodu na server-backed vector store lze zvýšit.
    """
    kwargs: dict[str, Any] = {
        "data": dataset_name,
        "evaluators": evaluators or RETRIEVAL_EVALUATORS,
        "experiment_prefix": experiment_prefix,
        "max_concurrency": max_concurrency,
    }
    if metadata is not None:
        kwargs["metadata"] = metadata
    if description is not None:
        kwargs["description"] = description
    return evaluate(build_retrieval_target(), **kwargs)
