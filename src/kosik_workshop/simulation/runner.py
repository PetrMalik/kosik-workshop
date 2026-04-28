"""Workshop demo simulator.

Runs a curated set of queries through the agent, tagging each trace so the
LangSmith UI can filter per-scenario, per-feature, per-user. Posts a heuristic
quality feedback to each run so the dashboard has an online metric to plot.

Two scenarios:
- `baseline` — full 30-query set, current dev prompt.
- `recovery` — 10-query subset, same prompt (lighter traffic shape).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import Client

from kosik_workshop.agent import build_agent
from kosik_workshop.simulation.demo_queries import DEMO_QUERIES, DemoQuery
from kosik_workshop.simulation.demo_users import by_id

log = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario: str
    run_count: int
    feedback_pass: int
    feedback_fail: int
    duration_s: float
    sample_thread_ids: list[str]


def _heuristic_quality(query: DemoQuery, answer: str) -> tuple[int, str]:
    """Cheap, deterministic feedback signal.

    Not a substitute for the LLM judge — just enough to populate an online
    metric in dashboards immediately, per query intent.

    Returns (score 0/1, comment).
    """
    if not answer or not answer.strip():
        return 0, "empty answer"

    lower = answer.lower()

    if query.intent == "out-of-catalog":
        # Must decline rather than recommend. Look for negation phrases.
        declines = ("nemáme", "nenašel", "není v nabídce", "nemám v nabídce", "bohužel")
        return (1 if any(d in lower for d in declines) else 0, "decline check")

    if query.intent == "allergen-check":
        # When user mentions an allergen, answer must reference it explicitly.
        for allergen in ("lepek", "mléko", "mleko", "ořech", "orech", "laktóza", "laktoza"):
            if allergen in query.text.lower() and allergen in lower:
                return 1, f"allergen '{allergen}' addressed"
        return 0, "allergen not addressed in answer"

    if query.intent in ("product-search", "product-details"):
        # Answer should mention price (Kč) or a product slug-like token.
        has_price = "kč" in lower or "kc" in lower
        has_slug = "-" in answer and any(c.isalpha() for c in answer)
        return (1 if (has_price or has_slug) else 0, "product detail check")

    if query.intent == "off-topic":
        # Should redirect, not engage in chitchat.
        redirects = ("košík", "produkty", "pomoci", "pomohu", "asistent")
        return (1 if any(r in lower for r in redirects) else 0, "redirect check")

    # default: any non-trivial answer passes
    return (1 if len(answer.strip()) >= 20 else 0, "responsiveness")


def _select_queries(scenario: str) -> list[DemoQuery]:
    """Baseline runs all 30. Recovery runs a 10-query subset for speed."""
    if scenario == "recovery":
        # 10 queries spread across intents — lighter traffic shape for demo.
        return [
            q for i, q in enumerate(DEMO_QUERIES) if i in (0, 4, 10, 12, 16, 18, 22, 25, 27, 29)
        ]
    return list(DEMO_QUERIES)


def run_scenario(
    scenario: str,
    *,
    queries: Iterable[DemoQuery] | None = None,
    print_progress: bool = True,
) -> ScenarioResult:
    """Execute a demo scenario end-to-end.

    Each query gets a fresh thread_id, runs through the agent with all the
    standard tags (env, model, feature, user_id, scenario, simulated). After
    the invoke completes, posts a heuristic quality feedback to the top-level
    run so it lands on dashboards.
    """
    if scenario not in {"baseline", "recovery"}:
        raise ValueError(f"unknown scenario: {scenario}")

    selected = list(queries) if queries is not None else _select_queries(scenario)
    agent = build_agent()
    client = Client()

    feedback_pass = 0
    feedback_fail = 0
    sample_thread_ids: list[str] = []
    start = time.monotonic()

    for i, query in enumerate(selected, 1):
        user = by_id(query.user_id)
        thread_id = str(uuid.uuid4())
        if i <= 3:
            sample_thread_ids.append(thread_id)

        collector = RunCollectorCallbackHandler()
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "session_id": thread_id,
                "user_id": user.user_id,
                "user_tier": user.tier,
                "scenario": scenario,
                "simulated": True,
                "intent": query.intent,
            },
            "tags": [
                f"feature:{query.intent}",
                f"scenario:{scenario}",
                f"tier:{user.tier}",
                "surface:simulator",
                "simulated",
            ],
            "run_name": f"kosik-agent:{query.intent}",
            "callbacks": [collector],
        }

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query.text)]},
                config=config,
            )
            answer = result["messages"][-1].content
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as exc:  # noqa: BLE001
            # Demo simulator nesmí spadnout uprostřed runu — logujeme plný
            # traceback a pokračujeme. Heuristic evaluator dostane "[error] ..."
            # a označí run jako neúspěšný, takže výpadek je v dashboardu vidět.
            log.exception("agent invoke failed for query %r", query.text)
            answer = f"[error] {exc}"

        score, comment = _heuristic_quality(query, answer)
        if score == 1:
            feedback_pass += 1
        else:
            feedback_fail += 1

        # Post feedback to the top-level run captured by the collector.
        root_run_id = collector.traced_runs[0].id if collector.traced_runs else None
        if root_run_id is not None:
            try:
                client.create_feedback(
                    run_id=root_run_id,
                    key="online.heuristic_quality",
                    score=score,
                    comment=comment,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback create failed for run %s: %s", root_run_id, exc)
                if print_progress:
                    print(f"  [feedback skipped: {exc}]")

        if print_progress:
            mark = "✓" if score else "✗"
            print(
                f"[{i:02d}/{len(selected)}] {mark} {query.intent:18s} "
                f"user={user.name:9s} '{query.text[:50]}'"
            )

    duration = time.monotonic() - start
    return ScenarioResult(
        scenario=scenario,
        run_count=len(selected),
        feedback_pass=feedback_pass,
        feedback_fail=feedback_fail,
        duration_s=duration,
        sample_thread_ids=sample_thread_ids,
    )
