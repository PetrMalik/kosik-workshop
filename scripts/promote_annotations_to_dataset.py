"""Promotni human-anotované runs z annotation queue do golden datasetu.

Pro každý run v queue, který má kompletní anotaci:
- "good" = correct_tools=1 AND helpful=1 AND safe=1 → přidej jako positive case
- "bad"  = helpful=0 OR safe=0 → přidej jako regression case (`is_regression: True`)

Z trace extrahuje:
- inputs.question — z root run.inputs (HumanMessage content)
- expected_tools — jména tool-runů z child traces
- expects_recommendation — heuristika: obsahuje answer slug-like token?
- human_comment — volnotextová poznámka anotátora (jen pro regression)

Idempotentní: skipne examples, jejichž `question` už v datasetu je.

Usage:
    uv run python scripts/promote_annotations_to_dataset.py --dry-run
    uv run python scripts/promote_annotations_to_dataset.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langsmith import Client

from kosik_workshop.config import load_env
from kosik_workshop.evals.dataset import DATASET_NAME

QUEUE_NAME_DEFAULT = "kosik-human-review"
HUMAN_KEYS = ("correct_tools", "helpful", "safe")
SLUG_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+){2,}\b")


@dataclass
class AnnotatedRun:
    run_id: UUID
    question: str
    answer: str
    tool_names: list[str]
    correct_tools: int
    helpful: int
    safe: int
    comment: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default=QUEUE_NAME_DEFAULT)
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _walk_queue_run_ids(client: Client, queue_id: UUID) -> list[UUID]:
    """Vrací run_ids jak z 'Needs Review', tak z 'Completed' sekce queue.

    `get_run_from_annotation_queue(index=...)` vidí jen pending runs — jakmile
    reviewer klikne Done, run ze queue zmizí. Completed runs proto dohledáme
    přes feedback s klíči, které queue používá (`HUMAN_KEYS`).
    """
    ids: list[UUID] = []
    seen: set[UUID] = set()

    idx = 0
    while True:
        try:
            run = client.get_run_from_annotation_queue(queue_id, index=idx)
        except Exception:  # noqa: BLE001
            break
        if run.id not in seen:
            ids.append(run.id)
            seen.add(run.id)
        idx += 1
        if idx > 5000:
            break

    for fb in client.list_feedback(feedback_key=list(HUMAN_KEYS)):
        if fb.run_id and fb.run_id not in seen:
            ids.append(fb.run_id)
            seen.add(fb.run_id)

    return ids


def _load_feedback_map(client: Client, run_ids: list[UUID]) -> dict[UUID, dict[str, Any]]:
    """run_id → {key: {"score": ..., "comment": ...}}"""
    out: dict[UUID, dict[str, Any]] = defaultdict(dict)
    if not run_ids:
        return out
    for fb in client.list_feedback(run_ids=run_ids):
        out[fb.run_id][fb.key] = {"score": fb.score, "comment": fb.comment or ""}
    return out


def _extract_question(run: Any) -> str:
    """Najdi text uživatelské otázky v run.inputs.

    Agent dostává `{"messages": [HumanMessage(content="...")]}`. Inputs jsou
    serializované — sáhneme po prvním 'content' v 'messages'.
    """
    inputs = getattr(run, "inputs", None) or {}
    msgs = inputs.get("messages") or inputs.get("input") or []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("content"):
                return str(m["content"])
            if isinstance(m, list) and len(m) >= 2 and m[0] in ("human", "user"):
                return str(m[1])
    if isinstance(msgs, str):
        return msgs
    return ""


def _extract_answer(run: Any) -> str:
    outputs = getattr(run, "outputs", None) or {}
    msgs = outputs.get("messages") or []
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        if isinstance(last, dict) and last.get("content"):
            return str(last["content"])
    if "output" in outputs:
        return str(outputs["output"])
    return ""


def _list_tool_names(client: Client, root_run: Any) -> list[str]:
    trace_id = getattr(root_run, "trace_id", None) or root_run.id
    project_id = getattr(root_run, "session_id", None)
    try:
        children = list(
            client.list_runs(
                project_id=project_id,
                trace_id=trace_id,
                run_type="tool",
            )
        )
    except Exception:  # noqa: BLE001
        return []
    seen: list[str] = []
    for c in children:
        if c.name and c.name not in seen:
            seen.append(c.name)
    return seen


def _expects_recommendation(answer: str) -> bool:
    return bool(SLUG_RE.search(answer.lower()))


def _build_annotated(client: Client, run: Any, fb: dict[str, Any]) -> AnnotatedRun | None:
    if not all(k in fb for k in HUMAN_KEYS):
        return None
    if any(fb[k]["score"] is None for k in HUMAN_KEYS):
        return None
    question = _extract_question(run)
    if not question:
        return None
    return AnnotatedRun(
        run_id=run.id,
        question=question,
        answer=_extract_answer(run),
        tool_names=_list_tool_names(client, run),
        correct_tools=int(round(float(fb["correct_tools"]["score"]))),
        helpful=int(round(float(fb["helpful"]["score"]))),
        safe=int(round(float(fb["safe"]["score"]))),
        comment=" / ".join(
            v["comment"] for k, v in fb.items() if v.get("comment") and k in HUMAN_KEYS
        ),
    )


def _classify(a: AnnotatedRun) -> str | None:
    if a.correct_tools == 1 and a.helpful == 1 and a.safe == 1:
        return "good"
    if a.helpful == 0 or a.safe == 0:
        return "bad"
    return None


def _build_example(a: AnnotatedRun, kind: str) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "expected_tools": a.tool_names,
        "expected_args": {},
        "expects_recommendation": _expects_recommendation(a.answer),
        "user_allergens_context": [],
        "category": "human_promoted" if kind == "good" else "human_regression",
        "source": "annotation_queue",
        "human_run_id": str(a.run_id),
    }
    if kind == "bad":
        outputs["is_regression"] = True
        outputs["human_comment"] = a.comment or ""
    return {"inputs": {"question": a.question}, "outputs": outputs}


def main() -> int:
    args = parse_args()
    load_env()

    client = Client()
    queue = next(iter(client.list_annotation_queues(name=args.queue)), None)
    if queue is None:
        print(f"error: queue '{args.queue}' not found", file=sys.stderr)
        return 1

    run_ids = _walk_queue_run_ids(client, queue.id)
    fb_map = _load_feedback_map(client, run_ids)

    runs_by_id = {r.id: r for r in client.list_runs(id=run_ids)} if run_ids else {}

    annotated: list[AnnotatedRun] = []
    pending = 0
    for rid in run_ids:
        run = runs_by_id.get(rid)
        if run is None:
            continue
        a = _build_annotated(client, run, fb_map.get(rid, {}))
        if a is None:
            pending += 1
        else:
            annotated.append(a)

    dataset = next(iter(client.list_datasets(dataset_name=args.dataset)), None)
    if dataset is None:
        print(f"error: dataset '{args.dataset}' not found — run seed_eval_dataset.py first")
        return 1
    existing_questions = {
        ex.inputs.get("question") for ex in client.list_examples(dataset_id=dataset.id)
    }

    to_add: list[tuple[str, dict[str, Any]]] = []
    skipped_existing = 0
    skipped_neutral = 0
    for a in annotated:
        kind = _classify(a)
        if kind is None:
            skipped_neutral += 1
            continue
        if a.question in existing_questions:
            skipped_existing += 1
            continue
        to_add.append((kind, _build_example(a, kind)))
        existing_questions.add(a.question)  # dedup within batch too

    n_good = sum(1 for k, _ in to_add if k == "good")
    n_bad = sum(1 for k, _ in to_add if k == "bad")

    print(f"Queue runs:           {len(run_ids)}")
    print(f"  fully annotated:    {len(annotated)}")
    print(f"  pending:            {pending}")
    print()
    print(f"To promote:           {len(to_add)}")
    print(f"  good (positive):    {n_good}")
    print(f"  bad  (regression):  {n_bad}")
    print(f"  skipped (in dataset): {skipped_existing}")
    print(f"  skipped (neutral):  {skipped_neutral}")
    print()

    if not to_add:
        print("Nothing to promote.")
        return 0

    if args.dry_run:
        print("DRY RUN — examples that would be created:")
        for kind, ex in to_add[:8]:
            mark = "+" if kind == "good" else "-"
            print(f"  [{mark}] {ex['inputs']['question'][:70]}")
            print(f"        tools={ex['outputs']['expected_tools']}")
        if len(to_add) > 8:
            print(f"  ... and {len(to_add) - 8} more")
        return 0

    client.create_examples(
        inputs=[ex["inputs"] for _, ex in to_add],
        outputs=[ex["outputs"] for _, ex in to_add],
        dataset_id=dataset.id,
    )
    print(f"✓ Promoted {len(to_add)} examples to '{args.dataset}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
