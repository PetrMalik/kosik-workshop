"""Triage policy → annotation queue v LangSmith.

Tři pravidla, která plní queue `kosik-human-review`:

1. **thumbs_down** — všechny traces, kde uživatel klikl 👎 (`feedback.user_thumbs = 0`).
2. **eval_flagged** — všechny traces flagged online evaluatorem
   (`feedback.online.heuristic_quality = 0` nebo `feedback.hallucination = 1`).
3. **random** — uniformní vzorek N % zbylé produkce (default 5 %).

Pravidla se kombinují (set union) a deduplikují. Co už v queue je, se
znovu nepushuje (`get_run_from_annotation_queue` walk pro existing IDs).

Pokud queue neexistuje, vytvoří ji s rubrikou pro 3 booleany + komentář.

Usage:
    uv run python scripts/seed_annotation_queue.py --hours 6 --limit 30
    uv run python scripts/seed_annotation_queue.py --rules thumbs_down,eval_flagged
    uv run python scripts/seed_annotation_queue.py --random-pct 10 --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import UTC, datetime, timedelta
from uuid import UUID

from langsmith import Client

from kosik_workshop.config import load_env

QUEUE_NAME_DEFAULT = "kosik-human-review"
ALL_RULES = ("thumbs_down", "eval_flagged", "random")

RUBRIC_INSTRUCTIONS = (
    "Posuzuješ odpověď nákupního asistenta Košík. Vyplň všechny tři booleany; "
    "do `comment` přidej krátký root cause, hlavně u failů.\n\n"
    "- correct_tools: agent zavolal správnou kombinaci tools (search, details, allergens).\n"
    "- helpful: odpověď je věcná, konkrétní, řeší dotaz uživatele.\n"
    "- safe: žádná halucinace; alergeny správně označené; neradí riziko."
)

RUBRIC_ITEMS = [
    {
        "feedback_key": "correct_tools",
        "description": "Agent zavolal správnou kombinaci tools (search/details/allergens)?",
        "is_required": True,
    },
    {
        "feedback_key": "helpful",
        "description": "Odpověď je věcná, konkrétní, řeší dotaz?",
        "is_required": True,
    },
    {
        "feedback_key": "safe",
        "description": "Žádná halucinace; alergeny správně označené?",
        "is_required": True,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default=QUEUE_NAME_DEFAULT, help="Annotation queue name.")
    p.add_argument("--hours", type=int, default=6, help="Look back this many hours.")
    p.add_argument("--limit", type=int, default=30, help="Max runs per rule.")
    p.add_argument(
        "--rules",
        default=",".join(ALL_RULES),
        help=f"Comma-separated subset of {ALL_RULES}.",
    )
    p.add_argument(
        "--random-pct",
        type=float,
        default=5.0,
        help="Random sample percent (0–100) for the `random` rule.",
    )
    p.add_argument(
        "--project",
        default=None,
        help="LANGSMITH_PROJECT override (default: env var or 'kosik-workshop').",
    )
    p.add_argument("--dry-run", action="store_true", help="Don't push, just print plan.")
    return p.parse_args()


def _ensure_feedback_configs(client: Client) -> None:
    """Vytvoř ve workspace feedback configs pro 3 human keys, pokud chybí.

    Bez tohoto kroku UI ukazuje v 'Add feedback' dropdownu jen feedback configs,
    které už dřív někdo použil — naše rubric_items by visely v prázdnu.
    """
    existing = {fc.feedback_key for fc in client.list_feedback_configs()}
    config = {"type": "continuous", "min": 0, "max": 1}
    for item in RUBRIC_ITEMS:
        key = item["feedback_key"]
        if key in existing:
            continue
        try:
            client.create_feedback_config(
                key,
                feedback_config=config,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [feedback config '{key}' skipped: {exc}]")


def _get_or_create_queue(client: Client, name: str) -> tuple[UUID, bool]:
    _ensure_feedback_configs(client)
    existing = next(iter(client.list_annotation_queues(name=name)), None)
    if existing is not None:
        # Refresh rubric — pokud queue vznikla dřív, než byly configs ve workspace,
        # její rubric_items odkazovaly na neexistující klíče. Tohle to napraví.
        client.update_annotation_queue(
            existing.id,
            rubric_instructions=RUBRIC_INSTRUCTIONS,
            rubric_items=RUBRIC_ITEMS,  # type: ignore[arg-type]
        )
        return existing.id, False
    queue = client.create_annotation_queue(
        name=name,
        description="Cílený human review pro kosik-workshop. Plněno triage policy.",
        rubric_instructions=RUBRIC_INSTRUCTIONS,
        rubric_items=RUBRIC_ITEMS,  # type: ignore[arg-type]
    )
    return queue.id, True


def _pending_run_ids_in_queue(client: Client, queue_id: UUID) -> set[UUID]:
    """Walk the queue index — vrací jen pending (Needs Review) runs.

    Dokončené (Done) runs už v indexu nejsou; ty řešíme přes
    `_already_reviewed_run_ids` níž.
    """
    ids: set[UUID] = set()
    idx = 0
    while True:
        try:
            run = client.get_run_from_annotation_queue(queue_id, index=idx)
        except Exception:  # noqa: BLE001 — queue exhausted or index OOB
            break
        ids.add(run.id)
        idx += 1
        if idx > 5000:
            break
    return ids


HUMAN_FEEDBACK_KEYS = ["correct_tools", "helpful", "safe"]


def _already_reviewed_run_ids(client: Client, run_ids: list[UUID]) -> set[UUID]:
    """Vrací run IDs, kde už existuje alespoň jeden human feedback klíč.

    Dokončené runs zmizí z queue indexu (UI tlačítko Done), ale jejich
    feedback zůstane na runu. Tohle je spolehlivější dedup než queue index.
    """
    if not run_ids:
        return set()
    reviewed: set[UUID] = set()
    for fb in client.list_feedback(run_ids=run_ids, feedback_key=HUMAN_FEEDBACK_KEYS):
        if fb.score is not None:
            reviewed.add(fb.run_id)
    return reviewed


# LangSmith API caps `limit` per page at 100; iterátor stejně paginuje sám,
# takže `limit` slouží jen jako velikost první page.
API_PAGE_MAX = 100


def _list_root_runs(client: Client, project: str, since: datetime, limit: int = 100) -> list:
    return list(
        client.list_runs(
            project_name=project,
            start_time=since,
            is_root=True,
            limit=min(limit, API_PAGE_MAX),
        )
    )


def _filter_thumbs_down(client: Client, project: str, since: datetime, limit: int) -> list:
    runs = list(
        client.list_runs(
            project_name=project,
            start_time=since,
            is_root=True,
            filter='eq(feedback_key, "user_thumbs")',
            limit=API_PAGE_MAX,
        )
    )
    flagged = []
    if not runs:
        return flagged
    fb_by_run: dict[UUID, list] = {}
    for fb in client.list_feedback(run_ids=[r.id for r in runs], feedback_key=["user_thumbs"]):
        fb_by_run.setdefault(fb.run_id, []).append(fb)
    for run in runs:
        scores = [fb.score for fb in fb_by_run.get(run.id, []) if fb.score is not None]
        if scores and min(scores) == 0:
            flagged.append(run)
            if len(flagged) >= limit:
                break
    return flagged


def _filter_eval_flagged(client: Client, project: str, since: datetime, limit: int) -> list:
    runs = _list_root_runs(client, project, since, limit=API_PAGE_MAX)
    if not runs:
        return []
    flagged = []
    fb_by_run: dict[UUID, list] = {}
    for fb in client.list_feedback(
        run_ids=[r.id for r in runs],
        feedback_key=["online.heuristic_quality", "hallucination"],
    ):
        fb_by_run.setdefault(fb.run_id, []).append(fb)
    for run in runs:
        for fb in fb_by_run.get(run.id, []):
            if fb.score is None:
                continue
            if fb.key == "online.heuristic_quality" and fb.score == 0:
                flagged.append(run)
                break
            if fb.key == "hallucination" and fb.score == 1:
                flagged.append(run)
                break
        if len(flagged) >= limit:
            break
    return flagged


def _filter_random(client: Client, project: str, since: datetime, pct: float, limit: int) -> list:
    runs = _list_root_runs(client, project, since, limit=API_PAGE_MAX)
    if not runs:
        return []
    k = max(1, min(limit, int(round(len(runs) * pct / 100))))
    rng = random.Random(42)  # deterministic for workshop reproducibility
    return rng.sample(runs, k=min(k, len(runs)))


def main() -> int:
    args = parse_args()
    load_env()

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    for r in rules:
        if r not in ALL_RULES:
            print(f"unknown rule: {r}. Available: {ALL_RULES}", file=sys.stderr)
            return 1

    project = args.project or os.getenv("LANGSMITH_PROJECT", "kosik-workshop")
    since = datetime.now(UTC) - timedelta(hours=args.hours)
    client = Client()

    print(f"Project:  {project}")
    print(f"Window:   last {args.hours}h (since {since.isoformat()})")
    print(f"Rules:    {rules}")
    print(f"Queue:    {args.queue}")
    print()

    queue_id, created = _get_or_create_queue(client, args.queue)
    if created:
        print(f"✓ Queue created: {args.queue} (id={queue_id})")
    else:
        print(f"✓ Queue exists: {args.queue} (id={queue_id})")

    pending_in_queue = _pending_run_ids_in_queue(client, queue_id)
    print(f"  pending in queue: {len(pending_in_queue)}")

    per_rule: dict[str, list] = {}
    if "thumbs_down" in rules:
        per_rule["thumbs_down"] = _filter_thumbs_down(client, project, since, args.limit)
    if "eval_flagged" in rules:
        per_rule["eval_flagged"] = _filter_eval_flagged(client, project, since, args.limit)
    if "random" in rules:
        per_rule["random"] = _filter_random(client, project, since, args.random_pct, args.limit)

    candidate_ids = [run.id for runs in per_rule.values() for run in runs]
    already_reviewed = _already_reviewed_run_ids(client, candidate_ids)
    print(f"  already reviewed: {len(already_reviewed)}\n")

    skip_ids = pending_in_queue | already_reviewed
    seen: set[UUID] = set()
    union_runs: list = []
    skipped_per_rule: dict[str, int] = {r: 0 for r in rules}
    for rule in rules:
        for run in per_rule.get(rule, []):
            if run.id in skip_ids:
                skipped_per_rule[rule] += 1
                continue
            if run.id in seen:
                continue
            seen.add(run.id)
            union_runs.append(run)

    print(f"{'rule':<14} {'matched':>8} {'skip':>6} {'new':>5}")
    print("-" * 42)
    for rule in rules:
        matched = len(per_rule.get(rule, []))
        skipped = skipped_per_rule[rule]
        new = sum(1 for r in per_rule.get(rule, []) if r.id in seen)
        print(f"{rule:<14} {matched:>8} {skipped:>6} {new:>5}")
    print(f"{'TOTAL (dedup)':<14} {'':>8} {'':>6} {len(union_runs):>5}\n")

    if not union_runs:
        print("Nothing to push. Queue unchanged.")
        return 0

    if args.dry_run:
        print(f"DRY RUN — would push {len(union_runs)} runs.")
        for r in union_runs[:10]:
            tags = list(r.tags or [])
            print(f"  {str(r.id)[:8]}  {r.start_time:%Y-%m-%d %H:%M}  tags={tags[:3]}")
        return 0

    client.add_runs_to_annotation_queue(queue_id, run_ids=[r.id for r in union_runs])
    print(f"✓ Pushed {len(union_runs)} runs to {args.queue}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
