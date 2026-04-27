"""Spočítej agreement mezi automated evaluatory a human anotacemi.

Pro každý run v annotation queue, který má kompletní human feedback (3 booleany),
porovná human score s odpovídajícím automated/online feedback klíčem.
Vytiskne tabulku s agreement % a pár vzorků kde se rozcházejí.

Mapování (automated → human):
    online.heuristic_quality  → helpful
    user_thumbs               → helpful
    hallucination (1=BAD)     → NOT safe   (lidi: safe=False, judge: hallucination=1)

Usage:
    uv run python scripts/eval_human_agreement.py --queue kosik-human-review
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any
from uuid import UUID

from langsmith import Client

from kosik_workshop.config import load_env

QUEUE_NAME_DEFAULT = "kosik-human-review"
HUMAN_KEYS = ("correct_tools", "helpful", "safe")

# (automated_key, human_key, invert) — invert=True když automated 1 znamená BAD
COMPARISONS: list[tuple[str, str, bool]] = [
    ("online.heuristic_quality", "helpful", False),
    ("user_thumbs", "helpful", False),
    ("hallucination", "safe", True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default=QUEUE_NAME_DEFAULT)
    p.add_argument(
        "--show-disagreements", type=int, default=3, help="Sample N disagreements per row."
    )
    return p.parse_args()


def _walk_queue_run_ids(client: Client, queue_id: UUID) -> list[UUID]:
    ids: list[UUID] = []
    idx = 0
    while True:
        try:
            run = client.get_run_from_annotation_queue(queue_id, index=idx)
        except Exception:  # noqa: BLE001
            break
        ids.append(run.id)
        idx += 1
        if idx > 5000:
            break
    return ids


def _load_feedback(client: Client, run_ids: list[UUID]) -> dict[UUID, dict[str, float]]:
    """For each run, take the last score for each feedback key (handles re-annotation)."""
    out: dict[UUID, dict[str, float]] = defaultdict(dict)
    if not run_ids:
        return out
    for fb in client.list_feedback(run_ids=run_ids):
        if fb.score is None:
            continue
        out[fb.run_id][fb.key] = float(fb.score)
    return out


def main() -> int:
    args = parse_args()
    load_env()

    client = Client()
    queue = next(iter(client.list_annotation_queues(name=args.queue)), None)
    if queue is None:
        print(f"error: queue '{args.queue}' not found", file=sys.stderr)
        return 1

    run_ids = _walk_queue_run_ids(client, queue.id)
    print(f"Queue:  {args.queue} ({len(run_ids)} runs)\n")

    fb = _load_feedback(client, run_ids)

    # Filter to runs with all 3 human booleans set.
    annotated_ids = [rid for rid in run_ids if all(k in fb[rid] for k in HUMAN_KEYS)]
    print(f"Fully annotated: {len(annotated_ids)} / {len(run_ids)}\n")

    if not annotated_ids:
        print("Žádné kompletně anotované runs — nejdřív projdi queue v UI.")
        return 0

    print(f"{'automated':<28} {'human':<14} {'agree':>6} {'n':>4}  {'note':<28}")
    print("-" * 86)

    rows: list[dict[str, Any]] = []
    for auto_key, human_key, invert in COMPARISONS:
        n_agree = 0
        n_total = 0
        disagreements: list[tuple[UUID, float, float]] = []
        for rid in annotated_ids:
            if auto_key not in fb[rid]:
                continue
            auto = fb[rid][auto_key]
            # Pravidlo: judge říká hallucinated=1 ⇒ unsafe; agree iff human safe=0
            normalized_auto = 1 - auto if invert else auto
            human = fb[rid][human_key]
            n_total += 1
            if int(round(normalized_auto)) == int(round(human)):
                n_agree += 1
            else:
                disagreements.append((rid, auto, human))

        if n_total == 0:
            print(f"{auto_key:<28} {human_key:<14} {'—':>6} {0:>4}  no overlap")
            continue
        pct = 100.0 * n_agree / n_total
        rows.append(
            {
                "auto": auto_key,
                "human": human_key,
                "agree_pct": pct,
                "n": n_total,
                "disagreements": disagreements,
            }
        )
        note = "judge přísnější" if invert and pct < 70 else ""
        print(f"{auto_key:<28} {human_key:<14} {pct:>5.1f}% {n_total:>4}  {note:<28}")

    if args.show_disagreements > 0:
        print()
        for row in rows:
            if not row["disagreements"]:
                continue
            print(f"\n  Disagreement sample for {row['auto']} vs {row['human']}:")
            for rid, auto, human in row["disagreements"][: args.show_disagreements]:
                print(f"    run {str(rid)[:8]}  auto={auto:.0f}  human={human:.0f}")

    print(
        "\nLekce: kde je agreement < 70 %, judge měří jiný koncept než lidi. "
        "Iteruj judge prompt nebo přidej regression cases do datasetu."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
