"""Diagnostika: tahá z LangSmith feedback per scenario a tiskne tabulku.

Usage:
    uv run python scripts/check_scenarios.py
    uv run python scripts/check_scenarios.py --hours 2
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from langsmith import Client

from kosik_workshop.config import load_env

SCENARIOS = ("baseline", "recovery")
FEEDBACK_KEYS = ("hallucination", "online.heuristic_quality")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=6, help="Look back this many hours.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env()

    project = os.getenv("LANGSMITH_PROJECT", "kosik-workshop")
    client = Client()
    since = datetime.now(UTC) - timedelta(hours=args.hours)

    print(f"\nProject: {project}")
    print(f"Window:  last {args.hours}h (since {since.isoformat()})\n")

    # Pull only top-level runs that are simulated (per-scenario tag).
    per_scenario_runs: dict[str, list] = defaultdict(list)
    for scenario in SCENARIOS:
        runs = list(
            client.list_runs(
                project_name=project,
                start_time=since,
                is_root=True,
                filter=f'and(has(tags, "simulated"), has(tags, "scenario:{scenario}"))',
            )
        )
        per_scenario_runs[scenario] = runs

    # For each scenario, collect feedback by key.
    rows = []
    for scenario in SCENARIOS:
        runs = per_scenario_runs[scenario]
        run_ids = [r.id for r in runs]
        n_runs = len(run_ids)

        scores_by_key: dict[str, list[float]] = defaultdict(list)
        if run_ids:
            for fb in client.list_feedback(run_ids=run_ids):
                if fb.key in FEEDBACK_KEYS and fb.score is not None:
                    scores_by_key[fb.key].append(float(fb.score))

        row = {"scenario": scenario, "n_runs": n_runs}
        for key in FEEDBACK_KEYS:
            scores = scores_by_key.get(key, [])
            if scores:
                row[key] = (
                    f"n={len(scores):2d} mean={sum(scores) / len(scores):.2f} "
                    f"ones={int(sum(scores))}/{len(scores)}"
                )
            else:
                row[key] = "NO DATA"
        rows.append(row)

    # Print table.
    header = (
        f"{'scenario':<12} {'runs':>5}  "
        f"{'hallucination (1=BAD)':<32}  {'heuristic_quality (1=GOOD)':<32}"
    )
    print(header)
    print("-" * 90)
    for row in rows:
        print(
            f"{row['scenario']:<12} {row['n_runs']:>5}  "
            f"{row['hallucination']:<32}  {row['online.heuristic_quality']:<32}"
        )

    print("\nExpected pattern (optimized agent, both scenarios):")
    print("  hallucination low  (~0.1–0.2)")
    print("  heuristic high     (~0.7–0.9)")
    print()

    # Sanity check on judge coverage.
    for scenario in SCENARIOS:
        n_runs = len(per_scenario_runs[scenario])
        for key in FEEDBACK_KEYS:
            scored = (
                sum(
                    1
                    for fb in client.list_feedback(
                        run_ids=[r.id for r in per_scenario_runs[scenario]]
                    )
                    if fb.key == key and fb.score is not None
                )
                if n_runs
                else 0
            )
            if n_runs and scored < n_runs and key == "hallucination":
                missing = n_runs - scored
                print(
                    f"  ⚠  {scenario}: hallucination scored on {scored}/{n_runs} runs "
                    f"({missing} pending — judge may still be processing)"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
