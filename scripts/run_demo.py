"""Workshop demo runner.

Usage:
    uv run python scripts/run_demo.py --scenario baseline
    uv run python scripts/run_demo.py --scenario recovery

Each scenario tags traces with `scenario:<name>` so you can filter in LangSmith.

Recommended live-demo flow:
    1. baseline  (~3 min, 30 traces — populate dashboard)
    2. recovery  (~1 min, 10 traces — lighter traffic shape)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse

from kosik_workshop.config import load_env
from kosik_workshop.simulation import run_scenario


def _langsmith_url(scenario: str) -> str:
    """Compose a LangSmith UI link pre-filtered to this scenario's traces."""
    project = os.getenv("LANGSMITH_PROJECT", "kosik-workshop")
    base = os.getenv("LANGSMITH_ENDPOINT", "https://eu.smith.langchain.com").rstrip("/")
    base = base.replace("api.", "")  # api.smith.langchain.com → smith.langchain.com
    query = urllib.parse.quote(f'has(tags, "scenario:{scenario}")')
    return f"{base}/projects/p?filter={query}&projectName={project}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=["baseline", "recovery"],
        required=True,
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query progress output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()

    print(f"\n=== Scenario: {args.scenario} ===\n")
    result = run_scenario(args.scenario, print_progress=not args.quiet)

    pass_rate = result.feedback_pass / result.run_count if result.run_count else 0.0
    print(
        f"\n--- Summary ---\n"
        f"Scenario:      {result.scenario}\n"
        f"Runs:          {result.run_count}\n"
        f"Feedback pass: {result.feedback_pass} ({pass_rate:.0%})\n"
        f"Feedback fail: {result.feedback_fail}\n"
        f"Duration:      {result.duration_s:.1f}s\n"
        f"Sample threads (first 3):\n"
    )
    for tid in result.sample_thread_ids:
        print(f"  - {tid}")

    print("\nLangSmith filter for this scenario:")
    print(f"  Tags filter: scenario:{args.scenario}")
    print(f"  Or open:     {_langsmith_url(args.scenario)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
