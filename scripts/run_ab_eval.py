"""A/B prompt evaluation runner.

Spustí offline evaluaci nad stejným datasetem pro N prompt variant a každou
zaregistruje jako samostatný LangSmith experiment. Společný `ab_group` v
metadatech slouží jako "anchor" — v UI Experiments tab podle něj filtruješ
spárované experimenty.

Usage:
    # Default: porovná všechny tři varianty pod skupinou "default-ab".
    uv run python scripts/run_ab_eval.py

    # Vybrat konkrétní varianty + custom group name.
    uv run python scripts/run_ab_eval.py \\
        --variants v1_baseline,v2_strict_citations \\
        --group citations-test

    # List dostupných variant.
    uv run python scripts/run_ab_eval.py --list

Po doběhnutí otevři https://eu.smith.langchain.com → Datasets & experiments →
kosik-eval-golden → tab Experiments. Označ checkboxy u relevantních experimentů
(filter `metadata.ab_group is "<group>"`) a klikni "Compare" pro side-by-side.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime

from kosik_workshop.agent import build_agent
from kosik_workshop.config import load_env
from kosik_workshop.evals.runner import run_evaluation
from kosik_workshop.prompts.variants import PROMPT_VARIANTS, get_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--variants",
        default="v1_baseline,v2_strict_citations,v3_loud_allergens",
        help="Comma-separated variant names (see --list).",
    )
    p.add_argument(
        "--group",
        default=None,
        help="ab_group label written to metadata. Default: timestamped 'ab-<ts>'.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model used for all variants (kept identical to isolate prompt effect).",
    )
    p.add_argument("--list", action="store_true", help="List variants and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available variants:")
        for name in PROMPT_VARIANTS:
            preview = PROMPT_VARIANTS[name].splitlines()[0][:80]
            print(f"  {name:<24} {preview}")
        return 0

    load_env()
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variant_names:
        if v not in PROMPT_VARIANTS:
            print(f"unknown variant: {v}", file=sys.stderr)
            print(f"available: {list(PROMPT_VARIANTS)}", file=sys.stderr)
            return 1

    group = args.group or f"ab-{datetime.now(UTC):%Y%m%d-%H%M%S}"

    print("\n=== A/B evaluation ===")
    print(f"Group:    {group}")
    print(f"Model:    {args.model}")
    print(f"Variants: {variant_names}\n")

    results = []
    for variant in variant_names:
        print(f"--- Running variant: {variant} ---")
        agent = build_agent(model=args.model, system_text=get_variant(variant))
        t0 = time.monotonic()
        result = run_evaluation(
            agent,
            experiment_prefix=f"{variant}__{group}",
            metadata={
                "ab_group": group,
                "prompt_variant": variant,
                "model": args.model,
            },
            description=f"A/B test '{group}' — variant {variant}",
        )
        elapsed = time.monotonic() - t0
        print(f"--- Done in {elapsed:.1f}s ---\n")
        results.append((variant, result))

    print("=== Summary ===")
    print(f"Group: {group}")
    print(f"In LangSmith UI: filter Experiments by metadata.ab_group = '{group}'")
    print("Or look for experiments named:\n")
    for variant, _ in results:
        print(f"  {variant}__{group}-...")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
