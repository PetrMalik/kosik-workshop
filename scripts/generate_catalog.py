from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

from kosik_workshop.catalog.generator import (
    _make_client,
    generate_batch,
    generate_catalog,
    generate_fill_gap,
)
from kosik_workshop.catalog.schema import Product
from kosik_workshop.catalog.taxonomy import TAXONOMY
from kosik_workshop.catalog.validate import validate_all
from kosik_workshop.config import load_env

log = logging.getLogger("generate_catalog")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = ROOT / "data" / "products.json"
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma"


def _dump_json(products: list[Product], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_products = sorted(products, key=lambda p: p.id)
    payload = [p.model_dump(mode="json") for p in sorted_products]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _per_category_counts(products: list[Product]) -> Counter[str]:
    return Counter(p.category for p in products)


def _compute_gaps(accepted: list[Product]) -> dict[str, int]:
    got = _per_category_counts(accepted)
    return {cat: meta["quota"] - got.get(cat, 0) for cat, meta in TAXONOMY.items()}


def _print_report(accepted: list[Product], rejected: list[tuple[Product, str]]) -> None:
    counts = _per_category_counts(accepted)
    print("\n=== Report ===")
    print(f"Accepted: {len(accepted)}")
    for cat, meta in TAXONOMY.items():
        got = counts.get(cat, 0)
        mark = "✓" if got >= meta["quota"] else "!"
        print(f"  {mark} {cat:35s} {got:3d}/{meta['quota']}")
    print(f"Rejected: {len(rejected)}")
    reason_counts: Counter[str] = Counter(r for _, r in rejected)
    for reason, n in reason_counts.most_common():
        print(f"  · {reason}: {n}")


async def _run_full(args: argparse.Namespace) -> list[Product]:
    taxonomy_subset = TAXONOMY
    if args.only:
        if args.only not in TAXONOMY:
            raise SystemExit(f"Unknown category: {args.only}")
        taxonomy_subset = {args.only: TAXONOMY[args.only]}

    log.info("Generating catalog…")
    raw = await generate_catalog(taxonomy=taxonomy_subset)
    log.info("Raw count: %d", len(raw))

    accepted, rejected = validate_all(raw)
    log.info("After validation: accepted=%d rejected=%d", len(accepted), len(rejected))

    gaps = {c: n for c, n in _compute_gaps(accepted).items() if c in taxonomy_subset and n > 0}
    if gaps:
        log.info("Fill-gap pass for: %s", gaps)
        extra = await generate_fill_gap(gaps)
        extra_accepted, extra_rejected = validate_all(extra)
        accepted += extra_accepted
        rejected += extra_rejected
        accepted, dup_rej = _dedupe_with_rejected(accepted)
        rejected += [(d, "duplicate across batches") for d in dup_rej]

    _print_report(accepted, rejected)
    return accepted


def _dedupe_with_rejected(products: list[Product]) -> tuple[list[Product], list[Product]]:
    from kosik_workshop.catalog.validate import dedupe

    return dedupe(products)


async def _run_dry() -> None:
    log.info("Dry run: 1 batch × 5 from '%s'", next(iter(TAXONOMY)))
    client = _make_client()
    cat = next(iter(TAXONOMY))
    products = await generate_batch(client, cat, TAXONOMY[cat], 5)
    for p in products:
        print(f"- {p.name}  [{p.category}/{p.subcategory}]  {p.price_czk} Kč / {p.unit}")
    print(f"\n({len(products)} generated)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_env()

    parser = argparse.ArgumentParser(description="Generate Košík.cz product catalog seed")
    parser.add_argument("--dry", action="store_true", help="Single batch, no writes")
    parser.add_argument("--only", type=str, help="Only generate this category")
    parser.add_argument("--no-index", action="store_true", help="Skip Chroma indexing")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    args = parser.parse_args()

    if args.dry:
        asyncio.run(_run_dry())
        return

    products = asyncio.run(_run_full(args))
    if not products:
        raise SystemExit("No products generated")

    _dump_json(products, args.json_out)
    log.info("Wrote %d products → %s", len(products), args.json_out)

    if not args.no_index:
        from kosik_workshop.catalog.store import build_chroma_index

        build_chroma_index(products, persist_dir=args.chroma_dir)
        log.info("Indexed into Chroma → %s", args.chroma_dir)


if __name__ == "__main__":
    main()
