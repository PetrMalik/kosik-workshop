"""Bootstrap the kosik-eval-golden dataset in LangSmith.

Idempotent: re-running only adds examples that aren't already in the dataset
(matched by `inputs.question`). Use `--replace` to wipe and re-upload.
"""

from __future__ import annotations

import argparse
import sys

from kosik_workshop.config import load_env
from kosik_workshop.evals import (
    GOLDEN_EXAMPLES,
    RETRIEVAL_EXAMPLES,
    seed_dataset,
    seed_retrieval_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing examples in the dataset and re-upload from code.",
    )
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    e2e_id = seed_dataset(replace=args.replace)
    print(f"E2E dataset id:       {e2e_id}  ({len(GOLDEN_EXAMPLES)} examples)")
    rag_id = seed_retrieval_dataset(replace=args.replace)
    print(f"Retrieval dataset id: {rag_id}  ({len(RETRIEVAL_EXAMPLES)} examples)")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
