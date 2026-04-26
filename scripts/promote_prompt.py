"""Promote the Košík prompt to the prod tag.

Pulls an existing commit (either a specific hash or the current `dev`) and
re-pushes it with `commit_tags=["prod"]`. This moves the prod tag to identical
content. Technically a new commit is created that is content-identical with dev;
this is visible in the LangSmith UI.

This is sufficient for the workshop. If you want a "tag move" without a new
commit, do it manually in the LangSmith UI (Commits → ⋯ → Edit tags).
"""

from __future__ import annotations

import argparse
import sys

from langsmith import Client

from kosik_workshop.config import load_env
from kosik_workshop.prompts.kosik_assistant import PROMPT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--commit",
        help="Specific commit hash to promote to prod.",
    )
    source.add_argument(
        "--from-tag",
        default="dev",
        help="Source tag to pull from (default: dev).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Description of the promote commit (e.g. 'promote v1.2 to prod').",
    )
    return parser.parse_args()


def _current_prod_commit(client: Client) -> str | None:
    try:
        prompt = client.pull_prompt(f"{PROMPT_NAME}:prod", include_model=False)
    except Exception:
        return None
    return getattr(prompt, "metadata", {}).get("lc_hub_commit_hash") if prompt else None


def main() -> int:
    load_env()
    args = parse_args()
    client = Client()

    source_ref = args.commit if args.commit else args.from_tag
    source_identifier = f"{PROMPT_NAME}:{source_ref}"
    print(f"Source: {source_identifier}")

    current_prod = _current_prod_commit(client)
    if current_prod:
        print(f"Current prod commit: {current_prod}")
    else:
        print("Current prod: <none> (prod tag does not exist yet)")

    if not args.yes:
        answer = input("Continue with promote to prod? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    prompt = client.pull_prompt(source_identifier)
    description = args.description or f"promote {source_ref} → prod"
    url = client.push_prompt(
        PROMPT_NAME,
        object=prompt,
        commit_description=description,
        commit_tags=["prod"],
    )
    print(f"Promoted → {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
