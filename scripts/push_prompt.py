"""Push the Košík system prompt to LangSmith Prompt Hub from code (bootstrap / rollback).

⚠️ The canonical source for prompts is the **LangSmith Playground**, not this script.
Use this script only:
  1. For the initial seed (prompt does not yet exist in the Hub).
  2. For manual rollback / rewrite from git (overwrites Playground work!).

The everyday workflow is: edit in the Playground → commit with tag `dev` → promote.
"""

from __future__ import annotations

import argparse
import sys

from langsmith import Client

from kosik_workshop.config import load_env
from kosik_workshop.prompts.kosik_assistant import PROMPT_NAME, build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Add an extra commit tag (repeatable). Tag 'dev' is always added.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Commit description visible in the LangSmith UI.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Push the prompt as public (default: private).",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print what would be pushed without making a network call.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (overwrites Playground work).",
    )
    return parser.parse_args()


def _prompt_exists(client: Client, name: str) -> bool:
    try:
        client.pull_prompt(name)
    except Exception:
        return False
    return True


def main() -> int:
    load_env()
    args = parse_args()

    prompt = build_prompt()
    commit_tags = ["dev"]
    if args.tag:
        commit_tags.extend(t for t in args.tag if t not in commit_tags)

    if args.dry:
        print(f"[dry] would push '{PROMPT_NAME}' with commit_tags={commit_tags}")
        print(f"[dry] description: {args.description!r}")
        print("[dry] prompt messages:")
        for msg in prompt.format_messages(input="<example>", agent_scratchpad=[]):
            preview = str(msg.content)[:120].replace("\n", " ")
            print(f"  - {msg.type}: {preview}...")
        return 0

    client = Client()

    if _prompt_exists(client, PROMPT_NAME) and not args.yes:
        print(
            f"⚠️  Prompt '{PROMPT_NAME}' already exists in the Hub. A push from code "
            "will create a new commit and any unpublished Playground work may be lost."
        )
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    url = client.push_prompt(
        PROMPT_NAME,
        object=prompt,
        commit_description=args.description,
        commit_tags=commit_tags,
        is_public=True if args.public else None,
    )
    print(f"Pushed {PROMPT_NAME} → {url}")
    print(f"Commit tags applied: {commit_tags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
