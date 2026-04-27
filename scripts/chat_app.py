"""Mini CLI chat surface se sběrem end-user feedbacku.

Multi-turn konverzace v jednom threadu (agent si pamatuje předchozí výměny).
Hodnocení 👍/👎 se posílá *na poslední odpověď* explicitními příkazy:

    /+               → 👍 na poslední odpověď
    /+ super odpověď → 👍 + komentář
    /-               → 👎 na poslední odpověď
    /- nezmínil cenu → 👎 + komentář

Ostatní příkazy:
    /quit, /q, /exit → konec
    /new, /reset     → nový thread (forget context)
    /help            → tato nápověda

Cokoli jiného = další zpráva v konverzaci.

Traces jsou tagované `surface:chat_app`, takže se v LangSmith UI dají
odlišit od synthetic traffic. `user_thumbs` feedback je ten signál,
který v `seed_annotation_queue.py` spouští pravidlo `thumbs_down`.

Usage:
    uv run python scripts/chat_app.py --user user-petr
"""

from __future__ import annotations

import argparse
import sys
import uuid
from uuid import UUID

from langchain_core.messages import HumanMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langsmith import Client

from kosik_workshop.agent import build_agent
from kosik_workshop.config import load_env
from kosik_workshop.simulation.demo_users import DEMO_USERS, by_id

HELP = (
    "Příkazy:\n"
    "  /+ [komentář]   👍 na poslední odpověď\n"
    "  /- [komentář]   👎 na poslední odpověď\n"
    "  /new            nový thread (smazat kontext)\n"
    "  /quit /q /exit  konec\n"
    "  /help           tato nápověda\n"
    "  cokoli jiného   pokračuj v konverzaci\n"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--user",
        default="user-petr",
        help=f"User ID. Available: {', '.join(u.user_id for u in DEMO_USERS)}",
    )
    return p.parse_args()


def _send_thumbs(client: Client, run_id: UUID | None, score: int, comment: str) -> None:
    if run_id is None:
        print("  [feedback skipped: žádná předchozí odpověď]")
        return
    try:
        client.create_feedback(
            run_id=run_id,
            key="user_thumbs",
            score=score,
            comment=comment or None,
        )
        mark = "👍" if score == 1 else "👎"
        print(f"  ({mark} feedback uložen na předchozí odpověď)\n")
    except Exception as exc:  # noqa: BLE001
        print(f"  [feedback skipped: {exc}]")


def main() -> int:
    args = parse_args()
    load_env()

    user = by_id(args.user)
    agent = build_agent()
    client = Client()

    thread_id = str(uuid.uuid4())
    last_run_id: UUID | None = None  # ID poslední root-run odpovědi pro retroaktivní rating

    print(f"Košík chat — user={user.name} ({user.user_id})")
    print(f"Thread: {thread_id[:8]} (agent si pamatuje celou konverzaci)")
    print("Napiš /help pro příkazy. Konec: /quit\n")

    while True:
        try:
            line = input(f"{user.name} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # --- Commands ---
        if line in ("/quit", "/q", "/exit"):
            break
        if line in ("/new", "/reset"):
            thread_id = str(uuid.uuid4())
            last_run_id = None
            print(f"  (nový thread: {thread_id[:8]})\n")
            continue
        if line in ("/help", "/?"):
            print(HELP)
            continue
        if line.startswith("/+"):
            _send_thumbs(client, last_run_id, score=1, comment=line[2:].strip())
            continue
        if line.startswith("/-"):
            _send_thumbs(client, last_run_id, score=0, comment=line[2:].strip())
            continue
        if line.startswith("/"):
            print(f"  (neznámý příkaz: {line!r}; /help)")
            continue

        # --- Conversation turn ---
        collector = RunCollectorCallbackHandler()
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "session_id": thread_id,
                "user_id": user.user_id,
                "user_tier": user.tier,
                "simulated": False,
            },
            "tags": [
                "surface:chat_app",
                f"tier:{user.tier}",
            ],
            "run_name": "kosik-agent:chat",
            "callbacks": [collector],
        }

        try:
            result = agent.invoke({"messages": [HumanMessage(content=line)]}, config=config)
            answer = result["messages"][-1].content
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as exc:  # noqa: BLE001
            answer = f"[error] {exc}"

        print(f"\nasistent > {answer}\n")
        last_run_id = collector.traced_runs[0].id if collector.traced_runs else None

    print("Konec. Traces najdeš v LangSmith pod tagem `surface:chat_app`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
