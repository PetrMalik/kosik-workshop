"""Diagnose what LangSmith's tracer hands to hide_inputs/hide_outputs hooks.

Runs ONE agent invoke with PII and prints every hook call to stderr. Useful
when redaction looks correct (hooks attached) but PII still appears in UI.
"""

from __future__ import annotations

import sys
import uuid

from kosik_workshop.config import load_env
from kosik_workshop.tracing import install_redaction


def main() -> int:
    load_env()
    install_redaction()

    # Imports AFTER install_redaction so any module-level Client() runs patched.
    from langchain_core.messages import HumanMessage

    from kosik_workshop.agent import build_agent

    agent = build_agent()
    thread_id = str(uuid.uuid4())
    print(f"thread_id: {thread_id}", file=sys.stderr)

    pii = (
        "Hledám veganské mléko do 50 Kč. Pošlete potvrzení na "
        "info@malikpetr.cz nebo zavolejte na +420 777 123 456."
    )
    agent.invoke(
        {"messages": [HumanMessage(content=pii)]},
        config={
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "session_id": thread_id,
                "user_id": "diag-redaction",
            },
            "tags": ["surface:diag", "feature:redaction-diag"],
        },
    )

    # Force background tracer to flush so hooks fire before exit.
    import langsmith.run_trees as rt

    client = rt.get_cached_client()
    if hasattr(client, "flush"):
        client.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
