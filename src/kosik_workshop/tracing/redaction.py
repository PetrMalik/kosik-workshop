"""PII redaction for LangSmith traces.

Built on the canonical `langsmith.anonymizer.create_anonymizer` API. The
anonymizer is wired to `Client(anonymizer=...)` — LangSmith serializes the
trace payload (including Pydantic objects like `HumanMessage`) to plain JSON
BEFORE applying the regex replacements, so message content reaches the rules
as a string and is properly scrubbed.

Why not `hide_inputs`/`hide_outputs` callbacks? Those receive the raw input
dict where Pydantic models are still objects, not dicts — a hand-written
walker would silently skip them. The `anonymizer` path avoids that pitfall
by serializing first.

Integration: call `install_redaction()` ONCE at app startup, BEFORE any
LangChain tracer is created. Patches `Client.__init__` and resets the cached
client so subsequent traces flow through the anonymizer.
"""

from __future__ import annotations

import re

from langsmith.anonymizer import StringNodeRule, create_anonymizer

# Czech-aware patterns. Conservative — false positives are better than leaks.
_RULES: list[StringNodeRule] = [
    {
        "pattern": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "replace": "[EMAIL]",
    },
    {
        # Czech phone (with or without +420, optional separators)
        "pattern": re.compile(r"(?:\+?420[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}\b"),
        "replace": "[PHONE]",
    },
    {
        # Czech birth number (rodné číslo): 6 / 3–4 digits
        "pattern": re.compile(r"\b\d{6}/\d{3,4}\b"),
        "replace": "[RC]",
    },
    {
        # IBAN (CZ) — must come BEFORE CARD (digit runs would otherwise match)
        "pattern": re.compile(r"\bCZ\d{2}(?:[\s]?\d{4}){5}\b"),
        "replace": "[IBAN]",
    },
    {
        # Credit card-ish: 13–19 digits with optional separators
        "pattern": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "replace": "[CARD]",
    },
]

anonymizer = create_anonymizer(_RULES)


def scrub_text(text: str) -> str:
    """Apply the same regex rules to a single string. For unit tests / preview."""
    for rule in _RULES:
        text = rule["pattern"].sub(rule["replace"] or "[redacted]", text)
    return text


_INSTALLED = False


def install_redaction() -> None:
    """Patch `langsmith.Client.__init__` to inject the anonymizer.

    Idempotent. Affects ALL future Client instances and resets the cached
    singleton used by LangChain's tracer. Call BEFORE any agent.invoke() so
    the tracer's first Client is already patched.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from langsmith import Client

    original_init = Client.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("anonymizer", anonymizer)
        original_init(self, *args, **kwargs)

    Client.__init__ = patched_init  # type: ignore[method-assign]

    # Reset the cached singleton — without this, any pre-install Client stays
    # in the cache and bypasses the anonymizer.
    import langsmith.run_trees as _rt

    _rt._CLIENT = None

    _INSTALLED = True


def uninstall_redaction() -> None:
    """Reset the install flag. Existing patched instances keep their hooks —
    use a fresh kernel for a truly unhooked baseline."""
    global _INSTALLED
    _INSTALLED = False
