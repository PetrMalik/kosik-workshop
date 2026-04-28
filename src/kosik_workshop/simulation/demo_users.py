"""Five hand-curated synthetic users for workshop demo.

Small enough to fit on one slide, varied enough to populate per-user filters
in LangSmith dashboards. Each user has a stable id (used as `metadata.user_id`)
so dashboards can filter per-person across scenarios.

Tools read per-run allergens from `RunnableConfig` (`configurable.user_allergens`),
so parallel invocations are isolated. The simulator (`simulation/runner.py`)
threads each user's allergens into the config on every `agent.invoke`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoUser:
    user_id: str
    name: str
    tier: str  # "free" / "premium"
    allergens: tuple[str, ...]
    notes: str


DEMO_USERS: list[DemoUser] = [
    DemoUser(
        user_id="user-anna",
        name="Anna",
        tier="premium",
        allergens=("lepek",),
        notes="Vegan, alergie na lepek. Často kontroluje složení.",
    ),
    DemoUser(
        user_id="user-petr",
        name="Petr",
        tier="free",
        allergens=(),
        notes="Bez omezení. Hledá hlavně cenu.",
    ),
    DemoUser(
        user_id="user-katerina",
        name="Kateřina",
        tier="premium",
        allergens=("mleko", "orechy"),
        notes="Více alergenů. Citlivá na cross-contamination warnings.",
    ),
    DemoUser(
        user_id="user-tomas",
        name="Tomáš",
        tier="free",
        allergens=(),
        notes="Vegan, žádné alergeny. Hledá rostlinné alternativy.",
    ),
    DemoUser(
        user_id="user-marie",
        name="Marie",
        tier="free",
        allergens=("laktoza",),
        notes="Laktóza intolerance. Bezlaktózové mléčné produkty OK.",
    ),
]


def by_id(user_id: str) -> DemoUser:
    for u in DEMO_USERS:
        if u.user_id == user_id:
            return u
    raise KeyError(user_id)
