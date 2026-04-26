"""Hand-curated query bank for workshop demo.

30 queries across 8 intents. Each query is labeled with:
- `intent`: matches the `feature:` tag set on traces
- `user_id`: the synthetic user persona this query best simulates (matches
  `kosik_workshop.simulation.demo_users.DEMO_USERS`)

The mix is intentional: enough variety for per-feature dashboards to be
populated, small enough to manually review every trace afterward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoQuery:
    text: str
    intent: str
    user_id: str


DEMO_QUERIES: list[DemoQuery] = [
    # product_search (10) — basic catalog browsing
    DemoQuery("Hledám veganské mléko do 50 Kč.", "product-search", "user-tomas"),
    DemoQuery("Najděte mi pečivo do 30 Kč.", "product-search", "user-petr"),
    DemoQuery("Máte v nabídce máslo?", "product-search", "user-petr"),
    DemoQuery("Hledám sýr do 100 Kč.", "product-search", "user-petr"),
    DemoQuery("Hledám veganské pečivo.", "product-search", "user-tomas"),
    DemoQuery("Co máte z mléčných výrobků?", "product-search", "user-marie"),
    DemoQuery("Najdi mi nejlevnější chleba.", "product-search", "user-petr"),
    DemoQuery("Hledám bezlaktózové mléko.", "product-search", "user-marie"),
    DemoQuery("Máte nějaké rostlinné alternativy mléka?", "product-search", "user-tomas"),
    DemoQuery("Hledám sýr bez ořechů.", "product-search", "user-katerina"),
    # product_details (6) — drill into one product
    DemoQuery("Kolik stojí Madeta Jihočeské máslo 250 g?", "product-details", "user-petr"),
    DemoQuery("Jaké je složení produktu madeta-jihoceske-maslo-250-g?", "product-details", "user-anna"),
    DemoQuery("V jakém balení je Madeta máslo?", "product-details", "user-petr"),
    DemoQuery("Je Madeta Jihočeské máslo vegan?", "product-details", "user-tomas"),
    DemoQuery("Kolik gramů má Madeta máslo?", "product-details", "user-petr"),
    DemoQuery("Je toto pečivo čerstvé?", "product-details", "user-petr"),
    # allergen_check (5) — explicit allergen-related questions
    DemoQuery("Obsahuje produkt madeta-jihoceske-maslo-250-g lepek?", "allergen-check", "user-anna"),
    DemoQuery("Mám alergii na lepek. Doporučte mi pečivo.", "allergen-check", "user-anna"),
    DemoQuery("Doporuč mi rohlík. Mám alergii na lepek.", "allergen-check", "user-anna"),
    DemoQuery("Hledám sýr bez mléčných alergenů.", "allergen-check", "user-katerina"),
    DemoQuery("Doporučte mi sýr, který mi neuškodí podle mého profilu.", "allergen-check", "user-katerina"),
    # recommendation (3) — open-ended
    DemoQuery("Co byste mi doporučili na večeři?", "recommendation", "user-petr"),
    DemoQuery("Doporučte mi něco veganského k snídani.", "recommendation", "user-tomas"),
    DemoQuery("Co je dobré ke kávě?", "recommendation", "user-petr"),
    # profile_query (2)
    DemoQuery("Jaké alergeny mám v profilu?", "profile-query", "user-anna"),
    DemoQuery("Co o mně víte?", "profile-query", "user-katerina"),
    # comparison (1)
    DemoQuery("Co je levnější — máslo nebo margarín?", "comparison", "user-petr"),
    # out_of_catalog (2) — should decline gracefully
    DemoQuery("Máte v nabídce čerstvého kapra?", "out-of-catalog", "user-petr"),
    DemoQuery("Najdete mi hovězí svíčkovou?", "out-of-catalog", "user-petr"),
    # off_topic (1)
    DemoQuery("Ahoj, jak se máš?", "off-topic", "user-petr"),
]


# 30 deliberately. If you change DEMO_QUERIES, keep ~30 for predictable demo timing.
assert len(DEMO_QUERIES) == 30, f"DEMO_QUERIES has {len(DEMO_QUERIES)}, expected 30"
