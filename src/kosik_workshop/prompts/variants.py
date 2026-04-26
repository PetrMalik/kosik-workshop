"""Prompt varianty pro A/B testování.

Každá varianta = jméno + system text. Variant `v1_baseline` je kopie aktuálního
produkčního promptu (`kosik_assistant.SYSTEM_PROMPT_V1`); ostatní jsou cílené
úpravy, které měříme proti baseline přes offline evaluators.

Pravidlo: měň jednu věc. Když měníš dvě věci najednou, nevíš, která dělá rozdíl.
"""

from __future__ import annotations

from kosik_workshop.prompts.kosik_assistant import SYSTEM_PROMPT_V1

# Jasnější citation rule. Hypotéza: zlepší `cites_product_id` evaluator skóre
# bez negativního dopadu na ostatní (allergen/no-hallucination beze změny).
SYSTEM_PROMPT_V2_STRICT_CITATIONS = SYSTEM_PROMPT_V1.replace(
    "Ceny uváděj v Kč, u doporučení vždy zmiň `product_id` (slug), aby se dal "
    "produkt jednoznačně dohledat.",
    "Ceny uváděj v Kč. U **každé** zmínky o konkrétním produktu MUSÍŠ uvést "
    "`product_id` (slug) v závorce hned za názvem. Příklad: "
    "*Madeta Jihočeské máslo 250 g (madeta-jihoceske-maslo-250-g) — 49 Kč*. "
    "Bez `product_id` odpověď není kompletní.",
)


# Posílení allergen warning. Hypotéza: zlepší `allergen_flagged_explicitly`.
_V3_OLD = (
    "Dříve než doporučíš produkt zákazníkovi s alergií, ověř alergeny nástrojem "
    'a riziko hlas explicitně (např. *„Obsahuje lepek — pro vás není vhodné."*).'
)
_V3_NEW = (
    "Když uživatel zmíní alergii NEBO má alergeny v profilu, MUSÍŠ na začátku "
    "odpovědi uvést řádek **⚠️ Upozornění na alergeny:** s konkrétním "
    "seznamem. Pokud doporučíš produkt s alergenem ze seznamu, riziko zopakuj "
    "tučně přímo u toho produktu (např. *Obsahuje lepek — NEDOPORUČUJEME*). "
    "Tohle pravidlo je nadřazené stručnosti."
)
SYSTEM_PROMPT_V3_LOUD_ALLERGENS = SYSTEM_PROMPT_V1.replace(_V3_OLD, _V3_NEW)


PROMPT_VARIANTS: dict[str, str] = {
    "v1_baseline": SYSTEM_PROMPT_V1,
    "v2_strict_citations": SYSTEM_PROMPT_V2_STRICT_CITATIONS,
    "v3_loud_allergens": SYSTEM_PROMPT_V3_LOUD_ALLERGENS,
}


def get_variant(name: str) -> str:
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Available: {list(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[name]
