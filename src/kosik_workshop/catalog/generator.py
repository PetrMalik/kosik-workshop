from __future__ import annotations

import asyncio
import logging
from typing import Any

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from kosik_workshop.catalog.schema import Product, ProductBatch
from kosik_workshop.catalog.taxonomy import TAXONOMY, CategoryMeta

log = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.8
BATCH_SIZE = 12


SYSTEM_PROMPT = """Jsi asistent, který generuje realistická testovací data pro český online
potravinový e-shop Košík.cz. Produkty musí působit autenticky — jako skutečné položky
v obchodě, které by český zákazník poznal.

Pravidla pro KAŽDÝ produkt:
- Název v češtině. Typicky: "Značka Typ produktu gramáž" (např. "Madeta Jihočeské máslo 250 g").
- `description` 1–2 věty v češtině, konkrétní (složení, použití, země původu).
- `allergens` pouze z EU-14 seznamu: lepek, korýši, vejce, ryby, arašídy, sója, mléko,
  skořápkové plody, celer, hořčice, sezam, oxid siřičitý, vlčí bob, měkkýši.
- `vegan=true` jen když v produktu skutečně nejsou živočišné suroviny.
- `country_of_origin` realistická (Česko, Polsko, Itálie, Španělsko...).
- `id` nech prázdné, vygeneruje se automaticky ze jména.
- V jedné dávce ŽÁDNÉ duplicitní kombinace značka + typ."""


def _batch_prompt(category: str, meta: CategoryMeta, n: int) -> str:
    subcats = ", ".join(meta["subcategories"])
    units = ", ".join(meta["units"])
    lo, hi = meta["price_range"]
    return (
        f'Vygeneruj {n} produktů v kategorii "{category}".\n\n'
        f"Povolené podkategorie (vyber jednu per produkt): {subcats}\n"
        f"Povolené jednotky: {units}\n"
        f"Cenové rozmezí v Kč: {lo}–{hi}\n"
        f"Kontext: {meta['note']}\n\n"
        f"V odpovědi pole `products` musí obsahovat právě {n} položek. "
        f'Všechny musí mít `category` přesně rovno "{category}".'
    )


async def generate_batch(
    client: AsyncOpenAI, category: str, meta: CategoryMeta, n: int
) -> list[Product]:
    try:
        response = await client.beta.chat.completions.parse(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _batch_prompt(category, meta, n)},
            ],
            response_format=ProductBatch,
        )
    except Exception as e:
        log.warning("batch failed for %s (n=%d): %s", category, n, e)
        return []

    parsed = response.choices[0].message.parsed
    if parsed is None:
        log.warning("batch returned None for %s", category)
        return []

    for p in parsed.products:
        p.category = category
    return parsed.products


def _split_into_batches(quota: int, size: int = BATCH_SIZE) -> list[int]:
    batches = [size] * (quota // size)
    rest = quota % size
    if rest:
        batches.append(rest)
    return batches


def _make_client() -> AsyncOpenAI:
    return wrap_openai(AsyncOpenAI())


async def generate_catalog(
    taxonomy: dict[str, CategoryMeta] | None = None,
    client: AsyncOpenAI | None = None,
) -> list[Product]:
    tax = taxonomy if taxonomy is not None else TAXONOMY
    cli = client if client is not None else _make_client()

    tasks: list[Any] = []
    for category, meta in tax.items():
        for batch_size in _split_into_batches(meta["quota"]):
            tasks.append(generate_batch(cli, category, meta, batch_size))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    products: list[Product] = []
    for r in results:
        if isinstance(r, list):
            products.extend(r)
        else:
            log.warning("task exception: %s", r)
    return products


async def generate_fill_gap(
    missing_per_category: dict[str, int],
    client: AsyncOpenAI | None = None,
) -> list[Product]:
    if not missing_per_category:
        return []
    cli = client if client is not None else _make_client()
    tasks = [
        generate_batch(cli, cat, TAXONOMY[cat], n)
        for cat, n in missing_per_category.items()
        if n > 0 and cat in TAXONOMY
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    products: list[Product] = []
    for r in results:
        if isinstance(r, list):
            products.extend(r)
    return products
