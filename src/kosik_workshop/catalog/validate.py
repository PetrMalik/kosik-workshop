from __future__ import annotations

import re
import unicodedata

from kosik_workshop.catalog.schema import Allergen, Product
from kosik_workshop.catalog.taxonomy import TAXONOMY

ANIMAL_CATEGORIES = {"Maso a ryby", "Uzeniny"}
DAIRY_KEYWORDS = ("mléko", "jogurt", "sýr", "máslo", "smetan", "tvaroh", "kefir")
GLUTEN_KEYWORDS = ("chléb", "rohlík", "bageta", "těstovin", "sušenk", "mouka", "bulg")


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", ascii_text)).strip()


def passes_business_rules(product: Product) -> tuple[bool, str | None]:
    meta = TAXONOMY.get(product.category)
    if meta is None:
        return False, f"unknown category: {product.category}"

    if product.subcategory not in meta["subcategories"]:
        return False, f"subcategory '{product.subcategory}' not in {product.category}"

    lo, hi = meta["price_range"]
    if not (lo <= product.price_czk <= hi):
        return False, f"price {product.price_czk} outside band {lo}-{hi} for {product.category}"

    if product.unit not in meta["units"]:
        return False, f"unit '{product.unit}' not typical for {product.category}"

    if product.vegan and product.category in ANIMAL_CATEGORIES:
        return False, "vegan=true incompatible with animal category"

    if product.vegan and Allergen.MILK in product.allergens:
        return False, "vegan=true but milk allergen present"

    name_lc = product.name.lower()
    if (
        product.category == "Mléčné výrobky a vejce"
        and any(kw in name_lc for kw in DAIRY_KEYWORDS)
        and Allergen.MILK not in product.allergens
    ):
        return False, "dairy product missing milk allergen"

    if (
        product.category == "Pečivo"
        and Allergen.GLUTEN not in product.allergens
        and any(kw in name_lc for kw in GLUTEN_KEYWORDS)
        and "bezlep" not in name_lc
    ):
        return False, "bread product missing gluten allergen"

    return True, None


def dedupe(products: list[Product]) -> tuple[list[Product], list[Product]]:
    seen: dict[str, Product] = {}
    duplicates: list[Product] = []
    for p in products:
        key = _normalize_name(p.name)
        if key in seen:
            duplicates.append(p)
        else:
            seen[key] = p
    return list(seen.values()), duplicates


def validate_all(
    products: list[Product],
) -> tuple[list[Product], list[tuple[Product, str]]]:
    accepted: list[Product] = []
    rejected: list[tuple[Product, str]] = []
    for p in products:
        ok, reason = passes_business_rules(p)
        if ok:
            accepted.append(p)
        else:
            rejected.append((p, reason or "unknown"))
    deduped, dups = dedupe(accepted)
    for d in dups:
        rejected.append((d, "duplicate name"))
    return deduped, rejected
