from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from kosik_workshop.catalog.schema import Allergen
from kosik_workshop.user_prefs import DEFAULT_USER

USER_ALLERGENS_CONFIG_KEY = "user_allergens"


def _resolve_user_allergens(config: RunnableConfig | None) -> list[str]:
    """Vrátí alergeny uživatele pro aktuální invocation.

    Priorita:
      1. `config["configurable"]["user_allergens"]` — produkční path (simulátor,
         eval, vícevláknové runs). Každý invoke má vlastní isolovaný kontext.
      2. `DEFAULT_USER.allergens` — workshop fallback pro `tool.invoke({})`
         z notebooků, kde žádný config není k dispozici.
    """
    if config is not None:
        configurable = config.get("configurable") or {}
        injected = configurable.get(USER_ALLERGENS_CONFIG_KEY)
        if injected is not None:
            return [a if isinstance(a, str) else a.value for a in injected]
    return [a.value for a in DEFAULT_USER.allergens]


ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_JSON = ROOT / "data" / "products.json"
CHROMA_DIR = ROOT / "data" / "chroma"


@lru_cache(maxsize=1)
def _load_products_by_id() -> dict[str, dict[str, Any]]:
    if not PRODUCTS_JSON.exists():
        raise FileNotFoundError(
            f"Product catalog not found at {PRODUCTS_JSON}. "
            "Run `uv run python -m scripts.generate_catalog` first."
        )
    raw = json.loads(PRODUCTS_JSON.read_text(encoding="utf-8"))
    return {p["id"]: p for p in raw}


@lru_cache(maxsize=1)
def _load_chroma():
    from kosik_workshop.catalog.store import load_chroma_index

    return load_chroma_index(CHROMA_DIR)


def _summary(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "subcategory": product["subcategory"],
        "price_czk": product["price_czk"],
        "unit": product["unit"],
        "vegan": product["vegan"],
    }


@tool
def search_products(
    query: str,
    category: str | None = None,
    max_price_czk: float | None = None,
    vegan_only: bool = False,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Najde produkty v katalogu Košík.cz sémanticky podle dotazu.

    Args:
        query: Přirozený dotaz v češtině (např. "bezlepkový chléb", "česká piva").
        category: Volitelný přesný filtr kategorie (např. "Pečivo", "Nápoje").
        max_price_czk: Volitelný strop ceny v Kč.
        vegan_only: Pokud True, vrátí jen veganské produkty.
        k: Počet výsledků (default 5, max 20).

    Returns:
        Seznam produktů se základními atributy.
    """
    k = max(1, min(k, 20))

    where_clauses: list[dict[str, Any]] = []
    if category is not None:
        where_clauses.append({"category": category})
    if vegan_only:
        where_clauses.append({"vegan": True})
    if max_price_czk is not None:
        where_clauses.append({"price_czk": {"$lte": float(max_price_czk)}})

    where: dict[str, Any] | None
    if len(where_clauses) == 0:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    db = _load_chroma()
    docs = db.similarity_search(query, k=k, filter=where)

    by_id = _load_products_by_id()
    results: list[dict[str, Any]] = []
    for d in docs:
        pid = d.metadata.get("id")
        if pid and pid in by_id:
            results.append(_summary(by_id[pid]))
    return results


@tool
def get_product_details(product_id: str) -> dict[str, Any]:
    """Vrátí plné informace o produktu podle jeho ID.

    Args:
        product_id: Slug produktu (např. "madeta-jihoceske-maslo-250-g").

    Returns:
        Všechny atributy produktu včetně popisu, alergenů a země původu.
        Pokud produkt neexistuje, vrátí `{"error": "not_found", "product_id": ...}`.
    """
    by_id = _load_products_by_id()
    product = by_id.get(product_id)
    if product is None:
        return {"error": "not_found", "product_id": product_id}
    return product


@tool
def check_allergens(
    product_id: str,
    user_allergens: list[str] | None = None,
    config: RunnableConfig = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Zkontroluje, zda je produkt bezpečný vzhledem k alergenům uživatele.

    Args:
        product_id: Slug produktu.
        user_allergens: Seznam alergenů uživatele (názvy jako "mléko", "lepek").
            Pokud None, převezme se z `RunnableConfig` (klíč
            `configurable.user_allergens`); pokud ani tam není, použije se
            `DEFAULT_USER.allergens`.

    Returns:
        {
          "safe": bool,
          "conflicts": [...alergeny v průniku...],
          "product_allergens": [...],
          "user_allergens": [...],
        }
        Nebo `{"error": "not_found", ...}` pokud produkt neexistuje.
    """
    by_id = _load_products_by_id()
    product = by_id.get(product_id)
    if product is None:
        return {"error": "not_found", "product_id": product_id}

    if user_allergens is None:
        effective_user = _resolve_user_allergens(config)
    else:
        valid = {a.value for a in Allergen}
        unknown = [a for a in user_allergens if a not in valid]
        if unknown:
            return {
                "error": "invalid_allergens",
                "unknown": unknown,
                "allowed": sorted(valid),
            }
        effective_user = list(user_allergens)
    product_allergens: list[str] = list(product.get("allergens", []))
    conflicts = sorted(set(product_allergens) & set(effective_user))
    return {
        "safe": len(conflicts) == 0,
        "conflicts": conflicts,
        "product_allergens": product_allergens,
        "user_allergens": effective_user,
    }


@tool
def user_allergens(config: RunnableConfig = None) -> list[str]:  # type: ignore[assignment]
    """Vrátí aktuální seznam alergenů uživatele.

    Čte z `RunnableConfig` (`configurable.user_allergens`); pokud nic,
    fallbackuje na `DEFAULT_USER.allergens` (workshop default).

    Returns:
        Seznam názvů alergenů (prázdný, pokud uživatel nemá žádné).
    """
    return _resolve_user_allergens(config)


ALL_TOOLS = [search_products, get_product_details, check_allergens, user_allergens]


def set_default_user_allergens(allergens: list[Allergen | str]) -> None:
    """Workshop helper: přepíše `DEFAULT_USER.allergens` bez perzistence.

    Slouží jen pro single-thread notebookové scénáře (`tool.invoke({})` bez
    `RunnableConfig`). V produkčním / paralelním kódu (simulátor, eval,
    A/B runs) **NEPOUŽÍVAT** — předávejte alergeny přes
    `config={"configurable": {"user_allergens": [...]}}` místo mutace globálu.
    """
    normalized: list[Allergen] = []
    for a in allergens:
        normalized.append(a if isinstance(a, Allergen) else Allergen(a))
    DEFAULT_USER.allergens = normalized
