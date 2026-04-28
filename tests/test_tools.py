import json

import pytest

from kosik_workshop.catalog.schema import Allergen
from kosik_workshop.tools import (
    _load_products_by_id,
    check_allergens,
    get_product_details,
    set_default_user_allergens,
    user_allergens,
)
from kosik_workshop.user_prefs import DEFAULT_USER


@pytest.fixture(autouse=True)
def _reset_user():
    before = list(DEFAULT_USER.allergens)
    yield
    DEFAULT_USER.allergens = before


@pytest.fixture(autouse=True)
def _fake_catalog(tmp_path, monkeypatch):
    products = [
        {
            "id": "test-maslo-250-g",
            "name": "Test Máslo 250 g",
            "category": "Mléčné výrobky a vejce",
            "subcategory": "Máslo",
            "price_czk": 69.9,
            "unit": "g",
            "description": "Testovací máslo pro unit testy.",
            "allergens": ["mléko"],
            "vegan": False,
            "country_of_origin": "Česko",
            "in_stock": True,
            "brand": "Test",
        },
        {
            "id": "test-chleb-500-g",
            "name": "Test Chléb 500 g",
            "category": "Pečivo",
            "subcategory": "Chléb",
            "price_czk": 35.0,
            "unit": "g",
            "description": "Testovací chléb pro unit testy.",
            "allergens": ["lepek"],
            "vegan": True,
            "country_of_origin": "Česko",
            "in_stock": True,
            "brand": "Test",
        },
    ]
    fake_path = tmp_path / "products.json"
    fake_path.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
    import kosik_workshop.tools as tools_module

    monkeypatch.setattr(tools_module, "PRODUCTS_JSON", fake_path)
    _load_products_by_id.cache_clear()
    yield
    _load_products_by_id.cache_clear()


def test_get_product_details_returns_full_record():
    result = get_product_details.invoke({"product_id": "test-maslo-250-g"})
    assert result["name"] == "Test Máslo 250 g"
    assert result["allergens"] == ["mléko"]


def test_get_product_details_missing_returns_error():
    result = get_product_details.invoke({"product_id": "does-not-exist"})
    assert result == {"error": "not_found", "product_id": "does-not-exist"}


def test_check_allergens_safe_when_no_overlap():
    result = check_allergens.invoke({"product_id": "test-chleb-500-g", "user_allergens": ["mléko"]})
    assert result["safe"] is True
    assert result["conflicts"] == []


def test_check_allergens_flags_conflict():
    result = check_allergens.invoke(
        {"product_id": "test-maslo-250-g", "user_allergens": ["mléko", "vejce"]}
    )
    assert result["safe"] is False
    assert result["conflicts"] == ["mléko"]


def test_check_allergens_uses_default_user_when_none_given():
    set_default_user_allergens([Allergen.GLUTEN])
    result = check_allergens.invoke({"product_id": "test-chleb-500-g"})
    assert result["safe"] is False
    assert result["conflicts"] == ["lepek"]


def test_check_allergens_missing_product():
    result = check_allergens.invoke({"product_id": "missing", "user_allergens": []})
    assert result["error"] == "not_found"


def test_user_allergens_reflects_default_user():
    set_default_user_allergens([Allergen.MILK, Allergen.GLUTEN])
    result = user_allergens.invoke({})
    assert sorted(result) == ["lepek", "mléko"]


def test_check_allergens_reads_user_from_config():
    # DEFAULT_USER prázdný; alergeny přijdou jen z config injection.
    config = {"configurable": {"user_allergens": ["lepek"]}}
    result = check_allergens.invoke({"product_id": "test-chleb-500-g"}, config=config)
    assert result["safe"] is False
    assert result["conflicts"] == ["lepek"]


def test_user_allergens_reads_from_config():
    config = {"configurable": {"user_allergens": ["mléko", "lepek"]}}
    result = user_allergens.invoke({}, config=config)
    assert sorted(result) == ["lepek", "mléko"]


def test_check_allergens_explicit_arg_overrides_config():
    # Explicitní user_allergens parametr má přednost před config injection.
    config = {"configurable": {"user_allergens": ["lepek"]}}
    result = check_allergens.invoke(
        {"product_id": "test-chleb-500-g", "user_allergens": []},
        config=config,
    )
    assert result["safe"] is True


def test_check_allergens_invalid_allergens_returned():
    result = check_allergens.invoke({"product_id": "test-chleb-500-g", "user_allergens": ["xyz"]})
    assert result["error"] == "invalid_allergens"
    assert "xyz" in result["unknown"]
