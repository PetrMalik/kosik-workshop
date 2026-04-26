import pytest

from kosik_workshop.catalog.schema import Allergen, Product, slugify


def test_slugify_removes_diacritics_and_spaces():
    assert slugify("Jihočeské máslo 250 g") == "jihoceske-maslo-250-g"


def test_id_auto_generated_from_name():
    p = Product(
        name="Madeta Jihočeské máslo 250 g",
        category="Mléčné výrobky a vejce",
        subcategory="Máslo",
        price_czk=69.9,
        unit="g",
        description="Tradiční jihočeské máslo z mléka od českých farmářů.",
        country_of_origin="Česko",
    )
    assert p.id == "madeta-jihoceske-maslo-250-g"


def test_allergens_deduplicated_and_sorted():
    p = Product(
        name="Tvaroh jemný 250 g",
        category="Mléčné výrobky a vejce",
        subcategory="Sýry",
        price_czk=35.0,
        unit="g",
        description="Jemný tvaroh vhodný k pečení i přímé spotřebě.",
        country_of_origin="Česko",
        allergens=[Allergen.MILK, Allergen.MILK],
    )
    assert p.allergens == [Allergen.MILK]


def test_price_must_be_positive():
    with pytest.raises(ValueError):
        Product(
            name="Test produkt",
            category="Pečivo",
            subcategory="Chléb",
            price_czk=0,
            unit="ks",
            description="Popis produktu má být dost dlouhý aby prošel.",
            country_of_origin="Česko",
        )
