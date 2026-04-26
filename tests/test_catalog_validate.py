from kosik_workshop.catalog.schema import Allergen, Product
from kosik_workshop.catalog.validate import dedupe, passes_business_rules, validate_all


def _product(**overrides) -> Product:
    base = dict(
        name="Madeta Jihočeské máslo 250 g",
        category="Mléčné výrobky a vejce",
        subcategory="Máslo",
        price_czk=69.9,
        unit="g",
        description="Tradiční jihočeské máslo z mléka od českých farmářů.",
        country_of_origin="Česko",
        allergens=[Allergen.MILK],
    )
    base.update(overrides)
    return Product(**base)


def test_valid_dairy_passes():
    ok, reason = passes_business_rules(_product())
    assert ok, reason


def test_unknown_category_rejected():
    ok, reason = passes_business_rules(_product(category="Nonexistent", subcategory="Xy"))
    assert not ok and "unknown category" in reason


def test_price_out_of_band_rejected():
    ok, reason = passes_business_rules(_product(price_czk=4999))
    assert not ok and "outside band" in reason


def test_vegan_cannot_have_milk_allergen():
    ok, reason = passes_business_rules(_product(vegan=True))
    assert not ok and "milk" in reason


def test_vegan_meat_rejected():
    ok, reason = passes_business_rules(
        _product(
            name="Veganská šunka 150 g",
            category="Maso a ryby",
            subcategory="Kuřecí",
            unit="g",
            price_czk=120,
            allergens=[],
            vegan=True,
        )
    )
    assert not ok and "animal category" in reason


def test_bread_without_gluten_rejected():
    ok, reason = passes_business_rules(
        _product(
            name="Odkolek Šumava chléb 1200 g",
            category="Pečivo",
            subcategory="Chléb",
            unit="g",
            price_czk=55,
            allergens=[],
        )
    )
    assert not ok and "gluten" in reason.lower()


def test_dedupe_by_normalized_name():
    a = _product(name="Madeta Jihočeské máslo 250 g")
    b = _product(name="MADETA  jihočeské máslo  250 g")
    c = _product(name="Kunín Jogurt bílý 150 g", subcategory="Jogurty", unit="g", price_czk=25)
    kept, dups = dedupe([a, b, c])
    assert len(kept) == 2
    assert len(dups) == 1


def test_validate_all_returns_accepted_and_rejected():
    good = _product()
    bad_price = _product(name="Jiný máslo 200 g", price_czk=4999)
    accepted, rejected = validate_all([good, bad_price])
    assert len(accepted) == 1
    assert len(rejected) == 1
