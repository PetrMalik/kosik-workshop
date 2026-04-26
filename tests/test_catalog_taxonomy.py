from kosik_workshop.catalog.taxonomy import TAXONOMY, total_quota


def test_total_quota_approximately_150():
    total = total_quota()
    assert 140 <= total <= 160, f"unexpected total quota: {total}"


def test_all_categories_have_subcategories_and_units():
    for cat, meta in TAXONOMY.items():
        assert meta["subcategories"], f"{cat} missing subcategories"
        assert meta["units"], f"{cat} missing units"
        lo, hi = meta["price_range"]
        assert 0 < lo < hi < 5000, f"{cat} bad price range"
        assert meta["quota"] > 0
