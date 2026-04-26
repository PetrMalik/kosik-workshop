from __future__ import annotations

from typing import TypedDict

from kosik_workshop.catalog.schema import Unit


class CategoryMeta(TypedDict):
    subcategories: list[str]
    quota: int
    price_range: tuple[float, float]
    units: list[Unit]
    note: str


TAXONOMY: dict[str, CategoryMeta] = {
    "Ovoce a zelenina": {
        "subcategories": ["Ovoce", "Zelenina", "Bylinky", "Saláty"],
        "quota": 18,
        "price_range": (15, 250),
        "units": ["kg", "ks", "balení"],
        "note": "Uvádět zemi původu (Česko, Španělsko, Maroko...). Název často obsahuje odrůdu.",
    },
    "Pečivo": {
        "subcategories": ["Chléb", "Rohlíky", "Sladké pečivo", "Bagety"],
        "quota": 12,
        "price_range": (10, 180),
        "units": ["ks", "g", "balení"],
        "note": (
            "Typicky obsahuje lepek. Česká pekárenská tradice (Odkolek, Penam, lokální pekárny)."
        ),
    },
    "Mléčné výrobky a vejce": {
        "subcategories": ["Mléko", "Jogurty", "Sýry", "Máslo", "Smetana", "Vejce"],
        "quota": 16,
        "price_range": (20, 350),
        "units": ["l", "ks", "kg", "g", "ml"],
        "note": "Značky: Madeta, Olma, Hollandia, Kunín, Choceňská mlékárna.",
    },
    "Maso a ryby": {
        "subcategories": ["Kuřecí", "Vepřové", "Hovězí", "Ryby", "Mleté maso"],
        "quota": 12,
        "price_range": (50, 800),
        "units": ["kg", "g", "balení"],
        "note": "Uvádět zemi původu. Chlazené i mražené varianty.",
    },
    "Uzeniny": {
        "subcategories": ["Šunky", "Salámy", "Párky", "Paštiky"],
        "quota": 10,
        "price_range": (30, 400),
        "units": ["g", "kg", "ks", "balení"],
        "note": "Typicky české značky: Krahulík, Kostelecké uzeniny, Váhala.",
    },
    "Trvanlivé potraviny": {
        "subcategories": ["Těstoviny", "Rýže", "Luštěniny", "Konzervy", "Oleje", "Mouka", "Cukr"],
        "quota": 18,
        "price_range": (15, 400),
        "units": ["g", "kg", "l", "ml", "ks", "balení"],
        "note": "Italské těstoviny (Barilla, De Cecco), české mlýny (Mlýn Pernštejn).",
    },
    "Nápoje": {
        "subcategories": ["Voda", "Limonády", "Džusy", "Káva", "Čaj", "Mléka rostlinná"],
        "quota": 14,
        "price_range": (15, 500),
        "units": ["l", "ml", "ks", "g", "balení"],
        "note": "Značky: Mattoni, Kofola, Relax, Lavazza, Dilmah.",
    },
    "Alkohol": {
        "subcategories": ["Pivo", "Víno", "Lihoviny"],
        "quota": 10,
        "price_range": (25, 1500),
        "units": ["l", "ml", "ks", "balení"],
        "note": "Česká piva (Pilsner Urquell, Budvar), moravská vína, sulfity typicky v allergens.",
    },
    "Mražené potraviny": {
        "subcategories": ["Zelenina mražená", "Zmrzliny", "Hotová jídla", "Ryby mražené"],
        "quota": 10,
        "price_range": (30, 400),
        "units": ["g", "kg", "ml", "l", "balení"],
        "note": "Značky: Nowaco, Ardo, Algida, Mövenpick.",
    },
    "Sladkosti a snacky": {
        "subcategories": ["Čokolády", "Sušenky", "Bonbony", "Chipsy", "Oříšky"],
        "quota": 14,
        "price_range": (15, 350),
        "units": ["g", "ks", "balení"],
        "note": "Orion, Opavia, Milka, Lindt, Bohemia Chips.",
    },
    "Drogerie": {
        "subcategories": ["Prací prostředky", "Mycí prostředky", "Hygiena", "Toaletní papír"],
        "quota": 10,
        "price_range": (25, 600),
        "units": ["l", "ml", "ks", "kg", "g", "balení"],
        "note": "Persil, Ariel, Jar, Zewa, Nivea. Alergeny většinou žádné.",
    },
    "Dětské": {
        "subcategories": ["Kojenecká výživa", "Plenky", "Mléka"],
        "quota": 6,
        "price_range": (30, 800),
        "units": ["g", "ml", "ks", "balení"],
        "note": "Hami, Sunar, Pampers, HiPP.",
    },
}


def total_quota() -> int:
    return sum(meta["quota"] for meta in TAXONOMY.values())
