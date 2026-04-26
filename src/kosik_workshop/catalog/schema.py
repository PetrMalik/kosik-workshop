from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Allergen(StrEnum):
    """EU 14 regulated allergens."""

    GLUTEN = "lepek"
    CRUSTACEANS = "korýši"
    EGG = "vejce"
    FISH = "ryby"
    PEANUT = "arašídy"
    SOY = "sója"
    MILK = "mléko"
    NUTS = "skořápkové plody"
    CELERY = "celer"
    MUSTARD = "hořčice"
    SESAME = "sezam"
    SULPHITES = "oxid siřičitý"
    LUPIN = "vlčí bob"
    MOLLUSCS = "měkkýši"


Unit = Literal["ks", "kg", "g", "l", "ml", "balení"]


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:80]


class Product(BaseModel):
    id: str = Field(default="", description="slug; generated from name if empty")
    name: str = Field(min_length=3, max_length=120)
    category: str = Field(min_length=2)
    subcategory: str = Field(min_length=2)
    price_czk: float = Field(gt=0, lt=5000)
    unit: Unit
    description: str = Field(min_length=20, max_length=400)
    allergens: list[Allergen] = Field(default_factory=list)
    vegan: bool = False
    country_of_origin: str = Field(min_length=2, max_length=40)
    in_stock: bool = True
    brand: str | None = None

    @field_validator("allergens")
    @classmethod
    def _unique_allergens(cls, v: list[Allergen]) -> list[Allergen]:
        return sorted(set(v), key=lambda a: a.value)

    @model_validator(mode="after")
    def _fill_id(self) -> Product:
        if not self.id:
            self.id = slugify(self.name)
        return self


class ProductBatch(BaseModel):
    products: list[Product]
