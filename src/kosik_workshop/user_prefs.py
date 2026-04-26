from __future__ import annotations

from pydantic import BaseModel

from kosik_workshop.catalog.schema import Allergen


class UserPreferences(BaseModel):
    """User preferences — reference default for tools."""

    name: str = "anonymous"
    allergens: list[Allergen] = []


DEFAULT_USER = UserPreferences()
