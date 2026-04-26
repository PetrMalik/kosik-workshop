from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from kosik_workshop.catalog.schema import Product

COLLECTION_NAME = "kosik_products"
EMBEDDING_MODEL = "text-embedding-3-small"


def _embedding_text(p: Product) -> str:
    return f"{p.name}\n{p.description}\n{p.category} > {p.subcategory}"


def _metadata(p: Product) -> dict[str, str | float | bool]:
    return {
        "id": p.id,
        "name": p.name,
        "category": p.category,
        "subcategory": p.subcategory,
        "price_czk": p.price_czk,
        "unit": p.unit,
        "vegan": p.vegan,
        "country_of_origin": p.country_of_origin,
        "brand": p.brand or "",
    }


def build_chroma_index(
    products: list[Product],
    persist_dir: str | Path = "data/chroma",
) -> Chroma:
    persist_dir = str(persist_dir)
    texts = [_embedding_text(p) for p in products]
    metadatas = [_metadata(p) for p in products]
    ids = [p.id for p in products]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )


def load_chroma_index(persist_dir: str | Path = "data/chroma") -> Chroma:
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
