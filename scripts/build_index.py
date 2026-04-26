"""Index-only build: data/products.json → data/chroma.

Na rozdíl od `generate_catalog.py` neprovádí OpenAI generování katalogu —
pouze načte už existující `products.json` a postaví Chroma index. Hodí se pro
CI, kde chceme deterministický a levný setup (volá pouze `text-embedding-3-small`
~jednou za N produktů).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kosik_workshop.catalog.schema import Product
from kosik_workshop.catalog.store import build_chroma_index
from kosik_workshop.config import load_env

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS_JSON = ROOT / "data" / "products.json"
CHROMA_DIR = ROOT / "data" / "chroma"


def main() -> int:
    load_env()

    if not PRODUCTS_JSON.exists():
        print(f"error: {PRODUCTS_JSON} not found", file=sys.stderr)
        return 1

    raw = json.loads(PRODUCTS_JSON.read_text(encoding="utf-8"))
    products = [Product.model_validate(item) for item in raw]
    print(f"Loaded {len(products)} products from {PRODUCTS_JSON.relative_to(ROOT)}")

    build_chroma_index(products, persist_dir=CHROMA_DIR)
    print(f"Indexed → {CHROMA_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
