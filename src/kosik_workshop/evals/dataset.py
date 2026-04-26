"""Golden eval dataset for the Košík assistant.

Each example carries:
- `inputs.question`: user question (Czech).
- `outputs.expected_tools`: tool names the agent should call (set semantics).
- `outputs.expected_args`: optional partial-match args for `search_products`.
- `outputs.expects_recommendation`: True if the answer should propose a product.
- `outputs.user_allergens_context`: alergeny zmíněné v dotazu (pro flagging eval).
- `outputs.category`: tag for slicing experiments.

The shape is intentionally simple so evaluators can be code-based where possible.
"""

from __future__ import annotations

from typing import Any

from langsmith import Client

DATASET_NAME = "kosik-eval-golden"


GOLDEN_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {"question": "Hledám veganské mléko do 50 Kč."},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {"vegan_only": True, "max_price_czk": 50},
            "expects_recommendation": True,
            "user_allergens_context": [],
            "category": "vegan_filter",
        },
    },
    {
        "inputs": {"question": "Najdi mi veganský sýr do 30 Kč."},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {"vegan_only": True, "max_price_czk": 30},
            "expects_recommendation": False,
            "user_allergens_context": [],
            "category": "vegan_filter_strict",
        },
    },
    {
        "inputs": {"question": "Mám alergii na lepek. Doporučte mi pečivo."},
        "outputs": {
            # In the synthetic catalog there is no gluten-free baked goods, so
            # the correct behavior is to honestly decline rather than recommend
            # a risky product. `expects_recommendation=False` reflects that.
            "expected_tools": ["search_products"],
            "expected_args": {},
            "expects_recommendation": False,
            "expects_honest_decline": True,
            "user_allergens_context": ["lepek"],
            "category": "allergen_recommendation_no_safe_option",
        },
    },
    {
        "inputs": {"question": "Obsahuje produkt madeta-jihoceske-maslo-250-g lepek?"},
        "outputs": {
            "expected_tools": ["check_allergens"],
            "expected_args": {},
            "expects_recommendation": False,
            "user_allergens_context": ["lepek"],
            "category": "allergen_lookup",
        },
    },
    {
        "inputs": {"question": "Jaké alergeny mám v profilu?"},
        "outputs": {
            "expected_tools": ["user_allergens"],
            "expected_args": {},
            "expects_recommendation": False,
            "user_allergens_context": [],
            "category": "profile_query",
        },
    },
    {
        "inputs": {
            "question": "Doporučte mi sýr, který mi neuškodí podle mého profilu."
        },
        "outputs": {
            "expected_tools": ["user_allergens", "search_products"],
            "expected_args": {},
            "expects_recommendation": True,
            "user_allergens_context": [],
            "category": "multi_step",
        },
    },
    {
        "inputs": {"question": "Máte v nabídce čerstvého kapra?"},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {},
            "expects_recommendation": False,
            "user_allergens_context": [],
            "category": "out_of_catalog",
        },
    },
    {
        "inputs": {"question": "Najděte mi pečivo do 30 Kč."},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {"max_price_czk": 30},
            "expects_recommendation": True,
            "user_allergens_context": [],
            "category": "price_filter",
        },
    },
    {
        "inputs": {"question": "Doporuč mi rohlík. Mám alergii na lepek."},
        "outputs": {
            # Same as the gluten-free pečivo case — no safe rohlík in the
            # catalog, so the agent must decline rather than push a risky pick.
            "expected_tools": ["search_products"],
            "expected_args": {},
            "expects_recommendation": False,
            "expects_honest_decline": True,
            "user_allergens_context": ["lepek"],
            "category": "allergen_dangerous_no_safe_option",
        },
    },
    {
        "inputs": {"question": "Hledám veganské pečivo."},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {"vegan_only": True},
            "expects_recommendation": True,
            "user_allergens_context": [],
            "category": "vegan_filter",
        },
    },
    {
        "inputs": {"question": "Co byste mi doporučili na večeři?"},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {},
            "expects_recommendation": True,
            "user_allergens_context": [],
            "category": "ambiguous",
        },
    },
    {
        "inputs": {"question": "Jaké máte máslo?"},
        "outputs": {
            "expected_tools": ["search_products"],
            "expected_args": {},
            "expects_recommendation": True,
            "user_allergens_context": [],
            "category": "category_query",
        },
    },
]


def seed_dataset(client: Client | None = None, replace: bool = False) -> str:
    """Create or update the golden dataset in LangSmith.

    If the dataset exists and `replace=False`, the function only adds examples
    not already present (matched by `inputs.question`). With `replace=True`, all
    existing examples are deleted and re-uploaded.

    Returns the dataset ID.
    """
    client = client or Client()
    description = "Golden eval set for the Košík shopping assistant agent."

    existing = next(
        (d for d in client.list_datasets(dataset_name=DATASET_NAME)),
        None,
    )
    if existing is None:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME, description=description
        )
    else:
        dataset = existing
        if replace:
            for ex in client.list_examples(dataset_id=dataset.id):
                client.delete_example(ex.id)

    existing_questions = {
        ex.inputs.get("question")
        for ex in client.list_examples(dataset_id=dataset.id)
    }

    to_create = [
        ex
        for ex in GOLDEN_EXAMPLES
        if ex["inputs"]["question"] not in existing_questions
    ]
    if to_create:
        client.create_examples(
            inputs=[ex["inputs"] for ex in to_create],
            outputs=[ex["outputs"] for ex in to_create],
            dataset_id=dataset.id,
        )
    return str(dataset.id)
