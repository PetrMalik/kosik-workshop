from __future__ import annotations

import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from kosik_workshop.prompts.kosik_assistant import PROMPT_NAME

logger = logging.getLogger(__name__)

_ENV_TO_TAG = {
    "development": "dev",
    "dev": "dev",
    "production": "prod",
    "prod": "prod",
}


def _resolve_identifier() -> str:
    pinned = os.getenv("KOSIK_PROMPT_COMMIT", "").strip()
    if pinned:
        return f"{PROMPT_NAME}:{pinned}"

    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    tag = _ENV_TO_TAG.get(env)
    if tag is None:
        raise ValueError(
            f"Unknown ENVIRONMENT={env!r}. Expected one of: {sorted(_ENV_TO_TAG)}."
        )
    return f"{PROMPT_NAME}:{tag}"


def load_kosik_prompt(client: Client | None = None) -> ChatPromptTemplate:
    """Pull the Košík system prompt from LangSmith Prompt Hub.

    Priority:
    1. `KOSIK_PROMPT_COMMIT` — pin to a specific commit hash (reproducibility).
    2. `ENVIRONMENT` → tag (`development`→`dev`, `production`→`prod`).

    Returns a ChatPromptTemplate. The model is bound separately in code.
    """
    identifier = _resolve_identifier()
    logger.info("Pulling prompt %s", identifier)
    client = client or Client()
    prompt = client.pull_prompt(identifier)
    if not isinstance(prompt, ChatPromptTemplate):
        raise TypeError(
            f"Expected ChatPromptTemplate from Hub, got {type(prompt).__name__}. "
            "Verify that push_prompt.py is pushing a ChatPromptTemplate."
        )
    return prompt
