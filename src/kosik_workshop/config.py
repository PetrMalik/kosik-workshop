"""Centrální konfigurace.

Drží `.env` loader a sdílené konstanty, které se jinak rozsypávají po
modulech (modely, teploty, batch size). Přepisovat lze přes prostředí —
viz `_env_*` helpery — aby se na CI / v evalu daly snadno tunit bez code change.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None and value.strip() else default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return int(raw)


# Agent model (used by build_agent default)
AGENT_MODEL: str = _env_str("KOSIK_AGENT_MODEL", "gpt-4o-mini")
AGENT_TEMPERATURE: float = _env_float("KOSIK_AGENT_TEMPERATURE", 0.0)

# Catalog generator
CATALOG_MODEL: str = _env_str("KOSIK_CATALOG_MODEL", "gpt-4o-mini")
CATALOG_TEMPERATURE: float = _env_float("KOSIK_CATALOG_TEMPERATURE", 0.8)
CATALOG_BATCH_SIZE: int = _env_int("KOSIK_CATALOG_BATCH_SIZE", 12)

# LLM-as-judge evaluators
JUDGE_MODEL: str = _env_str("KOSIK_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_TEMPERATURE: float = _env_float("KOSIK_JUDGE_TEMPERATURE", 0.0)
