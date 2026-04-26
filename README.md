# kosik-workshop

Python sandbox pro experimenty s **LangChain**, **LangSmith** a **Jupyter notebooky**.

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — package & env manager
- LangChain + LangSmith
- JupyterLab / notebook
- ruff, pytest, mypy

## Setup

```bash
# 1. Nainstaluj Python 3.12 a závislosti
uv sync --all-groups

# 2. Zkopíruj .env
cp .env.example .env
# vyplň OPENAI_API_KEY, LANGSMITH_API_KEY

# 3. Zaregistruj Jupyter kernel
uv run python -m ipykernel install --user --name kosik-workshop --display-name "Python (kosik-workshop)"
```

## Běžné příkazy

```bash
uv run pytest                 # testy
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src               # type check
uv run jupyter lab            # notebooky
```

## Struktura

```
src/kosik_workshop/   # hlavní balíček
tests/                # pytest
notebooks/            # Jupyter notebooky
```
