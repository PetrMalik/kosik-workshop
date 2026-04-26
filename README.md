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

## CI (GitHub Actions)

Dva workflows v `.github/workflows/`:

- **`ci.yml`** — ruff + mypy + pytest na každém PR a push do `main`.
  Žádné secrets, běží i na fork PR (~1–2 min).
- **`eval-gate.yml`** — spustí `kosik-eval-golden` proti PR brunchi a postne
  do PR komentář s per-evaluator skóre. Failne, když některý evaluator klesne
  pod threshold (default `0.8`). Triggeruje se jen na změny v `prompts/`,
  `agent.py`, `tools.py`, `evals/` a katalogu — zbytek repa nedělá zbytečné
  OpenAI volání.

**Secrets v repo settings:**
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`

**Manuální spuštění s jiným thresholdem** (Actions UI → eval-gate → Run workflow):

```bash
gh workflow run eval-gate.yml -f threshold=0.7
```

Lokální ekvivalent eval gate:

```bash
uv run python scripts/ci_eval_gate.py --threshold 0.8 --report-file report.md
```
