# kosik-workshop

A Python sandbox for hands-on experiments with **LangChain**, **LangGraph**, **LangSmith** and **Jupyter notebooks** — built around a fictional Košík.cz grocery-shopping assistant.

## Stack

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — package & environment manager
- LangChain + LangGraph + LangSmith
- Chroma (local vector store) for the product catalog
- JupyterLab
- ruff, pytest, mypy

## Repository layout

```
src/kosik_workshop/   # main package (agent, tools, prompts, evals, catalog)
notebooks/            # 00 → 06 workshop notebooks (run them in order)
scripts/              # CLI entrypoints (build index, run demo, eval gate, …)
data/products.json    # generated product catalog
data/chroma/          # local Chroma index (built from products.json)
tests/                # pytest
```

---

## 1. Prerequisites

You need:

- **Python 3.12** (managed automatically by `uv`)
- **uv** — install from <https://docs.astral.sh/uv/getting-started/installation/>
- An **OpenAI API key** (paid account, gpt-4o-mini and text-embedding-3-small are enough)
- A **LangSmith account** — free tier is fine: <https://smith.langchain.com/>

---

## 2. Clone & install

```bash
git clone <repo-url> kosik-workshop
cd kosik-workshop

# Installs Python 3.12 if needed and creates .venv with all dependencies
# (the `dev` and `notebook` groups are installed by default)
uv sync
```

Register the Jupyter kernel once so the notebooks can find your env:

```bash
uv run python -m ipykernel install --user \
    --name kosik-workshop \
    --display-name "Python (kosik-workshop)"
```

---

## 3. Configure your API keys

Copy the template and fill in your secrets:

```bash
cp .env.example .env
```

Open `.env` and set:

```dotenv
OPENAI_API_KEY=sk-...               # from platform.openai.com
LANGSMITH_API_KEY=lsv2_pt_...       # from smith.langchain.com → Settings → API Keys
LANGSMITH_TRACING=true              # leave on; this is what enables traces
LANGSMITH_PROJECT=kosik-workshop    # any name — shows up in LangSmith UI

# Use the dev tag of the prompt published on LangSmith Hub
ENVIRONMENT=development

# Optional: pin a specific prompt commit hash (overrides ENVIRONMENT)
# KOSIK_PROMPT_COMMIT=
```

> **Never commit `.env`** — it is in `.gitignore`. Only `.env.example` is tracked.

### Setting up LangSmith

1. Sign up at <https://smith.langchain.com/>.
2. Create a **personal workspace** (free tier).
3. Go to **Settings → API Keys → Create API Key**, copy it into `LANGSMITH_API_KEY`.
4. The first time you run any notebook or script, a project named `kosik-workshop`
   will appear automatically in the LangSmith UI — that is where you will see traces,
   datasets, evaluations, and human feedback.

---

## 4. Build the product catalog & Chroma index

The agent answers questions over a local Chroma vector store of fictional Košík
products. There are two options.

### Option A — use the pre-generated catalog (recommended for the workshop)

`data/products.json` is already in the repo. Just build the Chroma index from it:

```bash
uv run python scripts/build_index.py
```

This reads `data/products.json`, embeds each product with
`text-embedding-3-small`, and writes the index to `data/chroma/`.
Cost is a few cents and it takes under a minute.

### Option B — regenerate the whole catalog (slow, costs more)

This runs an OpenAI generation pipeline that synthesises a fresh catalog from
the taxonomy and then embeds it.

```bash
uv run python scripts/generate_catalog.py
```

You only need this if you want to play with the catalog generator itself.

After either option you should see something like:

```
data/chroma/
├── chroma.sqlite3
└── <uuid>/
```

You can verify the index works with:

```bash
uv run python -c "from kosik_workshop.catalog.store import load_chroma_index; \
print(load_chroma_index().similarity_search('mléko', k=3))"
```

---

## 5. Run the notebooks

```bash
uv run jupyter lab
```

Open the `notebooks/` folder and work through them in order. Each notebook is
self-contained but builds on the previous one:

| # | Notebook | Topic |
|---|----------|-------|
| 00 | `00_hello_langchain.ipynb` | First chat call, tracing into LangSmith |
| 01 | `01_tools_playground.ipynb` | Defining and calling tools |
| 02 | `02_prompt_and_agent.ipynb` | Prompt Hub + LangGraph agent loop |
| 03 | `03_evals.ipynb` | Datasets + offline evaluators |
| 04 | `04_pii_redaction.ipynb` | Tracing-time PII redaction |
| 05 | `05_ab_prompts.ipynb` | A/B comparing two prompt versions |
| 06 | `06_human_feedback_loop.ipynb` | Annotation queue → dataset promotion |

In each notebook, pick the **`Python (kosik-workshop)`** kernel.

---

## 6. Run the agent end-to-end

CLI chat (multi-turn, with thumbs-up/down feedback going to LangSmith):

```bash
uv run python scripts/chat_app.py
```

Demo runner (populates LangSmith with a batch of traces — useful before showing
dashboards):

```bash
uv run python scripts/run_demo.py --scenario baseline
uv run python scripts/run_demo.py --scenario recovery
```

LangGraph dev server (opens LangGraph Studio against the local graph):

```bash
uv run langgraph dev
```

---

## 7. Common commands

```bash
uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src               # type check
uv run jupyter lab            # notebooks
```

---

## 8. CI (GitHub Actions)

Two workflows in `.github/workflows/`:

- **`ci.yml`** — ruff + mypy + pytest on every PR and push to `main`.
  No secrets, runs on fork PRs (~1–2 min).
- **`eval-gate.yml`** — runs `kosik-eval-golden` against the PR branch and posts
  a per-evaluator score comment. Fails when any evaluator drops below the
  threshold (default `0.8`). Triggered only on changes to `prompts/`, `agent.py`,
  `tools.py`, `evals/` and the catalog — the rest of the repo skips OpenAI calls.

**Repository secrets required:**
- `OPENAI_API_KEY`
- `LANGSMITH_API_KEY`

**Manual run with a custom threshold** (Actions UI → eval-gate → Run workflow):

```bash
gh workflow run eval-gate.yml -f threshold=0.7
```

Local equivalent of the eval gate:

```bash
uv run python scripts/ci_eval_gate.py --threshold 0.8 --report-file report.md
```

---

## Troubleshooting

- **`OPENAI_API_KEY not set`** — your `.env` is missing or wasn't loaded; make
  sure you ran the command from the repo root.
- **Notebook can't find the kernel** — re-run the `ipykernel install` step
  from §2.
- **Empty Chroma results** — you forgot to build the index; run
  `uv run python scripts/build_index.py`.
- **No traces in LangSmith** — check `LANGSMITH_TRACING=true` and that
  `LANGSMITH_API_KEY` is valid; traces appear under the project named in
  `LANGSMITH_PROJECT`.
