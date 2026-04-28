# Workshop: LangChain + LangSmith v praxi (Košík demo)
**Hlavní take-aways:**
- Pochopit LangSmith tracing, datasets a evaluators
- Vidět produkční praktiky: prompt versioning, PII redakce, A/B testing, CI eval gate, RAG eval
- Umět odpovědět: *„Jak měříme, že je náš LLM agent dost dobrý — a jak ho bezpečně vypustíme do produkce?"*

---

## Předpoklady

Účastníci mají před workshopem připravené:

1. Naklonované repo a `uv sync --all-groups` proběhlo bez chyby
2. `.env` s `OPENAI_API_KEY` a `LANGSMITH_API_KEY` (workshop key vystavený organizátorem)
3. Zaregistrovaný Jupyter kernel `kosik-workshop`
4. Vygenerovaný katalog a Chroma index:
   ```bash
   uv run python scripts/generate_catalog.py
   uv run python scripts/build_index.py
   ```
5. Otevřený LangSmith projekt `kosik-workshop` v prohlížeči

První bod každé sekce je „check kernelu / tracingu" — nikdo nepojede dál, dokud nevidí svoje runs v LangSmith UI.

---

# Část 1 — Core (0:00 – 3:00)

## Úvod a motivace

- Kontext: Košík.cz demo asistent — agent radí s nákupem, hlídá alergeny, cituje produkty z katalogu
- Architektura na jednom slidu: **Uživatel → LangGraph agent → Tools (Chroma + JSON katalog) → odpověď**, s LangSmith jako „trace recorderem" přes všechno
- Proč to děláme: bez observability a evals neumíme odpovědět, jestli změna promptu něco zlepšila nebo rozbila

## Smoke test — *notebook `00_hello_langchain`*

- Ověření, že každému jede kernel, importy, `.env` se načítá
- První `ChatOpenAI` volání → ukázka, že trace se objeví v LangSmith UI

## Tools playground — *notebook `01_tools_playground`*

- 4 LangChain `@tool` funkce: `search_products`, `get_product_details`, `check_allergens`, `user_allergens`
- Hands-on: každý si zavolá `.invoke()` s vlastním dotazem (vegan sýr do 30 Kč, mléko bez laktózy…)
- Diskuse: proč jsou tools deterministické a oddělené od LLM — testovatelnost, traceabilita, replay

## Agent + tracing — *notebook `02_prompt_and_agent`*

Klíčový blok první půlky.

- LangGraph ReAct agent: `call_model` ↔ `tools` smyčka, `MemorySaver` checkpointer
- Pull systémového promptu z **LangSmith Prompt Hub** (`load_kosik_prompt`) — environment tag (`dev`/`prod`) a pin na konkrétní commit
- Konverzace přes `thread_id`: třetí turn vidí historii prvního
- **Hands-on:** každý položí 3 navazující dotazy, najde svůj thread v LangSmith UI
- Metadata + tagy v traci (`user_id`, `feature`, `session_id`) — jak filtrovat traces v UI
- Rozdíl mezi *trace* (jeden run) a *thread* (konverzace více traces)

## Simulační provoz a UI tour

- Spuštění `scripts/run_demo.py` → 30–40 traces nateče do LangSmith projektu během 2 minut
- Tour po LangSmith UI: filtry podle `scenario:*` tagu, latence, chyby, token cost
- Diskuse: co byste chtěli vidět na vlastním produkčním dashboardu? (latence p95, error rate, cost per session…)

## End-to-end evaluation — *notebook `03_evals`*

Druhý nejdůležitější blok.

- Co je *golden dataset*: 13 hand-curated příkladů v `evals/dataset.py`, vč. prompt-injection edge case
- **Schema enforcement** na datasetu: `inputs_schema` / `outputs_schema` v LangSmith odmítne nesedící rows (i z UI „Add to Dataset")
- Hands-on: `seed_eval_dataset.py` push do LangSmith → každý vidí dataset ve svém projektu
- 5 evaluators v `evals/evaluators.py`:
  - **Code-based deterministické** (`tool_called_correctly`, `cites_product_id`) — rychlé, levné, spustitelné v CI
  - **LLM-as-judge** (`allergen_flagged_explicitly`, `no_hallucinated_products`, `resists_prompt_injection`) — pro věci, které se nedají zachytit regexem (mj. faithfulness/groundedness a bezpečnostní eval)
- Spuštění experimentu z notebooku → každý vidí svoje výsledky v LangSmith Compare UI
- **Failure analysis společně:** otevřeme 2–3 příklady, co propadly, diskutujeme proč
- Rámec: **co se dá měřit kódem, měř kódem; LLM-judge použij jen tam, kde to jinak nejde**

---

# Část 2 — Volitelné moduly

## Modul A: Retrieval (RAG) eval — *prioritní pro většinu skupin*

- Framing: end-to-end eval ti při failu nepoví, jestli selhalo retrieval nebo generation. RAG eval rozkládá agenta na **retrieval** a **generation** vrstvu.
- Tři failure módy: missing retrieval, špatné pořadí, šum v top-k
- Tři metriky:
  - **`recall_at_k`** — kolik z relevantních produktů je v top-k? (deterministické, zdarma)
  - **`mrr`** (Mean Reciprocal Rank) — kde v ranku je první relevantní zásah? (deterministické)
  - **`context_relevance`** — LLM-judge: kolik z retrieved produktů je topicky relevantní k dotazu?
- Hands-on: druhý dataset `kosik-retrieval-golden` (12 příkladů), spuštění retrieval evalu přímo proti `search_products` (bez agenta)
- **A-ha moment:** vedle sebe E2E eval a retrieval eval ze stejného PR. Záměrně rozbijeme retrieval (zhorší se search query) → vidíme, kde je problém. Pak rozbijeme prompt → vidíme, že retrieval zůstává OK, ale E2E padá.
- Diskuse: u reálného katalogu (1M+ SKUs) je retrieval typicky 90 % problému. Co byste ladili? Embedding model? Re-ranker? Hybridní BM25 + vector?

## Modul B: PII redakce a GDPR — *notebook `04_pii_redaction`*

- Tři vrstvy maskování: `scrub_text` regex → `langsmith.anonymizer` → end-to-end přes redacting client
- Hands-on: každý pošle dotaz s falešným emailem a IBAN, najde v LangSmith UI `[EMAIL]` a `[IBAN]` místo plain textu
- **Diskuse pro produkťáky:** co regex NEZACHYTÍ — jména, adresy, kombinace polí. Redakce ≠ úplná anonymizace
- Connect na produkci: kdo z týmu vlastní data flow do LangSmith? Kdo schvaluje, co se loguje?

## Modul C: A/B testing promptů a Prompt Hub — *notebook `05_ab_prompts`*

- 3 varianty systémového promptu v `prompts/variants.py`: baseline, strict citations, loud allergens
- Hands-on: každá varianta proti `kosik-eval-golden` jako samostatný experiment, sdílený `ab_group` v metadatech
- Compare view v LangSmith — per-example diff, kde varianta zlepšila / zhoršila skóre
- **Workflow ukázka:**
  1. `push_prompt.py` → nová verze do Hubu jako `dev`
  2. Eval na `dev` → pokud projde, `promote_prompt.py` ji přetaguje na `prod`
  3. Production agent díky `load_kosik_prompt(env="prod")` automaticky chytí novou verzi — bez deploye
- Diskuse: kdo schvaluje promotion v reálném týmu? Engineering? PM? Compliance?

## Modul D: Annotation queue + human feedback loop — *notebook `06_human_feedback_loop`*

- Co je annotation queue: produkční runs zařazené do review fronty s vlastními feedback klíči
- Reviewer v LangSmith UI hodnotí 0/1 na `correct_tools`, `helpful`, `safe` + volitelný `comment`
- `promote_annotations_to_dataset.py`: heuristika, která rozhoduje, co se promotne do golden datasetu (good = přidat positive case, bad = přidat regression case, neutral = ignorovat)
- **Důležité varování:** „Add to Dataset" tlačítko v LangSmith UI bere `run.inputs` 1:1 — schema-enforced dataset to odmítne, pokud nesedí. Promote skript je tedy „kanonická cesta" mezi queue a datasetem.
- Připojení na to, že golden dataset držíme **code-first** (`GOLDEN_EXAMPLES` v gitu) — queue je jen vstupní kanál pro nové edge-casy, samotný dataset se mění přes PR

## Modul E: LangGraph Studio

- `uv run langgraph dev` — lokální API server + Studio UI v prohlížeči
- Hands-on: vizuální graf agenta (`call_model` ↔ `tools`), interaktivní chat panel, krokování po nodes
- **Time travel / forking:** vrátit se ke konkrétnímu kroku konverzace a forknout novou větev — užitečné pro debugging „co kdyby model odpověděl jinak"
- Edit state za běhu: ručně upravit messages a tool výsledky a nechat graf pokračovat
- Vztah k LangSmith traces: Studio drží **živé thready** v lokálním checkpointeru, traces jsou read-only logy. Nelze přímo „pokračovat" v produkčním threadu — uživatelská hodnota je v rychlé iteraci na nových dotazech.

---

## CI eval gate — produkční bezpečnostní síť

- `scripts/ci_eval_gate.py` + `.github/workflows/eval-gate.yml`
- Pointa: **každý PR, který sahá na `prompts/`, `agent.py`, `tools.py`, `data/products.json` nebo `evals/`, automaticky proběhne přes oba evaly (E2E + retrieval). Pod threshold = červený check, žádný merge.**
- Komentář v PR ukazuje per-evaluator skóre, link na experiment v LangSmith
- Manuální spuštění s vlastním thresholdem přes Actions UI:
  ```bash
  gh workflow run eval-gate.yml -f threshold=0.7
  ```

## Souhrn produkčních praktik

Šest klíčů, které jsme dnes viděli:

1. **Trace všechno** — tagy, metadata, threads
2. **Měř kvalitu offline před deployem** — golden dataset + evaluators
3. **Rozkládej eval na vrstvy** — retrieval i generation samostatně, ne jen end-to-end
4. **Verzuj prompty separátně od kódu** — Hub + dev/prod tagy
5. **A/B testuj změny promptu**, ne deploy a modli se
6. **PII redakce hned od první trace**, ne až když přijde audit

## Q&A + anketa pro follow-up

Otevřená otázka pro skupinu: co by měl pokrýt navazující workshop?
- LangGraph hlubší (multi-agent, subgraphs, persistence)
- Production monitoring (alerty na drop v evaluator scores, cost tracking, latency p95)
- Pokročilý RAG (re-ranker, hybrid BM25 + vector, structured output evaluators)
- Human-in-the-loop ve velkém (queue management, agreement metriky mezi reviewery)
- Cost & latency optimization (prompt caching, model routing, streaming)

---

## Pomocné skripty v repu (kompletní přehled)

Všechny skripty v `scripts/` jsou samostatně spustitelné přes `uv run python scripts/<jmeno>.py`. Některé používáme v hlavní agendě, jiné jsou bonus pro samostatné prozkoumání.

### Bootstrap dat a indexu

- **`generate_catalog.py`** — vygeneruje syntetický katalog 148 produktů přes OpenAI (značky, ceny, alergeny, kategorie) do `data/products.json`. Spouští se jen jednou při setupu repa.
- **`build_index.py`** — postaví Chroma vektorový index nad `data/products.json` do `data/chroma`. Bez OpenAI generování — jen embedding existujících produktů. Pouští se po každé změně katalogu.

### Datasety a evaluace

- **`seed_eval_dataset.py`** — push obou golden datasetů do LangSmith: `kosik-eval-golden` (E2E, 13 příkladů) + `kosik-retrieval-golden` (RAG, 12 příkladů). Idempotentní; `--replace` smaže a nahraje znovu.
- **`ci_eval_gate.py`** — CI gate: pustí oba evaly, spočte per-evaluator skóre, srovná s thresholdem, zapíše markdown report. Volá se z `.github/workflows/eval-gate.yml` na každý PR.
- **`run_ab_eval.py`** — A/B eval runner pro porovnání N prompt variant proti stejnému datasetu. Každá varianta jako samostatný experiment se sdíleným `ab_group` v metadatech (Compare view v LangSmith).
- **`eval_human_agreement.py`** — měří agreement mezi automated evaluatory a human anotacemi z queue (procentní shoda + Cohen's kappa). Pomáhá kalibrovat LLM-judge prompty.

### Prompt management

- **`push_prompt.py`** — bootstrap / rollback: pushne systémový prompt z kódu do LangSmith Hubu jako `dev` tag. Kanonický zdroj promptů je ale **Playground v LangSmith UI**, ne tento skript.
- **`promote_prompt.py`** — přetaguje existující commit promptu z `dev` na `prod`. Workflow: edit v Playgroundu → eval na `dev` → promote.

### Annotation queue (human-in-the-loop)

- **`seed_annotation_queue.py`** — triage policy, která plní queue `kosik-human-review` z produkčních runs podle pravidel (low confidence, allergen mention bez flagu, atd.).
- **`promote_annotations_to_dataset.py`** — promotne human-anotované runs z queue do golden datasetu. Klasifikuje `good` (positive case) / `bad` (regression case) / `neutral` (skip) podle 3 feedback klíčů (`correct_tools`, `helpful`, `safe`).

### Demo a debug

- **`run_demo.py`** — workshop demo runner: spustí 30–40 syntetických konverzací (`src/kosik_workshop/simulation/`) → traces nateklou do LangSmith projektu během 2 minut. Použité v Core sekci pro UI tour.
- **`chat_app.py`** — minimalistický CLI chat s agentem se sběrem end-user feedbacku (multi-turn v jednom threadu). Užitečné pro generování vlastních traces před evalem.
- **`check_scenarios.py`** — diagnostika: tahá z LangSmith feedback per `scenario:*` tag a tiskne tabulku. Užitečné pro post-hoc analýzu, kde scénáře propadají.
- **`diagnose_redaction.py`** — debug nástroj pro PII redakci: pustí jeden agent invoke s falešnou PII a tiskne, co `hide_inputs` / `hide_outputs` tracer hooks reálně vidí. Používá se při ladění regex maskování.
