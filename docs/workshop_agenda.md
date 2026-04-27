# Workshop: LangChain + LangSmith v praxi (Košík demo)

**Délka:** 4 hodiny (240 min, z toho 20 min pauza)
**Formát:** hands-on — každý účastník si pouští notebooky 00–05 lokálně
**Cílovka:** mix vývojáři + produkťáci / data lidi
**Hlavní take-aways:**
- Pochopit LangSmith tracing, datasets a evaluators
- Vidět production-ready praktiky: prompt versioning, PII redakce, A/B testing, CI eval gate
- Umět odpovědět: *"Jak měříme, že je náš LLM agent dost dobrý — a jak ho bezpečně vypustíme do produkce?"*

---

## Předpoklady (rozeslat den předem)

Účastníci mají před workshopem:

1. Naklonované repo a `uv sync --all-groups` proběhlo bez chyby
2. `.env` s `OPENAI_API_KEY` a `LANGSMITH_API_KEY` (workshop key vystavený organizátorem)
3. Zaregistrovaný Jupyter kernel `kosik-workshop`
4. Spuštěný `uv run python scripts/generate_catalog.py` a `uv run python scripts/build_index.py` (Chroma index s katalogem)
5. Otevřený LangSmith projekt `kosik-workshop` v prohlížeči

Prvních 5 minut workshopu věnujeme troubleshootingu — kdo nemá běžící kernel, sedí vedle někoho kdo ho má.

---

## Blok A — Od první chain k observabilnímu agentovi (0:00 – 1:50)

### 0:00 – 0:15 · Úvod a motivace (15 min) — *společně, slidy*

- Kontext: Košík.cz demo asistent — agent radí s nákupem, hlídá alergeny, cituje produkty z katalogu
- Architektura na jednom slidu: **Uživatel → LangGraph agent → Tools (Chroma + JSON katalog) → odpověď**, s LangSmith jako "trace recorderem" přes všechno
- Co je LangSmith ve třech větách: tracing, datasets+evals, prompt hub
- Proč to děláme: bez observability a evals neumíš odpovědět, jestli změna promptu něco zlepšila nebo rozbila

### 0:15 – 0:25 · Smoke test (10 min) — *notebook `00_hello_langchain`*

- Ověříme, že každému jede kernel, importy, `.env` se načítá
- První `ChatOpenAI` volání → ukázka, že trace se objeví v LangSmith UI
- **Záchytný bod:** kdo tady neuvidí trace, dál nepojede — řešíme hned

### 0:25 – 0:45 · Tools playground (20 min) — *notebook `01_tools_playground`*

- 4 LangChain `@tool` funkce: `search_products`, `get_product_details`, `check_allergens`, `user_allergens`
- Hands-on: každý si zavolá `.invoke()` s vlastním dotazem (vegan sýr do 30 Kč, mléko bez laktózy…)
- Diskuse: proč jsou tools deterministické a oddělené od LLM — testovatelnost, traceabilita, replay
- **Tip pro produkťáky:** každý tool call je v traci samostatně — můžete vidět, co agent "myslel", že potřebuje

### 0:45 – 1:35 · Agent + tracing (50 min) — *notebook `02_prompt_and_agent`*

Nejdůležitější blok první půlky. Pomalu, s prohlídkou LangSmith UI po každé fázi.

- LangGraph ReAct agent: `call_model` ↔ `tools` smyčka, `MemorySaver` checkpointer
- Pull systémového promptu z **LangSmith Prompt Hub** (`load_kosik_prompt`) — ukázka environment tagu (`dev`/`prod`) a pinu na konkrétní commit
- Konverzace přes `thread_id`: třetí turn vidí historii prvního
- **Hands-on cvičení:** každý položí 3 navazující dotazy, najde svůj thread v LangSmith UI, ukáže sousedovi
- Metadata + tagy v traci (`user_id`, `feature`, `session_id`) — jak filtrovat traces v UI
- Krátká odbočka: rozdíl mezi *trace* (jeden run) a *thread* (konverzace více traces)

### 1:35 – 1:50 · Live demo: simulační provoz a dashboard (15 min) — *společně*

- Spustíme `scripts/run_demo.py` → 30–40 traces nateče do LangSmith projektu během 2 minut
- Tour po LangSmith UI: filtry podle `scenario:*` tagu, latence, chyby, token cost
- Diskuse: co byste chtěli vidět na vlastním produkčním dashboardu? (latence p95, error rate, cost per session…)

### 1:50 – 2:10 · ☕ Pauza (20 min)

---

## Blok B — Měření kvality a produkční bezpečnost (2:10 – 4:00)

### 2:10 – 3:05 · Evals: golden dataset a evaluators (55 min) — *notebook `03_evals`*

Druhý nejdůležitější blok. Tady se rozhoduje, jestli si účastníci odnesou skutečnou hodnotu.

- Co je *golden dataset*: 12 hand-curated příkladů v `evals/dataset.py`, 5 kategorií (allergens, recommendations, citations…)
- Hands-on: `seed_eval_dataset.py` push do LangSmith → každý vidí dataset ve svém projektu
- 4 evaluators v `evals/evaluators.py`:
  - **Code-based deterministické** (`tool_called_correctly`, `cites_product_id`) — rychlé, levné, spustitelné v CI
  - **LLM-as-judge** (`allergen_flagged_explicitly`, `no_hallucinated_products`) — pro věci, které se nedají zachytit regexem
- Spuštění experimentu z notebooku → každý vidí svoje výsledky v LangSmith Compare UI
- **Failure analysis společně:** otevřeme 2–3 příklady, co propadly, diskutujeme proč
- Rámec: **co se dá měřit kódem, měř kódem; LLM-judge použij jen tam, kde to jinak nejde**

### 3:05 – 3:25 · PII redakce a GDPR (20 min) — *notebook `04_pii_redaction`*

- Tři vrstvy maskování: `scrub_text` regex → `langsmith.anonymizer` → end-to-end přes redacting client
- Hands-on: každý pošle dotaz s falešným emailem a IBAN, najde v LangSmith UI `[EMAIL]` a `[IBAN]` místo plain textu
- **Důležitá diskuse pro produkťáky:** co regex NEZACHYTÍ — jména, adresy, kombinace polí. Redakce ≠ úplná anonymizace
- Connect na produkci: kdo z týmu vlastní data flow do LangSmith? Kdo schvaluje, co se loguje?

### 3:25 – 3:50 · A/B testing promptů a Prompt Hub (25 min) — *notebook `05_ab_prompts`*

- 3 varianty systémového promptu v `prompts/variants.py`: baseline, strict citations, loud allergens
- Hands-on: každá varianta proti `kosik-eval-golden` jako samostatný experiment, sdílený `ab_group` v metadatech
- Compare view v LangSmith — per-example diff, kde varianta zlepšila / zhoršila skóre
- **Workflow ukázka:**
  1. `push_prompt.py` → nová verze do Hubu jako `dev`
  2. eval na `dev` → pokud projde, `promote_prompt.py` ji přetaguje na `prod`
  3. Production agent díky `load_kosik_prompt(env="prod")` automaticky chytí novou verzi — bez deploye
- Diskuse: kdo schvaluje promotion v reálném týmu? Engineering? PM? Compliance?

### 3:50 – 4:00 · CI eval gate, take-aways, Q&A (10 min)

- Krátká ukázka `scripts/ci_eval_gate.py` a `.github/workflows/eval-gate.yml`
- Pointa: **každý PR, který sahá na `prompts/`, `agent.py` nebo `tools.py`, automaticky proběhne přes evaly. Pod threshold = červený check, žádný merge.**
- Souhrn 5 produkčních praktik, co jsme dnes viděli:
  1. Trace všechno (tagy, metadata, threads)
  2. Měř kvalitu offline před deployem (golden dataset + evaluators)
  3. Verzuj prompty separátně od kódu (Hub + dev/prod tagy)
  4. A/B testuj změny promptu, ne deploy a modli se
  5. PII redakce hned od první trace, ne až když přijde audit
- Q&A, anketa: co byste chtěli na navazující sezení (LangGraph hlubší, RAG eval, monitoring v produkci, annotation queue + human feedback…)

---

## Záložní obsah, pokud zbude čas

- `seed_annotation_queue.py` + `promote_annotations_to_dataset.py` — feedback loop, jak rostou datasety z reálných produkčních traces
- `eval_human_agreement.py` — jak validovat, že LLM-judge souhlasí s člověkem
- `chat_app.py` — interaktivní CLI, ať si účastníci pohrají s vlastními dotazy

## Co si účastníci odnášejí

- Funkční repo s celou pipeline (agent + evals + CI gate) — můžou si ho forknout pro vlastní use-case
- Pochopení, proč LangSmith není "jen logger", ale infrastruktura pro iteraci LLM aplikací
- Konkrétní checklist pro vlastní LLM projekt: *trace, evals, prompt versioning, PII, CI gate*
