# InsightForge

**AI-Powered Business Intelligence Assistant** — an automated, RAG-grounded BI
assistant that analyzes sales data, generates natural-language insights, and
visualizes findings for small and medium-sized enterprises that don't have
the budget for traditional BI tooling.

Built to the two-part brief:
- **Part 1** — AI-Powered BI Assistant (data prep, knowledge base, LLM
  application with a custom retriever, prompt chaining, RAG, and memory).
- **Part 2** — LLMOps (QAEvalChain-style model evaluation, data
  visualization, and a Streamlit UI).

---

## 1. Why this architecture

| Business problem | Design decision |
|---|---|
| SMEs can't afford dedicated BI analysts | Natural-language Q&A over the data — no SQL, no dashboards to learn. |
| LLMs hallucinate numbers | **Fact-document RAG**: every statistic is pre-computed with pandas, rendered as a short factual sentence, and the LLM is only allowed to answer from retrieved facts (never asked to "do math" itself). |
| Teams may not have LLM API budget on day one | The whole pipeline (retrieval, chaining, memory, evaluation, dashboards) runs in a **fully offline "extractive" mode** with zero API keys — flip two env vars to upgrade to full generative narratives later. |
| Need to trust the assistant's answers | Built-in **evaluation harness** (`QAEvalChain`-equivalent) grades every answer against a reference set on every change. |
| Non-technical stakeholders need visuals | Streamlit dashboard with trend, product, regional, and segmentation charts, all sourced from the same analytics engine the chatbot uses (numbers can never disagree). |

## 2. Project layout

```
InsightForge/
├── app.py                       # ROOT ENTRY POINT — `streamlit run app.py`
├── app/
│   └── streamlit_app.py         # Part 2 Step 7: the actual UI implementation
├── src/
│   ├── config.py                 # env-driven settings (pydantic-settings)
│   ├── pipeline.py                # single orchestration entry point
│   ├── data/
│   │   ├── generate_sample_data.py  # synthetic SME sales dataset generator
│   │   └── loader.py                # Part 1 Step 2: load + structure data
│   ├── analysis/
│   │   └── summary_stats.py      # Part 1 Step 3: advanced data summary engine
│   ├── rag/
│   │   ├── fact_documents.py     # turns analytics into retrievable "facts"
│   │   ├── retriever.py          # Part 1 Step 3: custom retriever (BM25 + optional embeddings)
│   │   ├── prompts.py            # Part 1 Step 4: prompt engineering + chaining
│   │   ├── memory.py             # Part 1 Step 6: conversational memory
│   │   ├── llm_factory.py        # provider-agnostic LLM instantiation
│   │   └── chain.py              # Part 1 Step 5: RAG chain assembly (LCEL-ready)
│   ├── evaluation/
│   │   └── evaluator.py          # Part 2 Step 7: QAEvalChain-style evaluation
│   └── visualization/
│       └── plots.py              # Part 2 Step 7: Plotly chart library
├── eval/
│   └── eval_dataset.json         # reference Q&A pairs for evaluation
├── data/raw/sales_data.csv       # generated sample dataset (gitignored in prod)
├── tests/                        # pytest unit/integration tests
├── requirements.txt
└── .env.example
```

## 3. Quick start

```bash
# 1. Create a virtual environment & install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate the sample dataset (or point INSIGHTFORGE_DATA_PATH at your own CSV)
python -m src.data.generate_sample_data

# 3. (Optional) Configure an LLM provider for full generative narratives
cp .env.example .env
# edit .env: set INSIGHTFORGE_LLM_PROVIDER=openai and OPENAI_API_KEY=sk-...
# Without this step, InsightForge runs in offline/extractive mode automatically.

# 4. Run the tests
pytest tests/ -v

# 5. Launch the app — root-level entry point
streamlit run app.py
```

`streamlit run app.py` and `streamlit run app/streamlit_app.py` launch the
identical application; `app.py` is a thin delegator kept at the project
root purely for command-line discoverability/convention (many platforms
and reviewers default to looking for `app.py`).

Bring your own data by pointing `INSIGHTFORGE_DATA_PATH` at a CSV with these
columns (see `src/data/loader.py::REQUIRED_COLUMNS`):

```
order_id, order_date, region, country, product_category, product_name,
customer_id, customer_segment, age_group, units_sold, unit_price,
revenue, unit_cost, profit
```

## 4. How the RAG pipeline works

```
 CSV data
    │
    ▼
 load_dataset()              (src/data/loader.py)        — structuring, not cleaning
    │
    ▼
 build_full_analysis()       (src/analysis/summary_stats.py)
    │   ├─ sales_by_period()          → time-series trend + growth %
    │   ├─ product_performance() /
    │   │   category_performance()   → product & category rankings
    │   ├─ regional_analysis()        → country/region breakdown
    │   ├─ customer_segmentation()    → segment x demographic breakdown
    │   └─ descriptive_statistics()   → mean/median/std/variance/skew
    │
    ▼
 build_fact_documents()      (src/rag/fact_documents.py)
    │   turns every statistic into a short, precise natural-language fact
    ▼
 get_retriever()              (src/rag/retriever.py)
    │   BM25 keyword retriever (always available) or
    │   Chroma + OpenAI embeddings retriever (when configured)
    ▼
 InsightForgeRAGChain.ask()   (src/rag/chain.py)
    │   1. retrieve top-k relevant facts for the question
    │   2. ANALYSIS_PROMPT  → grounded structured draft   (src/rag/prompts.py)
    │   3. NARRATIVE_PROMPT → polished, memory-aware answer
    │   (in offline mode, steps 2-3 are replaced by a deterministic,
    │    template-based render of the retrieved facts — no LLM needed)
    ▼
 ConversationMemory           (src/rag/memory.py)
    │   stores the turn so follow-up questions have context
    ▼
 Streamlit UI                 (app.py -> app/streamlit_app.py)
     chat tab · dashboards tab · evaluation tab · data explorer tab
```

## 5. Evaluation (LLMOps)

`src/evaluation/evaluator.py` implements the brief's "Apply QAEvalChain to
assess the model's performance and accuracy" requirement with two backends:

- **`LangChainQAEvaluator`** — wraps `langchain.evaluation.qa.QAEvalChain`
  and uses an LLM-as-judge to grade correctness (used automatically once an
  LLM provider is configured).
- **`HeuristicEvaluator`** — a deterministic, zero-cost numeric-grounding
  grader used as the default/offline backend and in the test suite, so
  evaluation results are always reproducible in CI.

Run it from the CLI:

```bash
python scripts/run_evaluation.py
```

or from the **🧪 Model Evaluation** tab in the Streamlit app.

## 6. Modern stack rationale (2026)

- **LangChain 1.x + LCEL / LangGraph** for chain composition and agentic
  extensibility — the current production standard as of the 1.0 stable
  line (Oct 2025) and its 1.2+/1.4+ point releases.
- **Chroma** as the default lightweight vector store, with `langchain-openai`
  embeddings — swappable for FAISS/pgvector/Pinecone with no code change
  outside `retriever.py`.
- **Streamlit** for the UI: the fastest path from a Python analytics script
  to an interactive internal tool, with native `st.chat_message` /
  `st.session_state` support for conversational apps.
- **Plotly** for all charts: interactive, Streamlit-native, no headless
  browser/rendering step required.
- **pydantic-settings** for typed, `.env`-driven configuration instead of
  scattered `os.environ` calls.

## 7. Testing

```bash
pytest tests/ -v
```

Tests cover: statistical correctness (revenue/margin math, percentile sums,
skewness sanity checks), data loading/validation, the custom BM25 retriever
(relevance + category filtering + graceful fallback), the full RAG chain in
offline mode, conversational memory windowing, the evaluation harness
(including a deliberately-wrong prediction to confirm the grader actually
fails bad answers), and the root-level `app.py` entry point wiring itself.

All tests pass with **zero external API calls or network access**, which is
also how this project was verified end-to-end.
