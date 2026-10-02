"""
InsightForge — AI-Powered Business Intelligence Assistant
Streamlit UI (Part 2, Step 7).

Run with (from project root):
    streamlit run app.py
or directly:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("error", message=".*use_container_width.*")

import sys
from pathlib import Path

# Make `src` importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.data.loader import load_dataset
from src.evaluation.evaluator import (
    load_eval_dataset, run_evaluation, summarize_results, results_to_dicts,
)
from src.pipeline import build_app
from src.visualization import plots

st.set_page_config(
    page_title="InsightForge | AI Business Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached, expensive setup
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and building the knowledge base...")
def get_app(data_signature: str):
    return build_app()


@st.cache_data(show_spinner=False)
def _filter_options(df: pd.DataFrame):
    return {
        "regions": sorted(df["region"].unique()),
        "categories": sorted(df["product_category"].unique()),
        "segments": sorted(df["customer_segment"].unique()),
    }


settings = get_settings()
insight_app = get_app(str(settings.data_path))
df_full = insight_app.raw_data
options = _filter_options(df_full)

# ---------------------------------------------------------------------------
# Hero Banner (compact, non-clipping, aligned flush with the sidebar's top
# edge; badges stay on a single row regardless of viewport width)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Streamlit renders a fixed/sticky toolbar (hamburger menu + Deploy
       button) above the page content. If the content area's top padding
       is smaller than that toolbar's height, content renders underneath
       it and gets visually clipped. We give the MAIN content area just
       enough clearance to sit fully below the toolbar, and give the
       SIDEBAR the same top offset so both columns start flush/aligned
       with each other for a uniform look. */
    div[data-testid="stAppViewContainer"] > .main .block-container {
        padding-top: 2.6rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2.6rem !important;
    }
    /* Tighten vertical rhythm between the banner, KPI row, and tabs so the
       chat input is still visible on the home page without scrolling. */
    div[data-testid="stMetric"] {
        padding: 0.15rem 0 !important;
    }
    hr {
        margin: 0.5rem 0 !important;
    }
    .insightforge-banner {
        background: linear-gradient(135deg, #0F2C59 0%, #1F3864 45%, #2E5C99 100%);
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.7rem;
        box-shadow: 0 3px 10px rgba(15, 44, 89, 0.22);
        width: 100%;
        box-sizing: border-box;
    }
    .insightforge-banner h1 {
        color: #FFFFFF;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0 0 0.15rem 0;
        line-height: 1.25;
    }
    .insightforge-banner .subtitle {
        color: #E8EFFC;
        font-size: 0.68rem;
        font-weight: 500;
        margin: 0 0 0.3rem 0;
        line-height: 1.4;
    }
    .insightforge-banner .trendsetter {
        color: #FFD166;
        font-size: 0.7rem;
        font-weight: 700;
        font-style: italic;
        margin: 0 0 0.35rem 0;
        line-height: 1.4;
        border-left: 3px solid #FFD166;
        padding-left: 0.55rem;
    }
    .insightforge-badges {
        display: flex;
        flex-wrap: nowrap;
        justify-content: space-between;
        gap: 0.35rem;
        width: 100%;
        overflow-x: auto;
        scrollbar-width: none;
    }
    .insightforge-badges::-webkit-scrollbar {
        display: none;
    }
    .insightforge-badges span {
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.38);
        color: #FFFFFF;
        font-size: 0.58rem;
        font-weight: 600;
        padding: 0.16rem 0.55rem;
        border-radius: 999px;
        white-space: nowrap;
        flex: 0 0 auto;
    }
    </style>

    <div class="insightforge-banner">
        <h1>📊 InsightForge: AI-Powered Business Intelligence Assistant</h1>
        <p class="subtitle">
            Solving the enterprise problem of <strong>data-rich, insight-poor</strong> organizations —
            turning scattered sales, product, regional, and customer data into instant,
            trustworthy, plain-language answers, without a dedicated BI team, SQL skills,
            or an expensive analytics license.
        </p>
        <p class="trendsetter">
            🚀 Poised to be a trendsetter in the current market — transforming unutilized enterprise data
            into faster decisions, lower cost-to-insight, and a radically simpler user experience.
        </p>
        <div class="insightforge-badges">
            <span>⚡ Faster Speed to Decision</span>
            <span>💰 Cost Optimization</span>
            <span>🖱️ Extremely User-Friendly</span>
            <span>📈 Scalable</span>
            <span>🛡️ Reliable</span>
            <span>🧩 Extensible</span>
            <span>🏎️ Fast</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 InsightForge")
    st.caption("AI-Powered Business Intelligence Assistant")

    mode = insight_app.chain.mode
    if mode == "llm":
        st.success(f"LLM mode active ({settings.insightforge_llm_provider})")
    else:
        st.info(
            "Running in **stats-only / offline mode** — all numbers below are "
            "computed directly from data. Set `INSIGHTFORGE_LLM_PROVIDER` and "
            "an API key in `.env` to enable full generative narratives."
        )

    st.divider()
    st.subheader("Filters")
    sel_regions = st.multiselect("Region", options["regions"], default=options["regions"])
    sel_categories = st.multiselect("Product category", options["categories"], default=options["categories"])
    sel_segments = st.multiselect("Customer segment", options["segments"], default=options["segments"])

    min_d, max_d = df_full["order_date"].min().date(), df_full["order_date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    st.divider()
    st.caption(f"Dataset: {settings.data_path.name} · {len(df_full):,} transaction rows")

# Apply filters (view-level only; the RAG chain always answers over full data
# so retrieved statistics remain consistent with the narrative answers).
mask = (
    df_full["region"].isin(sel_regions)
    & df_full["product_category"].isin(sel_categories)
    & df_full["customer_segment"].isin(sel_segments)
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask &= (df_full["order_date"].dt.date >= start) & (df_full["order_date"].dt.date <= end)
df_view = df_full[mask]

from src.analysis.summary_stats import build_full_analysis
view_analysis = build_full_analysis(df_view) if len(df_view) else insight_app.analysis

# ---------------------------------------------------------------------------
# KPI Header
# ---------------------------------------------------------------------------
st.caption(
    "Ask questions in plain language and get grounded, cited answers backed "
    "by real statistical analysis of your business data."
)

totals = view_analysis["totals"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"${totals['total_revenue']:,.0f}")
k2.metric("Total Profit", f"${totals['total_profit']:,.0f}")
k3.metric("Margin", f"{totals['overall_margin_pct']:.1f}%")
k4.metric("Orders", f"{totals['total_orders']:,}")
k5.metric("Customers", f"{totals['total_customers']:,}")

st.divider()

tab_chat, tab_dashboard, tab_eval, tab_data = st.tabs(
    ["💬 Ask InsightForge", "📈 Dashboards", "🧪 Model Evaluation", "🗂️ Data Explorer"]
)

# ---------------------------------------------------------------------------
# Tab 1: Chat / RAG assistant
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask a business question")
    st.caption(
        "Examples: *\"How did revenue trend over the last few quarters?\"*, "
        "*\"Which product category is most profitable?\"*, "
        "*\"What's the standard deviation of unit price?\"*"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander(f"View {len(turn['facts'])} retrieved source facts"):
                for f in turn["facts"]:
                    st.caption(f"**[{f.category}]** (score {f.score}) {f.text}")

    question = st.chat_input("Ask InsightForge about your business data...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving facts and composing an answer..."):
                response = insight_app.chain.ask(question)
            st.write(response.answer)
            with st.expander(f"View {len(response.retrieved_facts)} retrieved source facts"):
                for f in response.retrieved_facts:
                    st.caption(f"**[{f.category}]** (score {f.score}) {f.text}")
        st.session_state.chat_history.append({
            "question": question, "answer": response.answer, "facts": response.retrieved_facts,
        })

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            insight_app.chain.memory.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Visual dashboards
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Sales Trends")
    st.plotly_chart(plots.sales_trend_chart(view_analysis["monthly_trend"]), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.product_performance_chart(view_analysis["top_products"]), use_container_width=True)
    with c2:
        st.plotly_chart(plots.category_share_chart(view_analysis["category_performance"]), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plots.regional_analysis_chart(view_analysis["regional_analysis"]), use_container_width=True)
    with c4:
        st.plotly_chart(plots.customer_segmentation_chart(view_analysis["customer_segmentation"]), use_container_width=True)

    st.subheader("Statistical Distributions")
    metric_choice = st.selectbox("Metric", ["revenue", "profit", "unit_price", "units_sold"])
    st.plotly_chart(plots.statistical_distribution_chart(df_view if len(df_view) else df_full, metric_choice), use_container_width=True)

    with st.expander("📐 Full statistical summary table"):
        stats_df = pd.DataFrame(view_analysis["statistical_summary"]).T
        st.dataframe(stats_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: LLMOps — model evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Model Evaluation (QAEvalChain-style grading)")
    st.caption(
        "Runs a curated set of question/reference-answer pairs through the "
        "live RAG chain and grades each prediction for factual accuracy — "
        "using LangChain's `QAEvalChain` when an LLM is configured, or a "
        "deterministic numeric-grounding grader in offline mode."
    )

    eval_path = PROJECT_ROOT / "eval" / "eval_dataset.json"
    if st.button("▶ Run evaluation suite", type="primary"):
        with st.spinner("Running evaluation..."):
            cases = load_eval_dataset(eval_path)
            results = run_evaluation(insight_app.chain, cases)
            summary = summarize_results(results)
        st.session_state["eval_results"] = results
        st.session_state["eval_summary"] = summary

    if "eval_summary" in st.session_state:
        summary = st.session_state["eval_summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{summary['accuracy_pct']:.1f}%")
        c2.metric("Correct", summary["correct"])
        c3.metric("Incorrect", summary["incorrect"])
        c4.metric("Evaluator", summary["evaluator"])

        st.dataframe(pd.DataFrame(results_to_dicts(st.session_state["eval_results"])), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Raw data explorer
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Filtered transaction data")
    st.caption(f"{len(df_view):,} of {len(df_full):,} rows match the current sidebar filters.")
    st.dataframe(df_view, use_container_width=True, height=460)
    st.download_button(
        "⬇ Download filtered data as CSV",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="insightforge_filtered_data.csv",
        mime="text/csv",
    )
