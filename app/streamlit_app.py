"""
InsightForge — AI-Powered Business Intelligence Assistant
Streamlit UI (Part 2, Step 7) — fixed two-panel layout.

Layout: a LEFT control panel (title, mode note, info links, filters) and a RIGHT content panel
(banner, KPIs, tabs). Both panels are always visible: there is no collapsible sidebar and no
collapse arrow, so the page looks the same in Chrome, Edge and at any zoom level.

Run with (from project root):
    streamlit run app.py
or directly:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

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
# Global styling
# ---------------------------------------------------------------------------
# The left panel is an ordinary keyed container (key="left_panel"). Streamlit adds the CSS class
# "st-key-left_panel" to it, which is what the rules below target.
st.markdown(
    """
    <style>
    /* Hide Streamlit's own top chrome (menu, toolbar, decoration, deploy button). This is safe here
       because the app has no sidebar, so nothing needs the "reopen sidebar" button that lives in the toolbar. */
    #MainMenu,
    header[data-testid="stHeader"],
    header.stAppHeader,
    div[data-testid="stDecoration"],
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stAppDeployButton"],
    .stAppDeployButton,
    div[data-testid="stToolbarActions"] {
        display: none !important;
        height: 0 !important;
        visibility: hidden !important;
    }
    /* Collapse the top padding that was reserved for the (now hidden) header. */
    div[data-testid="stAppViewContainer"] > .main .block-container,
    div[data-testid="stAppViewBlockContainer"],
    section.stMain .block-container,
    section[data-testid="stMain"] .block-container,
    .main .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    div[data-testid="stAppViewContainer"] .block-container > div:first-child {
        margin-top: 0 !important;
    }
    div[data-testid="stMetric"] { padding: 0.15rem 0 !important; }
    hr { margin: 0.5rem 0 !important; }

    /* ---------------- LEFT PANEL ---------------- */
    .st-key-left_panel {
        background: #F0F2F6;
        border: 1px solid #DCE3EE;
        border-radius: 12px;
        padding: 0.9rem 1rem 1rem 1rem;
    }
    .st-key-left_panel,
    .st-key-left_panel div[data-testid="stVerticalBlock"] {
        gap: 0.15rem !important;
    }
    .st-key-left_panel div[data-testid="stAlert"] {
        padding: 0.35rem 0.6rem !important;
        margin-bottom: 0.4rem !important;
    }
    .st-key-left_panel div[data-testid="stAlert"] p {
        font-size: 0.78rem !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }
    /* "Learn more" info links: clean, italic hyperlinks instead of boxed buttons. */
    .st-key-left_panel div[data-testid="stButton"] > button {
        display: block !important;
        background: transparent;
        border: none;
        color: #1F3864;
        font-weight: 600;
        font-style: italic !important;
        font-size: 0.92rem;
        text-align: left !important;
        padding: 0.2rem 0;
        margin: 0;
        width: 100%;
        box-shadow: none;
    }
    .st-key-left_panel div[data-testid="stButton"] > button div,
    .st-key-left_panel div[data-testid="stButton"] > button p,
    .st-key-left_panel div[data-testid="stButton"] > button span {
        text-align: left !important;
        justify-content: flex-start !important;
        display: block !important;
        width: 100%;
        font-style: italic !important;
    }
    .st-key-left_panel div[data-testid="stButton"] {
        margin-bottom: -1.1rem !important;
    }
    .st-key-left_panel div[data-testid="stButton"] > button:hover {
        color: #2E5C99;
        text-decoration: underline;
        background: transparent;
    }
    .st-key-left_panel div[data-testid="stButton"] > button:focus:not(:active) {
        color: #1F3864;
        border: none;
        box-shadow: none;
    }
    .st-key-left_panel h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Popup (st.dialog) modals: cap the width and scroll long content. */
    div[role="dialog"] {
        max-width: 620px !important;
        width: 92vw !important;
        max-height: 80vh !important;
        overflow-y: auto !important;
        margin: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dialog (modal popup) content for the 5 info links in the left panel
# ---------------------------------------------------------------------------
@st.dialog("Why InsightForge?")
def show_why_insightforge():
    banner_path = PROJECT_ROOT / "assets" / "why_insightforge_banner.png"
    if banner_path.exists():
        st.image(str(banner_path), width='stretch')
    st.markdown(
        """
        <div style="background:#EEF3FA;border-left:4px solid #2E5C99;padding:0.6rem 0.9rem;
                    border-radius:8px;margin-top:0.6rem;">
        <span style="color:#1F3864;font-weight:800;font-size:0.95rem;">The Problem We Solve</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
Organizations — especially small and medium-sized enterprises — accumulate meaningful volumes
of operational and sales data but consistently struggle to convert it into decisions. This gap
has three root causes:
- **Resource scarcity** — dedicated BI analysts, data engineers, and enterprise BI tooling
  (Power BI Premium, Tableau, Qlik) are cost-prohibitive for many organizations.
- **Skills gap** — business stakeholders can describe the question they want answered, but
  rarely know SQL or how to build a pivot table or dashboard themselves.
- **AI trust gap** — generic LLM chatbots readily answer business questions but frequently
  hallucinate numbers, making them unsafe for financial or operational decisions without
  additional grounding.
        """
    )
    st.markdown(
        """
        <div style="background:#FBF0DF;border-left:4px solid #B45309;padding:0.6rem 0.9rem;
                    border-radius:8px;margin-top:0.5rem;font-size:0.9rem;">
        <span style="color:#B45309;font-weight:700;">Problem statement:</span>
        <span style="color:#2B2B2B;"> &ldquo;Data-rich, insight-poor&rdquo; organizations need a trustworthy,
        self-service way to ask natural-language questions about their business data and receive
        numerically accurate, explainable answers.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.dialog("Business Benefits")
def show_business_benefits():
    st.markdown(
        """
        <div style="color:#1F3864;font-weight:800;font-size:1rem;margin-bottom:0.3rem;">
        What Makes InsightForge Unique
        </div>
        <p style="color:#2B2B2B;font-size:0.9rem;">
        Unlike generic AI chatbots, InsightForge is built on a <strong>fact-document RAG
        architecture</strong> — every statistic is pre-computed deterministically and the AI is
        only ever allowed to answer from verified, retrieved facts. This structurally eliminates
        the hallucinated-numbers problem that makes most AI assistants unsafe for business
        decisions.
        </p>
        """,
        unsafe_allow_html=True,
    )
    benefits = [
        ("💰", "Cost Avoidance", "No dedicated BI analyst headcount or enterprise BI license required to get started."),
        ("⚡", "Speed to Decision", "Answers generated in seconds instead of days waiting on an analyst."),
        ("🛡️", "Risk Reduction", "An automated evaluation harness continuously checks answer accuracy, reducing the risk of silent AI quality drift."),
        ("🧩", "Resilience by Design", "Runs in a fully deterministic offline mode at zero API cost when AI budget or network access is unavailable — value starts on day one."),
        ("📈", "Extensibility", "The retriever, LLM provider, and vector store are all pluggable; scales from a zero-cost pilot to full production without a redesign."),
    ]
    for icon, title, desc in benefits:
        st.markdown(
            f"""
            <div style="display:flex;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid #E5E9F0;">
                <div style="font-size:1.1rem;">{icon}</div>
                <div>
                    <div style="color:#1F3864;font-weight:700;font-size:0.9rem;">{title}</div>
                    <div style="color:#444;font-size:0.84rem;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

@st.dialog("How secure is it?")
def show_security():
    st.markdown(
        """
        <div style="color:#1F3864;font-weight:800;font-size:1rem;margin-bottom:0.4rem;">
        Security & Governance Protocols
        </div>
        """,
        unsafe_allow_html=True,
    )
    controls = [
        ("🔑", "Secrets Management", "API keys load exclusively via .env / environment variables; .env is git-ignored and never committed to source control."),
        ("🕵️", "Data Governance", "No raw customer PII in the standard schema — only pseudonymous customer IDs are used."),
        ("🧠", "Anti-Hallucination Control", "Prompts explicitly instruct the model to use ONLY retrieved, verified facts — never invented numbers."),
        ("✅", "Model Evaluation Gate", "An automated QAEvalChain / heuristic accuracy gate can block any deployment that falls below an agreed accuracy threshold."),
        ("🌐", "Network Resilience", "Falls back gracefully to a fully offline mode rather than crashing if outbound AI calls are blocked by a corporate firewall."),
        ("📜", "Auditability", "Every chat answer displays its retrieved source facts, so any number can be traced back to its origin."),
        ("📦", "Dependency Hygiene", "All package versions are pinned to reduce supply-chain risk."),
        ("🗂️", "Data Retention", "Conversation memory is session-only — nothing is persisted to disk or a database by default."),
    ]
    for icon, title, desc in controls:
        st.markdown(
            f"""
            <div style="display:flex;gap:0.6rem;padding:0.35rem 0;border-bottom:1px solid #E5E9F0;">
                <div style="font-size:1.05rem;">{icon}</div>
                <div>
                    <div style="color:#1F3864;font-weight:700;font-size:0.88rem;">{title}</div>
                    <div style="color:#444;font-size:0.82rem;">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div style="background:#FBEAEA;border-left:4px solid #B3261E;padding:0.6rem 0.9rem;
                    border-radius:8px;margin-top:0.6rem;font-size:0.88rem;">
        <span style="color:#B3261E;font-weight:700;">Transparency note:</span>
        <span style="color:#2B2B2B;"> Single sign-on (SSO) and role-based access control (RBAC)
        are not yet built in — see <strong>Future Roadmap</strong> for the planned rollout.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.dialog("Future Roadmap")
def show_roadmap():
    st.markdown(
        """
        <div style="color:#1F3864;font-weight:800;font-size:1rem;margin-bottom:0.4rem;">
        Where InsightForge Is Headed
        </div>
        <p style="color:#2B2B2B;font-size:0.86rem;">
        Aligned with industry direction toward agentic, enterprise-governed AI copilots embedded
        directly into business workflows.
        </p>
        """,
        unsafe_allow_html=True,
    )
    phases = [
        ("Phase 1 — Today", "#2E7D32", "Offline-first pilot with fact-grounded RAG, automated evaluation harness, and interactive dashboards — zero AI cost to prove value."),
        ("Phase 2 — Next", "#2E5C99", "Enterprise identity integration: Single Sign-On (SSO via Azure AD / Entra ID) and Role-Based Access Control (RBAC) to scope data access by department or role."),
        ("Phase 3 — Near-Term", "#B45309", "Multi-LLM orchestration (OpenAI, Anthropic, Azure OpenAI) and agentic multi-step reasoning (LangGraph) for deeper, multi-hop business questions."),
        ("Phase 4 — Scale-Out", "#6D28D9", "Multi-tenant, containerized deployment, horizontal scaling, upgraded vector store (pgvector / Pinecone), and proactive insights such as anomaly detection and forecasting."),
    ]
    for title, color, desc in phases:
        st.markdown(
            f"""
            <div style="border-left:4px solid {color};padding:0.4rem 0.8rem;margin-bottom:0.45rem;background:#F7F9FC;border-radius:6px;">
                <div style="color:{color};font-weight:800;font-size:0.88rem;">{title}</div>
                <div style="color:#2B2B2B;font-size:0.84rem;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Industry alignment: embedded copilot-style assistants, agentic AI workflows, and "
        "responsible-AI governance are becoming standard expectations across enterprise SaaS — "
        "this roadmap keeps InsightForge ahead of that curve rather than catching up to it."
    )

@st.dialog("Impact or Risk – Mitigation Strategy")
def show_risk_mitigation():
    st.markdown(
        """
        <div style="color:#1F3864;font-weight:800;font-size:1rem;margin-bottom:0.4rem;">
        Risk Register & Mitigation Strategy
        </div>
        """,
        unsafe_allow_html=True,
    )
    risks = [
        ("No authentication layer yet", "Unauthorized access if deployed broadly", "SSO + RBAC on the roadmap; restrict to a trusted pilot network in the meantime."),
        ("LLM API / network dependency", "Degraded generative experience during an outage", "Automatic fallback to deterministic offline mode — core Q&A keeps working with zero downtime."),
        ("Cost overrun at scale", "Budget risk from uncontrolled LLM usage", "Per-query cost monitoring and rate-limiting planned; offline mode avoids cost entirely during pilot."),
        ("Data quality / PII exposure", "Compliance risk if real customer data is ingested", "Schema validation, pseudonymous IDs only, and a mandatory DLP review before any PII ingestion."),
        ("Model hallucination", "Incorrect business decisions from fabricated figures", "Fact-document RAG architecture plus an automated accuracy gate rejects ungrounded answers."),
        ("Vendor / model lock-in", "Reduced flexibility if tied to one LLM provider", "Provider-abstraction layer allows swapping OpenAI / Anthropic / Azure with a config change only."),
    ]
    for risk, impact, mitigation in risks:
        st.markdown(
            f"""
            <div style="padding:0.5rem 0;border-bottom:1px solid #E5E9F0;">
                <div style="color:#B3261E;font-weight:700;font-size:0.86rem;">⚠ {risk}</div>
                <div style="color:#555;font-size:0.8rem;margin:0.15rem 0;"><strong>Impact:</strong> {impact}</div>
                <div style="color:#2E7D32;font-size:0.8rem;"><strong>Mitigation:</strong> {mitigation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Two fixed panels: LEFT = controls, RIGHT = content
# ---------------------------------------------------------------------------
left, right = st.columns([1, 3.4], gap="medium")

# ============================== LEFT PANEL ==============================
with left:
    with st.container(key="left_panel"):
        st.markdown(
            '<div style="font-size:2.2rem; font-weight:700; margin:0; line-height:1.1;">'
            '📊 InsightForge</div>'
            '<div style="font-size:0.85rem; color:#6B7280; font-weight:500; '
            'margin:0.3rem 0 0.7rem 0; line-height:1.3;">'
            'AI-Powered Business Intelligence Assistant</div>',
            unsafe_allow_html=True,
        )
        mode = insight_app.chain.mode
        if mode == "llm":
            st.success(f"✅ Smart Mode Active ({settings.insightforge_llm_provider})")
        else:
            st.markdown(
                '<div style="background-color:#e7f3fe; padding:0.75rem 1rem; border-radius:0.5rem; '
                'border-left:4px solid #2196F3; font-size:0.7rem; text-align:justify; '
                'margin-bottom:0.8rem;">'
                '⚡ <strong>Live Mode</strong> — Ask anything and get an instant, accurate '
                'answer calculated directly from your real data (no AI subscription needed '
                'to get started). Use the hyperlinks below to learn more about InsightForge '
                'and how it works. Also use the filters below to narrow down the data.'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div style="font-weight:700; margin:0.5rem 0 0.4rem 0;">Learn more:</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔍 Why InsightForge?", key="link_why", width='stretch'):
            show_why_insightforge()
        if st.button("💼 Business Benefits", key="link_benefits", width='stretch'):
            show_business_benefits()
        if st.button("🔐 How secure is it?", key="link_security", width='stretch'):
            show_security()
        if st.button("🗺️ Future Roadmap", key="link_roadmap", width='stretch'):
            show_roadmap()
        if st.button("⚖️ Impact or Risk - Mitigation Strategy", key="link_risk", width='stretch'):
            show_risk_mitigation()
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

# ============================== RIGHT PANEL ==============================
with right:
    # ---- Hero banner ----
    st.markdown(
        """
        <style>
        .insightforge-banner {
            background: linear-gradient(135deg, #0F2C59 0%, #1F3864 45%, #2E5C99 100%);
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            margin-top: 0;
            margin-bottom: 0.7rem;
            box-shadow: 0 3px 10px rgba(15, 44, 89, 0.22);
            width: 100%;
            box-sizing: border-box;
        }
        .insightforge-banner h1 {
            color: #FFFFFF;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 0 0 0.2rem 0;
            line-height: 1.25;
            padding: 0;
        }
        .insightforge-banner .subtitle {
            color: #E8EFFC;
            font-size: 0.7rem;
            font-weight: 500;
            margin: 0 0 0.35rem 0;
            line-height: 1.4;
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
        .insightforge-badges::-webkit-scrollbar { display: none; }
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
                InsightForge turns fragmented enterprise data into instant, AI-grounded answers —
                eliminating the need for dedicated analysts, SQL skills, or costly BI licenses.
                <strong>A trendsetter by design</strong>, it accelerates decisions, cuts cost-to-insight,
                and delivers secure, scalable, enterprise-grade intelligence to every business user.
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

    # ---- KPI header ----
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

    # -----------------------------------------------------------------------
    # Tab 1: Chat / RAG assistant
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Tab 2: Visual dashboards
    # -----------------------------------------------------------------------
    with tab_dashboard:
        st.subheader("Sales Trends")
        st.plotly_chart(plots.sales_trend_chart(view_analysis["monthly_trend"]), width='stretch')
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plots.product_performance_chart(view_analysis["top_products"]), width='stretch')
        with c2:
            st.plotly_chart(plots.category_share_chart(view_analysis["category_performance"]), width='stretch')
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plots.regional_analysis_chart(view_analysis["regional_analysis"]), width='stretch')
        with c4:
            st.plotly_chart(plots.customer_segmentation_chart(view_analysis["customer_segmentation"]), width='stretch')
        st.subheader("Statistical Distributions")
        metric_choice = st.selectbox("Metric", ["revenue", "profit", "unit_price", "units_sold"])
        st.plotly_chart(plots.statistical_distribution_chart(df_view if len(df_view) else df_full, metric_choice), width='stretch')
        with st.expander("📐 Full statistical summary table"):
            stats_df = pd.DataFrame(view_analysis["statistical_summary"]).T
            st.dataframe(stats_df, width='stretch')

    # -----------------------------------------------------------------------
    # Tab 3: LLMOps — model evaluation
    # -----------------------------------------------------------------------
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
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy", f"{summary['accuracy_pct']:.1f}%")
            e2.metric("Correct", summary["correct"])
            e3.metric("Incorrect", summary["incorrect"])
            e4.metric("Evaluator", summary["evaluator"])
            st.dataframe(pd.DataFrame(results_to_dicts(st.session_state["eval_results"])), width='stretch')

    # -----------------------------------------------------------------------
    # Tab 4: Raw data explorer
    # -----------------------------------------------------------------------
    with tab_data:
        st.subheader("Filtered transaction data")
        st.caption(f"{len(df_view):,} of {len(df_full):,} rows match the current filters.")
        st.dataframe(df_view, width='stretch', height=460)
        st.download_button(
            "⬇ Download filtered data as CSV",
            data=df_view.to_csv(index=False).encode("utf-8"),
            file_name="insightforge_filtered_data.csv",
            mime="text/csv",
        )
