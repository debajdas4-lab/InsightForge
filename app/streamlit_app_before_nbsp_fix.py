# ---------------------------------------------------------------------------
# Tab 2: Visual dashboards
# ---------------------------------------------------------------------------
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
st.dataframe(pd.DataFrame(results_to_dicts(st.session_state["eval_results"])), width='stretch')
 
# ---------------------------------------------------------------------------
# Tab 4: Raw data explorer
# ---------------------------------------------------------------------------
with tab_data:
st.subheader("Filtered transaction data")
st.caption(f"{len(df_view):,} of {len(df_full):,} rows match the current sidebar filters.")
st.dataframe(df_view, width='stretch', height=460)
st.download_button(
"⬇ Download filtered data as CSV",
data=df_view.to_csv(index=False).encode("utf-8"),
file_name="insightforge_filtered_data.csv",
mime="text/csv",
)