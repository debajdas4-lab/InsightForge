from __future__ import annotations

from src.analysis.summary_stats import build_full_analysis
from src.rag.fact_documents import build_fact_documents
from src.rag.retriever import KeywordFactRetriever


def test_fact_documents_are_built_for_every_category(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    categories = {d.category for d in docs}
    assert {"trend", "product", "region", "segment", "statistics"}.issubset(categories)
    assert len(docs) > 10


def test_keyword_retriever_returns_relevant_statistics_facts(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    retriever = KeywordFactRetriever(docs)

    results = retriever.retrieve("what is the standard deviation of unit price", top_k=3)
    assert results
    assert any("unit_price" in r.text or "unit price" in r.text.lower() for r in results)
    assert all(r.category in {"statistics", "trend", "product", "region", "segment"} for r in results)


def test_keyword_retriever_category_filter(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    retriever = KeywordFactRetriever(docs)

    results = retriever.retrieve("revenue", top_k=5, category="region")
    assert results
    assert all(r.category == "region" for r in results)


def test_keyword_retriever_never_returns_empty_for_vague_query(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    retriever = KeywordFactRetriever(docs)

    results = retriever.retrieve("xyzblargh nonsense query", top_k=3)
    assert len(results) > 0  # falls back to statistics/trend facts
