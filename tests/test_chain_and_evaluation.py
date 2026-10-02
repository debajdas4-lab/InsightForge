from __future__ import annotations

from pathlib import Path

from src.analysis.summary_stats import build_full_analysis
from src.config import Settings
from src.evaluation.evaluator import (
    HeuristicEvaluator, load_eval_dataset, run_evaluation, summarize_results,
)
from src.rag.chain import InsightForgeRAGChain
from src.rag.fact_documents import build_fact_documents
from src.rag.memory import ConversationMemory

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _offline_settings() -> Settings:
    return Settings(insightforge_llm_provider="none")


def test_chain_runs_in_extractive_mode_without_api_keys(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    chain = InsightForgeRAGChain(documents=docs, settings=_offline_settings())

    assert chain.mode == "extractive"
    response = chain.ask("Which product category performs best?")
    assert response.answer
    assert response.retrieved_facts


def test_chain_memory_accumulates_turns(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    memory = ConversationMemory(window=3)
    chain = InsightForgeRAGChain(documents=docs, settings=_offline_settings(), memory=memory)

    chain.ask("What is total revenue?")
    chain.ask("Which region is best?")
    assert len(memory.get_turns()) == 2


def test_conversation_memory_respects_window_size():
    memory = ConversationMemory(window=2)
    memory.add_turn("q1", "a1")
    memory.add_turn("q2", "a2")
    memory.add_turn("q3", "a3")
    turns = memory.get_turns()
    assert len(turns) == 2
    assert turns[0][0] == "q2"  # oldest turn dropped


def test_evaluation_harness_end_to_end_offline(sample_df):
    analysis = build_full_analysis(sample_df)
    docs = build_fact_documents(analysis)
    chain = InsightForgeRAGChain(documents=docs, settings=_offline_settings())

    cases = load_eval_dataset(PROJECT_ROOT / "eval" / "eval_dataset.json")
    results = run_evaluation(chain, cases, evaluator=HeuristicEvaluator())
    summary = summarize_results(results)

    assert summary["total_cases"] == len(cases)
    assert 0 <= summary["accuracy_pct"] <= 100
    assert summary["evaluator"] == "heuristic_numeric_overlap"


def test_heuristic_evaluator_flags_numeric_mismatch():
    evaluator = HeuristicEvaluator(min_overlap=0.9)
    result = evaluator.grade(
        question="What is total revenue?",
        reference="Total revenue is $500,000.",
        prediction="I think total revenue was roughly $12.",
        context=[],
    )
    assert result.grade == "INCORRECT"


def test_heuristic_evaluator_passes_matching_numbers():
    evaluator = HeuristicEvaluator(min_overlap=0.9)
    result = evaluator.grade(
        question="What is total revenue?",
        reference="Total revenue is $500,000.",
        prediction="Total revenue came in at $500,000 this year.",
        context=[],
    )
    assert result.grade == "CORRECT"
