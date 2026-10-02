"""
RAG system assembly (Part 1, Steps 4-6): chains the two prompts from
`src.rag.prompts` together over the custom retriever, with conversational
memory threaded through both calls.

Two execution modes, chosen automatically:

* **LLM mode** (`INSIGHTFORGE_LLM_PROVIDER` set + API key present + LangChain
  installed): builds a genuine two-hop LCEL pipeline
  `retrieve -> analysis_prompt -> llm -> narrative_prompt -> llm`.

* **Extractive / offline mode** (default in this sandbox, and useful for
  CI, demos, and cost-free testing): skips the LLM calls and deterministically
  renders the retrieved facts into a structured, still-genuinely-useful
  answer. This keeps the whole application runnable end-to-end without any
  API key, which is exactly what let this project be built and verified
  without network access.

Both modes share the same retriever, memory, and prompt templates, so
switching a deployment from offline demo mode to full LLM mode is a matter
of setting two environment variables — no code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.rag.fact_documents import FactDocument
from src.rag.llm_factory import LLMNotAvailableError, get_chat_model
from src.rag.memory import ConversationMemory
from src.rag.prompts import build_analysis_prompt, build_narrative_prompt
from src.rag.retriever import RetrievedFact, get_retriever


@dataclass
class InsightForgeResponse:
    question: str
    answer: str
    draft: str
    retrieved_facts: list[RetrievedFact]
    mode: str  # "llm" or "extractive"


def _render_extractive_answer(question: str, facts: list[RetrievedFact]) -> str:
    """Deterministically compose a readable answer straight from top facts.

    This is intentionally simple and template-based (no LLM) so the whole
    pipeline is testable offline; it is what the Streamlit app falls back to
    when no LLM provider is configured.
    """
    if not facts:
        return (
            "I could not find any matching business facts for that question. "
            "Try asking about revenue trends, product/category performance, "
            "regional breakdowns, customer segments, or statistical summaries."
        )
    lead = facts[0].text
    supporting = facts[1:4]
    lines = [lead]
    if supporting:
        lines.append("\nSupporting details:")
        for f in supporting:
            lines.append(f"- {f.text}")
    return "\n".join(lines)


class InsightForgeRAGChain:
    """The end-to-end InsightForge question-answering pipeline."""

    def __init__(
        self,
        documents: list[FactDocument],
        settings: Settings | None = None,
        memory: ConversationMemory | None = None,
        top_k: int = 6,
    ):
        self.settings = settings or get_settings()
        self.retriever = get_retriever(documents, self.settings)
        self.memory = memory or ConversationMemory()
        self.top_k = top_k

        self._llm = None
        if self.settings.llm_available:
            try:
                self._llm = get_chat_model(self.settings)
            except LLMNotAvailableError:
                self._llm = None

    @property
    def mode(self) -> str:
        return "llm" if self._llm is not None else "extractive"

    def _category_hint(self, question: str) -> str | None:
        q = question.lower()
        if any(w in q for w in ["trend", "month", "quarter", "growth", "period", "over time"]):
            return "trend"
        if any(w in q for w in ["product", "category", "sku"]):
            return "product"
        if any(w in q for w in ["region", "country", "market", "geograph"]):
            return "region"
        if any(w in q for w in ["segment", "customer", "demographic", "age"]):
            return "segment"
        if any(w in q for w in ["median", "average", "mean", "std", "deviation", "variance", "statistic"]):
            return "statistics"
        return None

    def ask(self, question: str) -> InsightForgeResponse:
        category = self._category_hint(question)
        facts = self.retriever.retrieve(question, top_k=self.top_k, category=category)
        if not facts and category:  # category filter too strict -> broaden
            facts = self.retriever.retrieve(question, top_k=self.top_k)

        fact_texts = [f.text for f in facts]
        memory_turns = self.memory.get_turns()

        if self._llm is not None:
            answer, draft = self._run_llm_chain(question, fact_texts, memory_turns)
        else:
            draft = _render_extractive_answer(question, facts)
            answer = draft

        self.memory.add_turn(question, answer)
        return InsightForgeResponse(
            question=question, answer=answer, draft=draft,
            retrieved_facts=facts, mode=self.mode,
        )

    def _run_llm_chain(
        self, question: str, fact_texts: list[str], memory_turns: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """Two-hop LCEL-style chain: analysis draft -> narrative polish."""
        analysis_prompt = build_analysis_prompt(question, fact_texts, memory_turns)
        draft = self._llm.invoke(analysis_prompt).content

        narrative_prompt = build_narrative_prompt(question, draft, memory_turns)
        final = self._llm.invoke(narrative_prompt).content
        return final, draft


def build_default_chain(documents: list[FactDocument]) -> InsightForgeRAGChain:
    return InsightForgeRAGChain(documents=documents)
