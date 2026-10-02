"""
Prompt engineering & prompt chaining for InsightForge (Part 1, Steps 3-4).

Two prompts are chained together (a "chain of prompts", per the brief):

  1. ANALYSIS_PROMPT — grounds the LLM strictly in the retrieved fact
     documents and produces a structured, cited draft answer.
  2. NARRATIVE_PROMPT — takes that structured draft plus the running
     conversation memory and turns it into a polished, executive-ready
     narrative response (coherent, contextually relevant, refers back to
     prior turns when useful).

Keeping these as plain string templates (rather than hard LangChain
`PromptTemplate` objects) means this module has no import-time dependency
on `langchain` — the RAG chain assembler (`src/rag/chain.py`) is the only
place that wraps them into LangChain-native objects, and only when
LangChain is actually installed/available.
"""
from __future__ import annotations

SYSTEM_PERSONA = (
    "You are InsightForge, an AI Business Intelligence Assistant for small "
    "and medium-sized enterprises. You analyze sales, product, regional, "
    "and customer data to deliver clear, decision-ready insights. "
    "You are precise with numbers, transparent about uncertainty, and you "
    "never invent statistics that are not present in the provided context."
)

# --- Prompt 1: Grounded analytical draft -----------------------------------
ANALYSIS_PROMPT_TEMPLATE = """{system_persona}

## Retrieved business facts (ground truth — use ONLY these numbers)
{context}

## Conversation memory (prior turns, for continuity)
{memory}

## User question
{question}

## Instructions
1. Answer using ONLY the numbers present in the "Retrieved business facts" section.
2. If the facts do not fully answer the question, explicitly say what is missing
   instead of guessing or inventing a number.
3. Structure your draft as:
   - Key finding (1-2 sentences)
   - Supporting evidence (bullet points with exact figures and their source fact)
   - Notable caveat or limitation (if any)
4. Be concise. Do not restate the entire context verbatim.

Draft analysis:
"""

# --- Prompt 2: Narrative polish / final response ----------------------------
NARRATIVE_PROMPT_TEMPLATE = """{system_persona}

You will now turn a structured analytical draft into a final answer for a
business stakeholder who wants a fast, confident read of the data.

## Draft analysis
{draft}

## Conversation memory (prior turns)
{memory}

## Original question
{question}

## Instructions
1. Write a natural, executive-ready answer (not bullet-only, blend prose and
   1-3 bullets max for supporting numbers).
2. Reference prior conversation turns only if they are directly relevant
   (e.g. "compared to what we discussed about Q3...").
3. End with one concise, actionable recommendation grounded in the data.
4. Keep the entire answer under ~180 words unless the question explicitly
   asks for a deep, exhaustive breakdown.
5. Never fabricate a statistic that was not in the draft analysis.

Final answer:
"""

EVAL_PROMPT_TEMPLATE = """You are grading whether an AI-generated business answer is factually
grounded in the provided source facts.

QUESTION: {question}
SOURCE FACTS: {context}
REFERENCE ANSWER: {reference}
AI ANSWER: {prediction}

Grade the AI ANSWER as CORRECT if it is factually consistent with the
SOURCE FACTS and captures the same key figures/insight as the REFERENCE
ANSWER (wording may differ). Grade it INCORRECT if it contradicts the
source facts, invents numbers, or misses the key insight.

Respond in exactly this format:
GRADE: <CORRECT|INCORRECT>
REASON: <one sentence>
"""


def format_context(facts: list[str]) -> str:
    if not facts:
        return "(no matching facts retrieved)"
    return "\n".join(f"- {f}" for f in facts)


def format_memory(turns: list[tuple[str, str]], max_turns: int = 4) -> str:
    """Render the last `max_turns` (question, answer) pairs as plain text."""
    if not turns:
        return "(no prior conversation)"
    recent = turns[-max_turns:]
    lines = []
    for q, a in recent:
        lines.append(f"User: {q}\nInsightForge: {a}")
    return "\n\n".join(lines)


def build_analysis_prompt(question: str, facts: list[str], memory_turns: list[tuple[str, str]]) -> str:
    return ANALYSIS_PROMPT_TEMPLATE.format(
        system_persona=SYSTEM_PERSONA,
        context=format_context(facts),
        memory=format_memory(memory_turns),
        question=question,
    )


def build_narrative_prompt(question: str, draft: str, memory_turns: list[tuple[str, str]]) -> str:
    return NARRATIVE_PROMPT_TEMPLATE.format(
        system_persona=SYSTEM_PERSONA,
        draft=draft,
        memory=format_memory(memory_turns),
        question=question,
    )


def build_eval_prompt(question: str, context: list[str], reference: str, prediction: str) -> str:
    return EVAL_PROMPT_TEMPLATE.format(
        question=question,
        context=format_context(context),
        reference=reference,
        prediction=prediction,
    )
