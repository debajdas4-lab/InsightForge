"""
Model evaluation (Part 2, Step 7): applies a QAEvalChain-style LLM-graded
evaluation to assess InsightForge's answer quality and accuracy.

Two evaluation modes, mirroring the RAG chain itself:

* **LLM-graded** (`LangChainQAEvaluator`): uses LangChain's
  `langchain.evaluation.qa.QAEvalChain` when an LLM is configured — this is
  the direct implementation of the brief's "Apply QAEvalChain to assess the
  model's performance and accuracy".

* **Heuristic fallback** (`HeuristicEvaluator`): a fully offline, numeric
  grader that checks whether every number mentioned in the reference answer
  also appears in the prediction (a strong proxy for factual grounding in a
  BI assistant, where correctness is almost always "did you cite the right
  figures"). This keeps the evaluation harness runnable with zero API cost,
  which is what this project uses for its bundled regression suite.

Both share one `EvalResult` schema and one `run_evaluation()` entry point so
`app/streamlit_app.py` and `tests/` don't need to know which backend ran.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from src.config import Settings, get_settings
from src.rag.chain import InsightForgeRAGChain
from src.rag.llm_factory import LLMNotAvailableError, get_chat_model

_NUMBER_RE = re.compile(r"\$?-?\d[\d,]*\.?\d*%?")


@dataclass
class EvalCase:
    question: str
    reference_answer: str


@dataclass
class EvalResult:
    question: str
    reference_answer: str
    prediction: str
    grade: str  # "CORRECT" | "INCORRECT"
    reason: str
    evaluator: str  # "qa_eval_chain" | "heuristic_numeric_overlap"


def load_eval_dataset(path: Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data]


def _extract_numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUMBER_RE.findall(text)}


class HeuristicEvaluator:
    """Offline numeric-grounding grader (no LLM, no API cost).

    A prediction is graded CORRECT if it contains at least `min_overlap`
    fraction of the numeric tokens present in the reference answer. This is
    a deliberately conservative, reproducible proxy metric well-suited to a
    BI assistant, where "right numbers cited" is the dominant correctness
    signal.
    """

    name = "heuristic_numeric_overlap"

    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def grade(self, question: str, reference: str, prediction: str, context: list[str]) -> EvalResult:
        ref_numbers = _extract_numbers(reference)
        pred_numbers = _extract_numbers(prediction)
        if not ref_numbers:
            grade = "CORRECT" if prediction.strip() else "INCORRECT"
            reason = "Reference contains no numeric claims to verify; checked for a non-empty answer."
        else:
            overlap = ref_numbers & pred_numbers
            frac = len(overlap) / len(ref_numbers)
            grade = "CORRECT" if frac >= self.min_overlap else "INCORRECT"
            reason = (
                f"{len(overlap)}/{len(ref_numbers)} reference figures "
                f"({', '.join(sorted(ref_numbers)) or 'none'}) found in the prediction."
            )
        return EvalResult(
            question=question, reference_answer=reference, prediction=prediction,
            grade=grade, reason=reason, evaluator=self.name,
        )


class LangChainQAEvaluator:
    """Wraps `langchain.evaluation.qa.QAEvalChain` for LLM-graded evaluation.

    Instantiated lazily; only usable when LangChain + a configured LLM
    provider are available. See `get_evaluator()` for the fallback logic.
    """

    name = "qa_eval_chain"

    def __init__(self, settings: Settings):
        from langchain.evaluation.qa import QAEvalChain  # noqa: WPS433

        llm = get_chat_model(settings, temperature=0.0)
        self._chain = QAEvalChain.from_llm(llm)

    def grade(self, question: str, reference: str, prediction: str, context: list[str]) -> EvalResult:
        graded = self._chain.evaluate(
            examples=[{"query": question, "answer": reference}],
            predictions=[{"result": prediction}],
        )[0]
        raw_text = graded.get("results") or graded.get("text", "")
        grade = "CORRECT" if "correct" in raw_text.lower() and "incorrect" not in raw_text.lower() else "INCORRECT"
        return EvalResult(
            question=question, reference_answer=reference, prediction=prediction,
            grade=grade, reason=raw_text.strip(), evaluator=self.name,
        )


def get_evaluator(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.llm_available:
        try:
            return LangChainQAEvaluator(settings)
        except (ImportError, LLMNotAvailableError):
            pass
    return HeuristicEvaluator()


def run_evaluation(
    chain: InsightForgeRAGChain,
    eval_cases: list[EvalCase],
    evaluator: Any | None = None,
) -> list[EvalResult]:
    """Run every eval case through the RAG chain and grade the outputs."""
    evaluator = evaluator or get_evaluator(chain.settings)
    results: list[EvalResult] = []
    for case in eval_cases:
        response = chain.ask(case.question)
        context = [f.text for f in response.retrieved_facts]
        result = evaluator.grade(case.question, case.reference_answer, response.answer, context)
        results.append(result)
    return results


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r.grade == "CORRECT")
    return {
        "total_cases": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy_pct": round(correct / total * 100, 2) if total else 0.0,
        "evaluator": results[0].evaluator if results else "n/a",
    }


def results_to_dicts(results: list[EvalResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
