"""
CLI convenience wrapper to run the InsightForge evaluation suite outside
Streamlit, e.g. as a CI regression check.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --eval-file eval/eval_dataset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import load_eval_dataset, results_to_dicts, run_evaluation, summarize_results
from src.pipeline import build_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the InsightForge evaluation suite.")
    parser.add_argument(
        "--eval-file", type=Path, default=PROJECT_ROOT / "eval" / "eval_dataset.json",
        help="Path to a JSON file of {question, reference_answer} objects.",
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=80.0,
        help="Exit with a non-zero status if accuracy falls below this percentage.",
    )
    args = parser.parse_args()

    app = build_app()
    cases = load_eval_dataset(args.eval_file)
    results = run_evaluation(app.chain, cases)
    summary = summarize_results(results)

    print(f"Mode: {app.chain.mode}")
    print(f"Evaluator: {summary['evaluator']}")
    print(f"Accuracy: {summary['accuracy_pct']}% ({summary['correct']}/{summary['total_cases']})\n")
    for r in results:
        print(f"[{r.grade}] {r.question}\n    -> {r.reason}\n")

    if summary["accuracy_pct"] < args.min_accuracy:
        print(f"FAILED: accuracy {summary['accuracy_pct']}% is below threshold {args.min_accuracy}%")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
