"""Run the agent against the evaluation test cases.

Drives every case in eval/test_cases.json through the full agent pipeline
(Claude via the Anthropic API -> anomaly check -> classify -> retrieve ->
self-reflect), then measures and reports quality metrics.

Usage:
    .venv/bin/python eval/run_eval.py            # run all cases
    .venv/bin/python eval/run_eval.py --limit 3  # quick smoke run

Note: each case makes several real LLM calls (agent turns + reflection,
looping up to 3x), so a full 20-case run costs API tokens and a few minutes.
Requires a valid ANTHROPIC_API_KEY and a running, loaded Weaviate for the
retrieval-coverage metric to be meaningful.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Make the project root importable so `from app.agent import run_agent` works
# regardless of the directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import run_agent  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASES_PATH = os.path.join(EVAL_DIR, "test_cases.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "results.json")

# Flagging thresholds (per the brief).
LOW_CONFIDENCE = 5   # flag when confidence_score < 5
MIN_RETRIEVAL = 1    # flag when fewer than this many incidents were found


def _norm(category) -> str:
    """Normalise a category label for comparison (lowercase, trimmed)."""
    return str(category or "").strip().lower()


def run_case(case: dict) -> dict:
    """Run one test case through the agent and score it."""
    expected = _norm(case["expected_fault_category"])

    try:
        result = run_agent(
            equipment_id=case["equipment_id"],
            log_entry=case["log_entry"],
        )
        error = None
    except Exception as exc:  # noqa: BLE001 - record, don't abort the whole run
        result = {}
        error = f"{type(exc).__name__}: {exc}"

    predicted = _norm(result.get("fault_category"))
    confidence = result.get("confidence_score", 0) or 0
    iterations = result.get("iterations_taken", 0) or 0
    found = result.get("similar_incidents_found", 0) or 0

    # A prediction only counts as correct if it actually matches the expected
    # category (an "unknown"/empty prediction is never correct).
    correct = bool(predicted) and predicted == expected

    # Build the per-case flag list.
    flags = []
    if error:
        flags.append("pipeline_error")
    if confidence < LOW_CONFIDENCE:
        flags.append("low_confidence")
    if found < MIN_RETRIEVAL:
        flags.append("no_retrieval")

    return {
        "test_id": case["test_id"],
        "equipment_id": case["equipment_id"],
        "expected_category": case["expected_fault_category"],
        "predicted_category": result.get("fault_category"),
        "correct": correct,
        "confidence_score": confidence,
        "iterations_taken": iterations,
        "similar_incidents_found": found,
        "flags": flags,
        "flagged": bool(flags),
        "is_ambiguous": case["test_id"] in ("TC-19", "TC-20"),
        "error": error,
        # Keep a trimmed answer for eyeballing; full text would bloat the file.
        "final_answer": (result.get("final_answer") or "")[:600],
    }


def summarise(results: list) -> dict:
    """Aggregate per-case results into the headline metrics."""
    total = len(results)
    scored = [r for r in results if r["error"] is None]
    n_scored = len(scored) or 1  # avoid divide-by-zero

    correct = sum(1 for r in scored if r["correct"])
    with_retrieval = sum(1 for r in scored if r["similar_incidents_found"] >= MIN_RETRIEVAL)
    avg_conf = sum(r["confidence_score"] for r in scored) / n_scored
    avg_iters = sum(r["iterations_taken"] for r in scored) / n_scored

    # Per-category accuracy breakdown (helps spot which fault types are weak).
    per_category = {}
    for r in scored:
        cat = r["expected_category"]
        bucket = per_category.setdefault(cat, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(r["correct"])
    for cat, b in per_category.items():
        b["accuracy"] = round(b["correct"] / b["total"], 3) if b["total"] else 0.0

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_cases": total,
        "cases_scored": len(scored),
        "pipeline_errors": total - len(scored),
        "classifier_accuracy": round(correct / n_scored, 3),
        "retrieval_coverage": round(with_retrieval / n_scored, 3),
        "avg_confidence": round(avg_conf, 2),
        "avg_iterations": round(avg_iters, 2),
        "flagged_count": sum(1 for r in results if r["flagged"]),
        "per_category_accuracy": per_category,
    }


def print_summary(summary: dict, results: list):
    """Print a compact one-page report to the terminal."""
    line = "=" * 64
    print("\n" + line)
    print(" PREDICTIVE MAINTENANCE COPILOT — EVALUATION SUMMARY")
    print(line)
    print(f" Run at              : {summary['timestamp']}")
    print(f" Cases run           : {summary['total_cases']}"
          f"  (scored {summary['cases_scored']}, "
          f"errors {summary['pipeline_errors']})")
    print("-" * 64)
    print(f" Classifier accuracy : {summary['classifier_accuracy']:.1%}")
    print(f" Retrieval coverage  : {summary['retrieval_coverage']:.1%}  "
          f"(>= {MIN_RETRIEVAL} incident found)")
    print(f" Avg confidence      : {summary['avg_confidence']} / 10")
    print(f" Avg iterations      : {summary['avg_iterations']}")
    print(f" Flagged cases       : {summary['flagged_count']}")
    print("-" * 64)
    print(" Per-category accuracy:")
    for cat, b in sorted(summary["per_category_accuracy"].items()):
        print(f"   {cat:18s} {b['correct']}/{b['total']}  ({b['accuracy']:.0%})")

    flagged = [r for r in results if r["flagged"]]
    if flagged:
        print("-" * 64)
        print(" Flagged cases (confidence < 5, no retrieval, or error):")
        for r in flagged:
            amb = " [ambiguous]" if r["is_ambiguous"] else ""
            print(f"   {r['test_id']:6s} conf={r['confidence_score']} "
                  f"found={r['similar_incidents_found']} "
                  f"-> {', '.join(r['flags'])}{amb}")
    print(line)
    print(f" Detailed report written to: {RESULTS_PATH}")
    print(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the copilot evaluation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N test cases (for a quick smoke run).")
    args = parser.parse_args()

    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)
    if args.limit:
        cases = cases[:args.limit]

    print(f"Running {len(cases)} test case(s) through the agent pipeline...\n")

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['test_id']} "
              f"(equip {case['equipment_id']}, expect {case['expected_fault_category']})...",
              flush=True)
        r = run_case(case)
        status = "OK" if not r["flagged"] else f"FLAGGED: {','.join(r['flags'])}"
        print(f"         pred={r['predicted_category']} "
              f"conf={r['confidence_score']} iters={r['iterations_taken']} "
              f"found={r['similar_incidents_found']}  [{status}]")
        results.append(r)

    summary = summarise(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print_summary(summary, results)


if __name__ == "__main__":
    main()
