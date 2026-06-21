#!/usr/bin/env python3
"""Pre/post-processing for the brute-force (independent) formalization baseline.

Preprocess: expand each problem × candidate into a separate row suitable for
  gen_data.py with --input_key problem --output_key formal_problem.

Postprocess: aggregate per-candidate formalization results back into per-problem
  rows with candidate_results[].

Can also be run standalone for quick tests.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset


def preprocess(ds: Dataset) -> Dataset:
    """Expand each problem's K candidates into separate rows for formalization.

    Each output row has: problem, answer (= the candidate), plus tracking fields
    (problem_index, candidate_index, gt_answer, candidate_answers, original_answer).
    """
    rows = []
    for i, example in enumerate(ds):
        problem = example.get("problem", "") or example.get("informal_problem", "")
        candidates = [str(c) for c in example.get("candidate_answers", example.get("ranked_answers", []))]
        gt = str(example.get("gt_answer", example.get("answer", "")))
        original_answer = str(example.get("original_answer", ""))

        for cand_i, cand in enumerate(candidates):
            rows.append({
                "problem": problem,
                "answer": cand,
                "problem_index": i,
                "candidate_index": cand_i,
                "candidate": cand,
                "gt_answer": gt,
                "original_answer": original_answer,
            })

    return Dataset.from_list(rows)


def postprocess(ds: Dataset, method_name: str = "brute_force") -> list[dict[str, Any]]:
    """Aggregate per-candidate results back into per-problem rows.

    Returns a list of dicts, one per problem, with candidate_results[].
    Also picks the first Lean-passing candidate as the base statement
    (sets formal_problem and original_answer on the result row).
    """
    by_problem: dict[int, list] = {}
    for example in ds:
        pi = int(example["problem_index"])
        if pi not in by_problem:
            by_problem[pi] = []
        by_problem[pi].append(example)

    results = []
    for pi in sorted(by_problem):
        cand_rows = sorted(by_problem[pi], key=lambda x: int(x["candidate_index"]))
        candidates = [r["candidate"] for r in cand_rows]
        gt = cand_rows[0]["gt_answer"]
        problem = cand_rows[0]["problem"]

        candidate_results = []
        best_base_rank = None
        best_base_formal = None
        best_base_answer = None
        for rank, r in enumerate(cand_rows):
            passed = bool(r.get("passed", False))
            candidate_results.append({
                "candidate": r["candidate"],
                "is_gt": str(r["candidate"]) == str(gt),
                "formalize_ok": bool(r.get("formal_problem")),
                "lean_passed": passed,
            })
            if passed and best_base_rank is None:
                best_base_rank = rank
                best_base_formal = r.get("formal_problem", "")
                best_base_answer = r["candidate"]

        row = {
            "problem_index": pi,
            "problem": problem,
            "method_name": method_name,
            "candidates": candidates,
            "candidate_answers": candidates,
            "gt_answer": gt,
            "candidate_results": candidate_results,
        }

        if best_base_rank is not None:
            row["formal_problem"] = best_base_formal
            row["original_answer"] = best_base_answer
            row["best_base"] = {
                "base_answer": best_base_answer,
                "base_rank_0based": best_base_rank,
                "base_is_gt": candidate_results[best_base_rank]["is_gt"],
                "formalize_calls_until_base": best_base_rank + 1,
            }

        results.append(row)

    return results


def compute_summary(results: list[dict[str, Any]]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    n_any_passed = sum(
        1 for r in results
        if any(c["lean_passed"] for c in r["candidate_results"])
    )
    n_gt_passed = sum(
        1 for r in results
        if any(c["lean_passed"] and c["is_gt"] for c in r["candidate_results"])
    )
    total_candidates = sum(len(r["candidate_results"]) for r in results)
    total_formalized = sum(
        sum(1 for c in r["candidate_results"] if c["formalize_ok"])
        for r in results
    )
    total_lean_passed = sum(
        sum(1 for c in r["candidate_results"] if c["lean_passed"])
        for r in results
    )

    return {
        "num_problems": n,
        "total_candidates": total_candidates,
        "total_formalized": total_formalized,
        "total_lean_passed": total_lean_passed,
        "any_candidate_passed": n_any_passed,
        "any_candidate_passed_rate": n_any_passed / n,
        "gt_candidate_passed": n_gt_passed,
        "gt_candidate_passed_rate": n_gt_passed / n,
    }
