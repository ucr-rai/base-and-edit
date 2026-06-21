#!/usr/bin/env python3
"""R2: LeanScribe SFT fill-answer post-processing.

Provides preprocess() and postprocess() for the R2 stage of the hybrid pipeline.
R2 takes problems where R1 failed, runs LeanScribe SFT to get fill_answer code,
then applies it to each candidate and Lean-verifies.

Used as pre/postprocess hooks in gen_data.py.
"""

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset

from verify import verify_lean
from base_and_edit.run_eval_ours_R1 import (
    find_occurrences_outside_comments,
    filter_in_theorem_body,
    mark_occurrence,
)


# ---------------------------------------------------------------------------
# JSON parsing helpers (from run_eval_ours_hybrid.py)
# ---------------------------------------------------------------------------
def extract_b2_json(text: str) -> tuple[str, str, bool]:
    if not text:
        return "", "", False
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e > s:
            candidate = text[s:e + 1]
    if candidate is None:
        return "", "", False
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "", "", False
    olb = parsed.get("original_local_block", "") or ""
    fac = parsed.get("fill_answer_code", "") or ""
    if not isinstance(olb, str):
        olb = ""
    if not isinstance(fac, str):
        fac = ""
    return olb, fac, True


def exec_fill_answer(function_code: str):
    namespace: dict[str, Any] = {}
    try:
        exec(function_code, namespace)
    except Exception as exc:
        return None, f"exec error: {type(exc).__name__}: {exc}"
    fn = namespace.get("fill_answer")
    if not callable(fn):
        return None, "fill_answer not defined"
    return fn, ""


def safe_call(fn, arg: str) -> tuple[str, str]:
    try:
        out = fn(arg)
        return (str(out) if out is not None else ""), ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Preprocess: filter R1 output to rows that need R2
# ---------------------------------------------------------------------------
def preprocess(ds: Dataset) -> Dataset:
    """Filter to rows where R1 failed (needs_r2=True)."""
    return ds.filter(lambda x: bool(x.get("needs_r2", False)))


# ---------------------------------------------------------------------------
# Postprocess: apply fill_answer to candidates, Lean-verify, merge with R1
# ---------------------------------------------------------------------------
def postprocess(ds_r2: Dataset, ds_r1: Dataset = None) -> list[dict[str, Any]]:
    """Apply R2 fill_answer results to candidates and Lean-verify.

    Args:
        ds_r2: Dataset with r2_fill_answer output (from gen_data.py).
        ds_r1: Original R1 output dataset (all rows). If provided, R2 results
               are merged back into the full result set.

    Returns:
        List of per-problem result dicts with candidate_results[].
    """
    MARKER = "__DISAMBIG_TARGET__"
    all_lean_codes = []
    all_lean_meta = []
    r2_results = []

    for row_idx, row in enumerate(ds_r2):
        pi = row.get("problem_index", row_idx)
        formal = row.get("formal_problem", "") or ""
        original_answer = str(row.get("original_answer", ""))
        gt = str(row.get("gt_answer", row.get("answer", "")))
        candidates = [str(c) for c in (row.get("candidate_answers", row.get("candidates", [])) or [])]
        raw_output = row.get("r2_fill_answer", row.get("_output_raw", ""))

        olb, fac, json_ok = extract_b2_json(raw_output)
        code_occs = find_occurrences_outside_comments(formal, olb) if (json_ok and olb) else []
        code_occs = filter_in_theorem_body(formal, code_occs)

        fill_fn = None
        marked_formal = None
        route = "sft::b2"

        if json_ok and olb and len(code_occs) == 1 and fac:
            fn, err = exec_fill_answer(fac)
            if fn is not None:
                fill_fn = fn
                marked_formal = mark_occurrence(formal, olb, code_occs[0], MARKER)
            else:
                route = "sft::b2_exec_fail"
        else:
            route = "sft::b2_parse_fail"

        cand_results = []
        for cand_i, cand in enumerate(candidates):
            cr: dict[str, Any] = {
                "candidate": cand,
                "is_gt": cand == gt,
                "lean_passed": False,
                "route": route,
            }

            # Matches run_eval_ours_hybrid.py's base_shortcircuit: the candidate
            # equal to original_answer is exactly the pre-verified base
            # statement, so skip the fill_answer round-trip for it. Without
            # this, an imperfect fill_answer roundtrip can spuriously fail the
            # base candidate slot, depressing Pass@K below the base coverage rate.
            if cand == original_answer:
                cr["lean_passed"] = True
                cr["route"] = route + "+base_cached"
                cr["new_formal"] = formal
                cand_results.append(cr)
                continue

            if fill_fn is not None and marked_formal is not None:
                out, _ = safe_call(fill_fn, cand)
                if out:
                    new_formal = marked_formal.replace(MARKER, out, 1)
                    cr["new_formal"] = new_formal
                    all_lean_codes.append(new_formal)
                    all_lean_meta.append((row_idx, cand_i))

            cand_results.append(cr)

        r2_results.append({
            "problem_index": pi,
            "problem": row.get("problem", row.get("informal_problem", "")),
            "method_name": row.get("method_name", "hybrid"),
            "route": route,
            "needs_r2": False,
            "candidates": candidates,
            "candidate_answers": candidates,
            "gt_answer": gt,
            "original_answer": original_answer,
            "formal_problem": formal,
            "candidate_results": cand_results,
        })

    if all_lean_codes:
        print(f"Lean-verifying {len(all_lean_codes)} R2 reconstructions...")
        lean_results = verify_lean(all_lean_codes)
        for (row_i, cand_i), lr in zip(all_lean_meta, lean_results):
            r2_results[row_i]["candidate_results"][cand_i]["lean_passed"] = bool(lr.get("passed"))

    if ds_r1 is None:
        return r2_results

    # Merge R2 results back into full R1 output
    r2_by_index = {}
    for r in r2_results:
        r2_by_index[r["problem_index"]] = r

    merged = []
    for i, row in enumerate(ds_r1):
        pi = row.get("problem_index", i)
        if row.get("needs_r2") and pi in r2_by_index:
            r2_row = r2_by_index[pi]
            r1_cands = row.get("candidate_results", [])
            r2_cands = r2_row.get("candidate_results", [])
            # Merge per-candidate: keep R1 result unless the candidate needed R2
            merged_cands = []
            for ci in range(max(len(r1_cands), len(r2_cands))):
                r1_cr = r1_cands[ci] if ci < len(r1_cands) else {}
                r2_cr = r2_cands[ci] if ci < len(r2_cands) else {}
                if r1_cr.get("needs_r2") or r1_cr.get("route") == "pending_r2":
                    merged_cands.append(r2_cr)
                else:
                    merged_cands.append(r1_cr)
            out_row = dict(row)
            out_row["candidate_results"] = merged_cands
            out_row["needs_r2"] = False
            out_row["route"] = r2_row["route"]
            merged.append(out_row)
        else:
            merged.append(dict(row))
    return merged


def compute_summary(results: list[dict[str, Any]]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    n_any_passed = sum(
        1 for r in results
        if any(c.get("lean_passed") for c in r["candidate_results"])
    )
    n_gt_passed = sum(
        1 for r in results
        if any(c.get("lean_passed") and c.get("is_gt") for c in r["candidate_results"])
    )
    route_counts: dict[str, int] = {}
    for r in results:
        rt = r.get("route", "unknown")
        route_counts[rt] = route_counts.get(rt, 0) + 1

    return {
        "num_problems": n,
        "any_candidate_passed": n_any_passed,
        "any_candidate_passed_rate": n_any_passed / n,
        "gt_candidate_passed": n_gt_passed,
        "gt_candidate_passed_rate": n_gt_passed / n,
        "route_distribution": route_counts,
    }
