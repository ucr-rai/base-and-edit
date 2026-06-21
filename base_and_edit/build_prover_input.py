#!/usr/bin/env python3
"""Build candidate-level prover input from hybrid or brute-force results.

Flattens per-problem method results into one row per candidate with the
Lean statement under `new_formal`.  Output can be fed to:

    python base_and_edit/run_deepseek_prover_jsonl.py --input-jsonl <output> ...

Usage (hybrid, default):
    python base_and_edit/build_prover_input.py \
        --eval-jsonl amc83_k8_gemini_hybrid \
        --source-eval-jsonl amc83_k8_gemini_brute_force \
        --output-jsonl outputs/amc83_k8_gemini_hybrid_prover_input.jsonl

Usage (brute-force):
    python base_and_edit/build_prover_input.py \
        --brute-force \
        --eval-jsonl amc83_k8_gemini_brute_force \
        --brute-formalize-jsonl amc83_k8_gemini_brute_force \
        --output-jsonl outputs/amc83_k8_gemini_brute_prover_input.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_brute(args: argparse.Namespace) -> None:
    eval_rows = load_jsonl(args.eval_jsonl)
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    eval_by_idx = {int(r["index"]): r for r in eval_rows}

    brute_rows = load_jsonl(args.eval_jsonl)
    allowed = set(eval_by_idx)
    brute_rows = [r for r in brute_rows if int(r["index"]) in allowed]

    formal_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for r in load_jsonl(args.brute_formalize_jsonl):
        key = (int(r["index"]), int(r["candidate_index"]))
        formal_by_pair[key] = r

    n_emit = 0
    n_missing = 0
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in brute_rows:
            idx = int(row["index"])
            eval_row = eval_by_idx[idx]
            original_answer = str(eval_row.get("original_answer", ""))
            original_formal = eval_row.get("formal_problem", "") or ""
            informal_problem = eval_row.get("informal_problem", "") or ""
            for cand_i, cr in enumerate(row.get("candidate_results") or []):
                if args.only_statement_passed and not cr.get("lean_passed"):
                    continue
                cand = str(cr.get("candidate", ""))
                if cand == original_answer and original_formal:
                    formal = original_formal
                    formalize_time = 0.0
                else:
                    fr = formal_by_pair.get((idx, cand_i), {})
                    formal = fr.get("formal_problem", "") or ""
                    formalize_time = float(fr.get("formalize_time_s") or 0.0)
                if not formal:
                    n_missing += 1
                    continue
                f.write(json.dumps({
                    "index": idx,
                    "method_name": row.get("method_name", "brute_force"),
                    "route": "brute_force",
                    "candidate_index": cand_i,
                    "candidate": cand,
                    "is_gt": bool(cr.get("is_gt")),
                    "gt_answer": row.get("gt_answer"),
                    "statement_lean_passed": bool(cr.get("lean_passed")),
                    "formalize_time_s": formalize_time,
                    "informal_problem": informal_problem,
                    "new_formal": formal,
                }, ensure_ascii=False) + "\n")
                n_emit += 1

    print(json.dumps({
        "mode": "brute_force",
        "eval_jsonl": args.eval_jsonl,
        "brute_formalize_jsonl": args.brute_formalize_jsonl,
        "output_jsonl": args.output_jsonl,
        "n_eval_rows": len(eval_rows),
        "n_brute_rows": len(brute_rows),
        "n_candidate_rows": n_emit,
        "n_missing_formal": n_missing,
        "only_statement_passed": args.only_statement_passed,
    }, ensure_ascii=False, indent=2))


def build_hybrid(args: argparse.Namespace) -> None:
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    informal_by_index: dict[Any, str] = {}
    if args.source_eval_jsonl:
        with open(args.source_eval_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                informal = str(row.get("informal_problem") or row.get("problem") or "")
                informal_by_index[row.get("index")] = informal

    n_rows = 0
    n_cands = 0
    n_missing = 0
    n_with_informal = 0
    with open(args.eval_jsonl, encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            n_rows += 1
            informal_problem = str(
                row.get("informal_problem")
                or row.get("problem")
                or informal_by_index.get(row.get("index"))
                or ""
            )
            for cand_i, cr in enumerate(row.get("candidate_results") or []):
                if args.top_k is not None and cand_i >= args.top_k:
                    break
                if args.only_statement_passed and not cr.get("lean_passed"):
                    continue
                new_formal = cr.get("new_formal") or ""
                if not new_formal:
                    n_missing += 1
                    continue
                fout.write(json.dumps({
                    "index": row.get("index"),
                    "method_name": row.get("method_name"),
                    "route": cr.get("route") or row.get("route"),
                    "candidate_index": cand_i,
                    "candidate": cr.get("candidate"),
                    "is_gt": bool(cr.get("is_gt")),
                    "gt_answer": row.get("gt_answer"),
                    "statement_lean_passed": bool(cr.get("lean_passed")),
                    "informal_problem": informal_problem,
                    "new_formal": new_formal,
                }, ensure_ascii=False) + "\n")
                n_cands += 1
                if informal_problem:
                    n_with_informal += 1

    print(json.dumps({
        "mode": "hybrid",
        "eval_jsonl": args.eval_jsonl,
        "source_eval_jsonl": args.source_eval_jsonl,
        "output_jsonl": args.output_jsonl,
        "n_problem_rows": n_rows,
        "n_candidate_rows": n_cands,
        "n_candidate_rows_with_informal_problem": n_with_informal,
        "n_missing_new_formal": n_missing,
        "only_statement_passed": args.only_statement_passed,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", nargs="?", default=None,
                        help="File prefix (e.g. outputs/math500_k8_gemini). "
                             "Derives --eval-jsonl, --source-eval-jsonl, --output-jsonl automatically.")
    parser.add_argument("--eval-jsonl", default=None,
                        help="Main input JSONL (hybrid results, or brute-force results).")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--no-only-statement-passed", action="store_true",
                        help="Include candidates that did not pass Lean (default: only passed).")
    parser.add_argument("--brute-force", action="store_true",
                        help="Use brute-force mode (requires --brute-formalize-jsonl).")
    parser.add_argument("--brute-formalize-jsonl", default=None,
                        help="Brute-force formalization JSONL (brute-force mode only).")
    parser.add_argument("--source-eval-jsonl", default=None,
                        help="Optional source eval JSONL with informal_problem keyed by index (hybrid mode only).")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Keep only the first K candidates per problem (hybrid mode only).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N eval rows (brute-force mode only).")

    args = parser.parse_args()
    if args.prefix and not args.eval_jsonl:
        if args.brute_force:
            args.eval_jsonl = f"{args.prefix}_brute_force.jsonl"
            if not args.brute_formalize_jsonl:
                args.brute_formalize_jsonl = f"{args.prefix}_brute_force.jsonl"
            if not args.output_jsonl:
                args.output_jsonl = f"{args.prefix}_brute_prover_input.jsonl"
        else:
            args.eval_jsonl = f"{args.prefix}_hybrid.jsonl"
            if not args.source_eval_jsonl:
                args.source_eval_jsonl = f"{args.prefix}_brute_force.jsonl"
            if not args.output_jsonl:
                args.output_jsonl = f"{args.prefix}_hybrid_prover_input.jsonl"
    if not args.eval_jsonl:
        parser.error("Provide either a prefix or --eval-jsonl.")
    if not args.output_jsonl:
        parser.error("Provide either a prefix or --output-jsonl.")
    args.only_statement_passed = not args.no_only_statement_passed
    if args.brute_force:
        if not args.brute_formalize_jsonl:
            parser.error("--brute-force requires --brute-formalize-jsonl")
        build_brute(args)
    else:
        build_hybrid(args)


if __name__ == "__main__":
    main()
