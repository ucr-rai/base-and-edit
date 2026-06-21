#!/usr/bin/env python3
"""Aggregate DSP-v2 prove results from HF datasets into paper-table metrics.

Pulls 4 sharded prove output repos per method, computes:
  - Per-cand proof pass:  rows where samples[0].passed = True
  - Pass@K:               problems with >=1 passing candidate
  - GT-pass:              problems where GT candidate's proof passes
  - Acc|GT@K:             first-pass selected candidate is GT (and Lean-passed)
  - Acc-overall:          first-pass selected candidate is GT (overall n)

Output: a markdown table comparing the methods in your earlier format:
  Method | Prover | Problems | Candidates | GT@K | Pass@K | GT-pass | Per-cand pass | Acc|GT@K
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "prefix", nargs="?", default=None,
        help="File prefix (e.g. outputs/math500_k8_gemini). "
             "Derives method-names, prove-repo-prefixes, and output paths automatically.",
    )
    p.add_argument(
        "--method-name",
        action="append",
        default=None,
        help="Method tag, e.g. 'hybrid_v7_8b_best_base' or 'brute_force'. "
             "Provide multiple times alongside matching --prove-repo-prefix entries.",
    )
    p.add_argument(
        "--prove-repo-prefix",
        action="append",
        default=None,
        help="HF repo prefix (without -shardN suffix). One per method.",
    )
    p.add_argument(
        "--shard-total", type=int, default=4,
    )
    p.add_argument(
        "--prover-name", default="DeepSeek-Prover-V2-7B-CoT",
    )
    p.add_argument(
        "--output-md", default=None,
    )
    p.add_argument(
        "--output-summary", default=None,
    )
    p.add_argument(
        "--output-jsonl-prefix", default=None,
        help="Per-method aggregated jsonl: <prefix>_<method>.jsonl",
    )
    args = p.parse_args()
    if args.prefix and not args.method_name:
        stem = Path(args.prefix).name
        args.method_name = [f"brute_force_{stem}", f"hybrid_{stem}"]
        args.prove_repo_prefix = [f"{stem}_brute_dspv2_proof", f"{stem}_hybrid_dspv2_proof"]
    if not args.method_name or not args.prove_repo_prefix:
        p.error("Provide either a prefix or --method-name/--prove-repo-prefix.")
    if args.output_md is None:
        args.output_md = f"{args.prefix}_prover_table.md" if args.prefix else "outputs/prove_results_table.md"
    if args.output_summary is None:
        args.output_summary = f"{args.prefix}_prover_summary.json" if args.prefix else "outputs/prove_results_summary.json"
    if args.output_jsonl_prefix is None:
        args.output_jsonl_prefix = f"{args.prefix}_prove" if args.prefix else "outputs/prove_results"
    return args


def load_prove_dataset(repo_prefix: str, shard_total: int) -> list[dict]:
    """Load all shards of prove output from HF, return list of rows."""
    from datasets import load_dataset

    rows: list[dict] = []
    for sid in range(shard_total):
        repo = f"{repo_prefix}-shard{sid}"
        split = f"shard{sid}of{shard_total}"
        try:
            ds = load_dataset(repo, split=split)
        except Exception as e:
            print(f"  ⚠️  Failed loading {repo}[{split}]: {type(e).__name__}: {e}")
            continue
        for r in ds:
            rows.append(dict(r))
        print(f"  loaded {repo}[{split}]: {len(ds)} rows")
    return rows


def candidate_passed(row: dict) -> bool:
    """Returns True if this candidate's proof Lean-passed.

    The prove pipeline stores `passed` at the top level (bool from Lean
    kernel verification of the final proof). `samples` is just a list of
    raw model proof attempts (strings).
    """
    return bool(row.get("passed"))


def compute_metrics(rows: list[dict], method_name: str) -> dict:
    """Compute paper-table metrics from candidate-level rows."""
    by_problem: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        idx = int(r.get("index", -1))
        by_problem[idx].append(r)

    n_problems = len(by_problem)
    n_candidates = len(rows)

    n_per_cand_pass = sum(1 for r in rows if candidate_passed(r))

    n_pass_at_k = 0       # problems with >=1 candidate passed
    n_gt_pass = 0         # problems where GT candidate's proof passed
    n_gt_in_k = 0         # problems where GT in candidate set
    n_correct = 0         # selected (first-pass) is GT and passed
    n_correct_acc_gtk_only = 0  # restricted to GT@K problems

    for idx, cands in by_problem.items():
        # Order by candidate_index for stable selection
        cands_sorted = sorted(cands, key=lambda x: x.get("candidate_index", 0))
        any_pass = any(candidate_passed(c) for c in cands_sorted)
        gt_pass = any(c.get("is_gt") and candidate_passed(c) for c in cands_sorted)
        gt_in = any(c.get("is_gt") for c in cands_sorted)
        first_pass = next((c for c in cands_sorted if candidate_passed(c)), None)
        selected_is_gt = bool(first_pass and first_pass.get("is_gt"))

        if any_pass:
            n_pass_at_k += 1
        if gt_pass:
            n_gt_pass += 1
        if gt_in:
            n_gt_in_k += 1
        if selected_is_gt and gt_pass:
            n_correct += 1
            if gt_in:
                n_correct_acc_gtk_only += 1

    return {
        "method": method_name,
        "n_problems": n_problems,
        "n_candidates": n_candidates,
        "gt_in_k": n_gt_in_k,
        "pass_at_k": n_pass_at_k,
        "gt_pass": n_gt_pass,
        "per_cand_pass": n_per_cand_pass,
        "acc_overall_correct": n_correct,
        "acc_gt_at_k": (n_correct_acc_gtk_only / n_gt_in_k) if n_gt_in_k else 0.0,
        "acc_overall": n_correct / n_problems if n_problems else 0.0,
        "per_cand_pass_rate": n_per_cand_pass / n_candidates if n_candidates else 0.0,
        "pass_at_k_rate": n_pass_at_k / n_problems if n_problems else 0.0,
        "gt_pass_rate_over_gt_in_k": (n_gt_pass / n_gt_in_k) if n_gt_in_k else 0.0,
        "gt_pass_rate_over_n": n_gt_pass / n_problems if n_problems else 0.0,
    }


def render_md(metrics_list: list[dict], prover_name: str) -> str:
    headers = [
        "Method", "Prover", "Problems", "Candidates", "GT@K", "Pass@K",
        "GT-pass", "Per-cand pass", "Acc|GT@K", "Acc-overall",
    ]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for m in metrics_list:
        out.append("| " + " | ".join([
            m["method"],
            prover_name,
            str(m["n_problems"]),
            str(m["n_candidates"]),
            f"{m['gt_in_k']}/{m['n_problems']}",
            f"{m['pass_at_k']}/{m['n_problems']} ({m['pass_at_k_rate']:.1%})",
            f"{m['gt_pass']}/{m['gt_in_k']} ({m['gt_pass_rate_over_gt_in_k']:.1%})",
            f"{m['per_cand_pass']}/{m['n_candidates']} ({m['per_cand_pass_rate']:.1%})",
            f"{m['acc_overall_correct']}/{m['gt_in_k']} ({m['acc_gt_at_k']:.1%})",
            f"{m['acc_overall_correct']}/{m['n_problems']} ({m['acc_overall']:.1%})",
        ]) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    if len(args.method_name) != len(args.prove_repo_prefix):
        raise SystemExit("--method-name and --prove-repo-prefix must be paired one-to-one")

    metrics_list: list[dict] = []
    out_jsonl_prefix = Path(args.output_jsonl_prefix)
    out_jsonl_prefix.parent.mkdir(parents=True, exist_ok=True)

    for method, prefix in zip(args.method_name, args.prove_repo_prefix):
        print(f"\n=== Loading {method} from {prefix} ===")
        rows = load_prove_dataset(prefix, args.shard_total)
        if not rows:
            print(f"  No rows loaded for {method}, skipping.")
            continue

        # Save aggregated rows for inspection
        per_method_jsonl = Path(f"{out_jsonl_prefix}_{method}.jsonl")
        with per_method_jsonl.open("w") as f:
            for r in rows:
                # Drop large embeddings if any
                f.write(json.dumps({
                    "index": r.get("index"),
                    "candidate_index": r.get("candidate_index"),
                    "candidate": r.get("candidate"),
                    "is_gt": r.get("is_gt"),
                    "gt_answer": r.get("gt_answer"),
                    "method_name": r.get("method_name"),
                    "route": r.get("route"),
                    "passed": bool(r.get("passed")),
                    "complete": bool(r.get("complete")),
                    "n_samples": len(r.get("samples") or []),
                }, ensure_ascii=False) + "\n")
        print(f"  wrote {per_method_jsonl}")

        m = compute_metrics(rows, method)
        metrics_list.append(m)
        print(json.dumps(m, ensure_ascii=False, indent=2))

    md = render_md(metrics_list, args.prover_name)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(md)
    Path(args.output_summary).write_text(json.dumps(metrics_list, ensure_ascii=False, indent=2))
    print()
    print("=== Final Table ===")
    print(md)
    print()
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_summary}")


if __name__ == "__main__":
    main()
