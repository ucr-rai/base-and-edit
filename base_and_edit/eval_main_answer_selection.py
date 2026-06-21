#!/usr/bin/env python3
"""End-to-end answer selection evaluation for the paper main table.

Reads the eval set + per-method `method_results` JSONL files (each method's
per-problem per-candidate Lean results + timings) and produces a comparison
table with the headline metrics.

Method results JSONL schema (one row per problem):
    {
      "problem_index": <int>,
      "method_name": "<str>",
      "candidate_results": [
        {
          "candidate": "<str>",
          "is_gt": <bool>,
          "formalize_ok": <bool>,
          "lean_passed": <bool>,
          "formalize_time_s": <float>,
          "replace_time_s": <float>,
          "lean_time_s": <float>
        }, ...
      ],
      "total_time_s": <float>
    }

Selection strategies (we report both):
  first_pass   first candidate (in candidate_answers order) whose lean_passed=True
  unique_pass  selection succeeds only if exactly one candidate Lean-passes
               (more conservative; treats ambiguous Lean-passes as failure)
"""

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi


def resolve_method_files(prefix: str) -> list[str]:
    """Given a prefix like 'outputs/math500_k8_gemini', return brute_force and hybrid JSONL paths.

    Paths may be local files or HF dataset names (load_jsonl handles both).
    """
    return [f"{prefix}_brute_force.jsonl", f"{prefix}_hybrid.jsonl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="?", default=None,
                        help="File prefix (e.g. outputs/math500_k8_gemini). "
                             "Expands to all matching *_brute_force.jsonl / *_hybrid.jsonl files.")
    parser.add_argument("--eval-set", default=None,
                        help="Eval-set JSONL. Defaults to the first method-results file.")
    parser.add_argument("--method-results", nargs="+", default=None,
                        help="One or more JSONL files of per-method results.")
    parser.add_argument("--output-summary", default=None,
                        help="Output summary JSON. Defaults to <prefix>_main_summary.json.")
    parser.add_argument("--gpu-dollar-per-hour", type=float, default=1.5,
                        help="Cost model for GPU compute. Used only for estimated cost tables.")
    parser.add_argument("--api-dollar-per-call", type=float, default=0.05,
                        help="Cost model for answer-generation/API calls. Used only for estimated cost tables.")
    args = parser.parse_args()
    if args.prefix and not args.method_results:
        args.method_results = resolve_method_files(args.prefix)
    if not args.method_results:
        parser.error("Provide either a prefix or --method-results.")
    if args.eval_set is None:
        args.eval_set = args.method_results[0]
    if args.output_summary is None:
        if args.prefix:
            args.output_summary = f"{args.prefix}_main_summary.json"
        else:
            stem = Path(args.method_results[0]).stem
            args.output_summary = str(Path(args.method_results[0]).parent / f"{stem}_main_summary.json")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        repo_id = path.stem if "/" in path.stem else f"{HfApi().whoami()['name']}/{path.stem}"
        ds_dict = load_dataset(repo_id)
        split = "test" if "test" in ds_dict else next(iter(ds_dict))
        return [dict(row) for row in ds_dict[split]]
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_first_pass(cand_results: list[dict]) -> dict | None:
    for c in cand_results:
        if c.get("lean_passed"):
            return c
    return None


def select_unique_pass(cand_results: list[dict]) -> dict | None:
    passing = [c for c in cand_results if c.get("lean_passed")]
    return passing[0] if len(passing) == 1 else None


def compute_method_metrics(eval_rows: list[dict], method_rows: list[dict]) -> dict:
    eval_by_idx = {int(r["problem_index"]): r for r in eval_rows}
    method_by_idx = {int(r["problem_index"]): r for r in method_rows}

    # Only brute-force per-candidate results carry "formalize_ok" (each candidate
    # is formalized independently, so formalization can fail per-candidate).
    # R1/R2/hybrid formalize once for the shared base and edit in place, so the
    # key is never set there -- treat every candidate as formalize-ok for those.
    tracks_formalize_ok = any(
        "formalize_ok" in c
        for m in method_rows
        for c in (m.get("candidate_results") or [])
    )

    n_total = len(eval_rows)
    n_evaluated = 0
    n_with_gt_in_cands_all = 0
    n_with_gt_in_cands_evaluated = 0

    n_problem_solved_any = 0     # >=1 candidate Lean-passed
    n_no_formalize = 0           # all candidates failed formalize
    n_all_lean_fail = 0          # formalize ok but all Lean failed
    n_ambiguous = 0              # >1 candidates Lean-passed
    n_first_correct = 0
    n_first_wrong = 0
    n_unique_correct = 0
    n_unique_wrong = 0
    n_gt_passed = 0              # GT candidate is among Lean-passing candidates
    n_candidate_results = 0
    n_candidate_lean_passed = 0

    total_formalize_time = 0.0
    total_replace_time = 0.0
    total_lean_time = 0.0

    # Two-level amortization counters
    total_gemini_calls = 0       # candidate-generation API calls
    total_formalize_calls = 0    # Kimina formalization calls
    total_effective_formalize_calls = 0
    total_model_inf_calls = 0    # SFT model inference calls (B2)
    total_lean_calls = 0

    for idx, eval_row in eval_by_idx.items():
        gt = eval_row.get("gt_answer")
        candidates = eval_row.get("candidate_answers") or []
        gt_in = gt in candidates
        if gt_in:
            n_with_gt_in_cands_all += 1

        m = method_by_idx.get(idx)
        n_evaluated += 1
        if gt_in:
            n_with_gt_in_cands_evaluated += 1

        if m is None:
            # Method never produced a result for this problem (e.g. brute-force
            # formalization failed for every candidate, so there was no base to
            # edit). Count it as a full failure rather than excluding it, so
            # every method is scored against the same fixed population.
            n_no_formalize += 1
            continue

        cand_results = m.get("candidate_results") or []
        n_passing = sum(1 for c in cand_results if c.get("lean_passed"))
        if tracks_formalize_ok:
            n_formalize_ok = sum(1 for c in cand_results if c.get("formalize_ok"))
        else:
            n_formalize_ok = len(cand_results)
        gt_passed = any(c.get("is_gt") and c.get("lean_passed") for c in cand_results)
        n_gt_passed += bool(gt_passed)
        n_candidate_results += len(cand_results)
        n_candidate_lean_passed += n_passing

        for c in cand_results:
            total_formalize_time += float(c.get("formalize_time_s") or 0.0)
            total_replace_time += float(c.get("replace_time_s") or 0.0)
            total_lean_time += float(c.get("lean_time_s") or 0.0)
        row_gemini_calls = int(m.get("gemini_calls", 0) or 0)
        row_formalize_calls = int(m.get("formalize_calls", 0) or 0)
        total_gemini_calls += row_gemini_calls
        total_formalize_calls += row_formalize_calls
        total_model_inf_calls += int(m.get("model_inf_calls", 0) or 0)
        total_lean_calls += int(m.get("lean_calls", 0) or len(cand_results))

        # For best-base / hybrid eval sets, the method row's formalize_calls only
        # records the selected base as "already available". The actual method
        # cost is the number of candidate formalizations needed to find that
        # base, stored in eval_row.best_base.formalize_calls_until_base.
        row_method = str(m.get("method_name", "")).lower()
        best_base = eval_row.get("best_base") or {}
        is_hybrid_method = (("best_base" in row_method) or ("hybrid" in row_method)) and "brute" not in row_method
        if is_hybrid_method and best_base:
            total_effective_formalize_calls += int(
                best_base.get("formalize_calls_until_base", row_formalize_calls) or 0
            )
        else:
            total_effective_formalize_calls += row_formalize_calls

        if n_formalize_ok == 0:
            n_no_formalize += 1
        elif n_passing == 0:
            n_all_lean_fail += 1
        else:
            n_problem_solved_any += 1
            if n_passing > 1:
                n_ambiguous += 1

        first = select_first_pass(cand_results)
        if first is not None:
            if str(first["candidate"]) == str(gt):
                n_first_correct += 1
            else:
                n_first_wrong += 1
        unique = select_unique_pass(cand_results)
        if unique is not None:
            if str(unique["candidate"]) == str(gt):
                n_unique_correct += 1
            else:
                n_unique_wrong += 1

    def safe_div(a, b): return a / b if b else 0.0
    n = max(n_evaluated, 1)
    return {
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "n_with_gt_in_cands": n_with_gt_in_cands_evaluated,
        "n_with_gt_in_cands_all": n_with_gt_in_cands_all,
        "gt_in_topk_rate": safe_div(n_with_gt_in_cands_evaluated, n_evaluated),
        "gt_in_topk_rate_all": safe_div(n_with_gt_in_cands_all, n_total),
        "problem_solved_any": n_problem_solved_any,
        "pass_at_k": safe_div(n_problem_solved_any, n_evaluated),
        "gt_passed": n_gt_passed,
        "gt_pass_rate": safe_div(n_gt_passed, n_evaluated),
        "gt_pass_rate_conditional": safe_div(n_gt_passed, n_with_gt_in_cands_evaluated),
        "candidate_lean_passed": n_candidate_lean_passed,
        "candidate_total": n_candidate_results,
        "candidate_lean_pass_rate": safe_div(n_candidate_lean_passed, n_candidate_results),
        "no_formalize": n_no_formalize,
        "all_lean_fail": n_all_lean_fail,
        "ambiguous_pass": n_ambiguous,
        "first_correct": n_first_correct,
        "first_wrong": n_first_wrong,
        "unique_correct": n_unique_correct,
        "unique_wrong": n_unique_wrong,
        # Fair denominator: use n_total so brute (n=500) and hybrid (n=494) are comparable.
        # Problems that were not evaluated by a method count as failures for that method.
        "first_acc_overall": safe_div(n_first_correct, n_total),
        "first_acc_conditional": safe_div(n_first_correct, n_with_gt_in_cands_evaluated),
        "unique_acc_overall": safe_div(n_unique_correct, n_total),
        "unique_acc_conditional": safe_div(n_unique_correct, n_with_gt_in_cands_evaluated),
        "total_formalize_time_s": total_formalize_time,
        "total_replace_time_s": total_replace_time,
        "total_lean_time_s": total_lean_time,
        "total_time_s": total_formalize_time + total_replace_time + total_lean_time,
        "avg_time_per_problem_s": safe_div(
            total_formalize_time + total_replace_time + total_lean_time, n_evaluated
        ),
        # Two-level amortization metrics (per problem averages)
        "gemini_calls_per_problem": total_gemini_calls / n,
        "formalize_calls_per_problem": total_formalize_calls / n,
        "effective_formalize_calls_per_problem": total_effective_formalize_calls / n,
        "model_inf_calls_per_problem": total_model_inf_calls / n,
        "lean_calls_per_problem": total_lean_calls / n,
        "total_gemini_calls": total_gemini_calls,
        "total_formalize_calls": total_formalize_calls,
        "total_effective_formalize_calls": total_effective_formalize_calls,
        "total_lean_calls": total_lean_calls,
    }


def infer_answer_api_calls_per_problem(method_name: str, metrics: dict) -> float:
    """Estimate answer-generation API calls per problem.

    The eval files already contain candidate_answers, so this is an analytic
    cost model rather than a measured runtime. For the paper efficiency claim:
      brute-force one-by-one baseline: K answer calls per problem
      ours: one call returns K candidates

    If a method result explicitly records gemini_calls, keep that as a measured
    lower-level counter elsewhere, but this inferred value is the headline
    answer-generation call count.
    """
    name = method_name.lower()
    if "brute" in name:
        return metrics["lean_calls_per_problem"]
    return 1.0 if metrics["n_evaluated"] else 0.0


def estimated_costs_per_1k(method_name: str, metrics: dict,
                           gpu_dollar_per_hour: float,
                           api_dollar_per_call: float) -> dict[str, float]:
    answer_calls = infer_answer_api_calls_per_problem(method_name, metrics)
    gpu_hours = (metrics["avg_time_per_problem_s"] * 1000.0) / 3600.0
    api_cost = answer_calls * 1000.0 * api_dollar_per_call
    gpu_cost = gpu_hours * gpu_dollar_per_hour
    return {
        "answer_calls_per_problem": answer_calls,
        "gpu_hours_per_1k": gpu_hours,
        "api_cost_per_1k": api_cost,
        "gpu_cost_per_1k": gpu_cost,
        "total_cost_per_1k": api_cost + gpu_cost,
    }


def render_main_table_md(rows: list[dict], gpu_dollar_per_hour: float = 1.5) -> str:
    """Paper main table for answer-selection.

    Columns (per user spec):
        Method | GT@K | Acc-overall | Acc | GT@K | Avg time/prob | Speedup

    Claim: comparable Acc|GT@K at X× speedup.
    """
    baseline_time = None
    for r in rows:
        if "brute" in r["method_name"].lower():
            baseline_time = r["metrics"]["avg_time_per_problem_s"]
            break

    headers = [
        "Method", "n",
        "GT@K (cov)",
        "Acc-overall", "Acc-cond",
        "Avg time/prob (s)", "Speedup",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        avg_t = m["avg_time_per_problem_s"]
        speedup = (baseline_time / avg_t) if (baseline_time and avg_t) else float("nan")
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['n_evaluated']}",
            f"{m['gt_in_topk_rate']*100:.1f}%",
            f"{m['first_acc_overall']*100:.1f}%",
            f"{m['first_acc_conditional']*100:.1f}%",
            f"{avg_t:.1f}",
            f"{speedup:.2f}×" if speedup == speedup else "—",
        ]) + " |")
    return "\n".join(lines)


def render_paper_accuracy_table_md(rows: list[dict]) -> str:
    """Paper Table 1: answer-selection accuracy/effectiveness."""
    headers = [
        "Method", "n", "GT@K",
        "Acc-overall", "Acc-cond",
        "Pass@K", "GT-pass", "Per-cand pass",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['n_evaluated']}",
            f"{m['gt_in_topk_rate']*100:.1f}%",
            f"{m['first_acc_overall']*100:.1f}%",
            f"{m['first_acc_conditional']*100:.1f}%",
            f"{m['pass_at_k']*100:.1f}%",
            f"{m['gt_pass_rate_conditional']*100:.1f}%",
            f"{m['candidate_lean_pass_rate']*100:.1f}%",
        ]) + " |")
    return "\n".join(lines)


def render_paper_cost_table_md(rows: list[dict]) -> str:
    """Paper Table 2: method-level cost/efficiency."""
    baseline_formalizer = None
    for r in rows:
        if "brute" in r["method_name"].lower():
            baseline_formalizer = r["metrics"].get("effective_formalize_calls_per_problem")
            break
    if not baseline_formalizer:
        baseline_formalizer = rows[0]["metrics"].get("effective_formalize_calls_per_problem", 0) if rows else 0

    headers = [
        "Method", "Formalizer calls/prob", "Relative formalizer cost",
        "Model calls/prob", "Lean checks/prob", "GPU-h/1k",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        m = r["metrics"]
        formalizer = float(m.get("effective_formalize_calls_per_problem", 0.0) or 0.0)
        rel = (formalizer / baseline_formalizer) if baseline_formalizer else 0.0
        gpu_hours = (m["avg_time_per_problem_s"] * 1000.0) / 3600.0
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{formalizer:.2f}",
            f"{rel:.2f}×",
            f"{m['model_inf_calls_per_problem']:.2f}",
            f"{m['lean_calls_per_problem']:.2f}",
            f"{gpu_hours:.2f}",
        ]) + " |")
    return "\n".join(lines)


def render_reliability_table_md(rows: list[dict]) -> str:
    """Reliability / failure-mode table.

    Columns: Method | No pass | One pass | Multi pass | Ambiguous rate
    """
    headers = ["Method", "n", "No pass", "One pass", "Multi pass", "Ambiguous rate"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        n = max(m["n_evaluated"], 1)
        no_pass = m["n_evaluated"] - m["problem_solved_any"]
        one_pass = m["problem_solved_any"] - m["ambiguous_pass"]
        multi_pass = m["ambiguous_pass"]
        ambig_rate = multi_pass / n
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['n_evaluated']}",
            f"{no_pass}",
            f"{one_pass}",
            f"{multi_pass}",
            f"{ambig_rate*100:.1f}%",
        ]) + " |")
    return "\n".join(lines)


def render_lean_pass_table_md(rows: list[dict]) -> str:
    """Lean-pass coverage table independent of answer-selection strategy."""
    headers = [
        "Method", "n",
        "Pass@K", "GT-pass", "Per-cand pass",
        "GT passed / n", "Cand passed / total",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['n_evaluated']}",
            f"{m['pass_at_k']*100:.1f}%",
            f"{m['gt_pass_rate_conditional']*100:.1f}%",
            f"{m['candidate_lean_pass_rate']*100:.1f}%",
            f"{m['gt_passed']}/{m['n_evaluated']}",
            f"{m['candidate_lean_passed']}/{m['candidate_total']}",
        ]) + " |")
    return "\n".join(lines)


def render_amortization_breakdown_md(rows: list[dict]) -> str:
    """Where the savings come from: Gemini/Formalize/ModelInf calls per problem."""
    headers = ["Method",
               "Answer API calls/prob", "Formalize calls/prob",
               "Model inf/prob", "Lean/prob"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{infer_answer_api_calls_per_problem(r['method_name'], m):.2f}",
            f"{m.get('effective_formalize_calls_per_problem', m['formalize_calls_per_problem']):.2f}",
            f"{m['model_inf_calls_per_problem']:.2f}",
            f"{m['lean_calls_per_problem']:.2f}",
        ]) + " |")
    return "\n".join(lines)


def render_extended_main_table_md(rows: list[dict], gpu_dollar_per_hour: float = 1.5,
                                  api_dollar_per_call: float = 0.05) -> str:
    """Wider variant with cost + both accuracy metrics."""
    baseline_time = None
    for r in rows:
        if "brute" in r["method_name"].lower():
            baseline_time = r["metrics"]["avg_time_per_problem_s"]
            break

    headers = [
        "Method", "n",
        "Answer API/prob", "Formalize/prob", "ModelInf/prob",
        "Avg time/prob (s)", "Speedup",
        "GPU-h/1k", "API $/1k", "GPU $/1k", "Total $/1k",
        "Acc-overall", "Acc-cond (GT in top-K)",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        avg_t = m["avg_time_per_problem_s"]
        speedup = (baseline_time / avg_t) if (baseline_time and avg_t) else float("nan")
        costs = estimated_costs_per_1k(
            r["method_name"], m, gpu_dollar_per_hour, api_dollar_per_call
        )
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['n_evaluated']}",
            f"{costs['answer_calls_per_problem']:.2f}",
            f"{m.get('effective_formalize_calls_per_problem', m['formalize_calls_per_problem']):.2f}",
            f"{m['model_inf_calls_per_problem']:.2f}",
            f"{avg_t:.1f}",
            f"{speedup:.2f}×" if speedup == speedup else "—",
            f"{costs['gpu_hours_per_1k']:.2f}",
            f"${costs['api_cost_per_1k']:.2f}",
            f"${costs['gpu_cost_per_1k']:.2f}",
            f"${costs['total_cost_per_1k']:.2f}",
            f"{m['first_acc_overall']*100:.1f}%",
            f"{m['first_acc_conditional']*100:.1f}%",
        ]) + " |")
    return "\n".join(lines)


def render_compute_cost_table_md(rows: list[dict], gpu_dollar_per_hour: float,
                                 api_dollar_per_call: float) -> str:
    """Compute/money-centric table for efficiency discussion."""
    headers = [
        "Method", "Answer API calls/1k", "Formalizer calls/1k", "Lean calls/1k",
        "GPU-h/1k", "API $/1k", "GPU $/1k", "Total $/1k",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        m = r["metrics"]
        costs = estimated_costs_per_1k(
            r["method_name"], m, gpu_dollar_per_hour, api_dollar_per_call
        )
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{costs['answer_calls_per_problem'] * 1000:.0f}",
            f"{m.get('effective_formalize_calls_per_problem', m['formalize_calls_per_problem']) * 1000:.0f}",
            f"{m['lean_calls_per_problem'] * 1000:.0f}",
            f"{costs['gpu_hours_per_1k']:.2f}",
            f"${costs['api_cost_per_1k']:.2f}",
            f"${costs['gpu_cost_per_1k']:.2f}",
            f"${costs['total_cost_per_1k']:.2f}",
        ]) + " |")
    return "\n".join(lines)


def render_efficiency_breakdown_md(rows: list[dict]) -> str:
    """Per-stage time breakdown — shows where savings come from."""
    headers = ["Method", "formalize (s)", "replace (s)", "lean (s)", "total (s)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        n = max(m["n_evaluated"], 1)
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['total_formalize_time_s']/n:.1f}",
            f"{m['total_replace_time_s']/n:.2f}",
            f"{m['total_lean_time_s']/n:.1f}",
            f"{m['avg_time_per_problem_s']:.1f}",
        ]) + " |")
    return "\n".join(lines)


def render_accuracy_table_md(rows: list[dict]) -> str:
    """Detailed accuracy breakdown (selection strategies + failure modes)."""
    headers = [
        "Method", "n_eval", "GT-in-topK",
        "problem_solved", "ambig",
        "first-pass acc (cond)", "unique-pass acc (cond)",
        "no_formalize", "all_lean_fail",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        lines.append("| " + " | ".join([
            r["method_name"],
            f"{m['n_evaluated']}",
            f"{m['gt_in_topk_rate']*100:.1f}%",
            f"{m['problem_solved_any']}/{m['n_evaluated']}",
            f"{m['ambiguous_pass']}",
            f"{m['first_acc_conditional']*100:.1f}%",
            f"{m['unique_acc_conditional']*100:.1f}%",
            f"{m['no_formalize']}",
            f"{m['all_lean_fail']}",
        ]) + " |")
    return "\n".join(lines)


# Backward-compat alias
def render_table_md(rows: list[dict]) -> str:
    return render_main_table_md(rows)


def main() -> None:
    args = parse_args()
    eval_rows = load_jsonl(Path(args.eval_set))

    # Auto-merge best_base info from sibling file when missing.
    # Two cases:
    #   (a) Q3/R1Q3 pipeline: eval_set is *_best_base_eval.jsonl WITHOUT best_base,
    #       sibling *_hybrid_eval.jsonl has it
    #   (b) Gemini K=10 pipeline: eval_set is *_eval.jsonl WITHOUT best_base,
    #       sibling *_best_base_eval.jsonl has it
    eval_set_path = Path(args.eval_set)
    if eval_rows and not eval_rows[0].get("best_base"):
        candidates = []
        if "best_base_eval.jsonl" in str(eval_set_path):
            candidates.append(Path(str(eval_set_path).replace("best_base_eval.jsonl", "hybrid_eval.jsonl")))
        if str(eval_set_path).endswith("_eval.jsonl") and "best_base_eval" not in str(eval_set_path):
            candidates.append(Path(str(eval_set_path).replace("_eval.jsonl", "_best_base_eval.jsonl")))
        for sibling in candidates:
            if sibling.exists() and sibling != eval_set_path:
                bb_rows = load_jsonl(sibling)
                bb_by_idx = {int(r["problem_index"]): r.get("best_base") for r in bb_rows if r.get("best_base")}
                merged = 0
                for er in eval_rows:
                    idx = int(er["problem_index"])
                    if not er.get("best_base") and bb_by_idx.get(idx):
                        er["best_base"] = bb_by_idx[idx]
                        merged += 1
                if merged:
                    print(f"Merged best_base from {sibling.name}: {merged} rows")
                    break

    by_method = []
    for path in args.method_results:
        method_rows = load_jsonl(Path(path))
        method_name = method_rows[0].get("method_name", Path(path).stem) if method_rows else Path(path).stem
        metrics = compute_method_metrics(eval_rows, method_rows)
        by_method.append({"method_name": method_name, "source": str(path), "metrics": metrics})

    summary = {"eval_set": str(args.eval_set), "by_method": by_method}
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print paper Table 1 (answer-selection accuracy/effectiveness)
    print("\n=== Paper Table 1: Answer-Selection Accuracy ===\n")
    print(render_paper_accuracy_table_md(by_method))
    print()


if __name__ == "__main__":
    main()
