#!/usr/bin/env python3
"""R1: Rule-based answer substitution for the hybrid pipeline.

For each problem, tries to locate original_answer in formal_problem via:
  1. Direct unique locate (no model needed)
  2. Direct multi-occurrence + disambiguation (needs base model)
  3. Translate unique locate (LaTeX→Lean rules, no model needed)
  4. Translate multi-occurrence + disambiguation (needs base model)

Problems where R1 succeeds get their candidates substituted and Lean-verified.
Problems where R1 fails are flagged for R2 (LeanScribe SFT via gen_data.py).

Output dataset has one row per problem with candidate_results[] and a
`needs_r2` flag for downstream processing.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from tqdm import tqdm

from verify import verify_lean


def _ensure_hf_namespace(name: str) -> str:
    if "/" not in name and not Path(name).suffix:
        username = HfApi().whoami()["name"]
        return f"{username}/{name}"
    return name


# ---------------------------------------------------------------------------
# LaTeX -> Lean translation (rule-based, conservative)
# ---------------------------------------------------------------------------
def translate_to_lean_candidates(answer: str) -> list[tuple[str, str]]:
    a = answer.strip()
    out: list[tuple[str, str]] = []

    def add(x: str, rule: str):
        x = x.strip()
        if x and (x, rule) not in out:
            out.append((x, rule))

    add(a, "literal")
    if a.startswith("$") and a.endswith("$"):
        add(a[1:-1].strip(), "strip_dollars")
    add(a.replace("\\{", "{").replace("\\}", "}"), "strip_set_escapes")
    add(re.sub(r"\^", " ^ ", a), "space_power")

    m_eq = re.match(r"^[A-Za-z][A-Za-z0-9_']*\s*=\s*(.+)$", a)
    if m_eq:
        add(m_eq.group(1), "strip_equation_lhs")

    m_num_unit = re.match(r"^([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*[A-Za-z][A-Za-z/ ]+$", a)
    if m_num_unit:
        add(m_num_unit.group(1), "strip_units")

    if "," in a:
        add(a.replace(",", ""), "strip_commas")
        add(a.replace(",", ", "), "comma_space")
        add(a.replace(";", ","), "semicolon_to_comma")

    for m in re.finditer(r"\\sqrt\s*\{([^}]+)\}", a):
        x = m.group(1).strip()
        for repl, rule in [
            (f"Real.sqrt {x}", "sqrt_real"),
            (f"sqrt {x}", "sqrt_bare"),
            (f"Real.sqrt ({x})", "sqrt_real_paren"),
            (f"√{x}", "sqrt_unicode"),
            (f"√({x})", "sqrt_unicode_paren"),
        ]:
            add(a.replace(m.group(0), repl), rule)

    for m in re.finditer(r"\\sqrt\s*\[([^\\]]+)\]\s*\{([^}]+)\}", a):
        n, x = m.group(1).strip(), m.group(2).strip()
        add(a.replace(m.group(0), f"{x} ^ (1 / {n})"), "nth_root_power")
        add(a.replace(m.group(0), f"{x} ^ ((1 : ℝ) / {n})"), "nth_root_real_power")

    for m in re.finditer(r"([+-]?\d+)\s*\\sqrt\s*\{([^}]+)\}", a):
        coef, x = m.group(1), m.group(2).strip()
        add(a.replace(m.group(0), f"{coef} * sqrt {x}"), "coef_sqrt_bare")
        add(a.replace(m.group(0), f"{coef} * Real.sqrt {x}"), "coef_sqrt_real")
        add(a.replace(m.group(0), f"{coef} * √{x}"), "coef_sqrt_unicode")

    for m in re.finditer(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", a):
        num, den = m.group(1).strip(), m.group(2).strip()
        for repl, rule in [
            (f"{num} / {den}", "frac_plain"),
            (f"({num}) / ({den})", "frac_paren"),
            (f"({num} : ℝ) / {den}", "frac_real_cast"),
            (f"({num} : ℚ) / {den}", "frac_rat_cast"),
            (f"(↑{num} : ℝ) / {den}", "frac_up_real_cast"),
            (f"({num} : NNReal) / {den}", "frac_nnreal_cast"),
        ]:
            add(a.replace(m.group(0), repl), rule)

    if "\\pi" in a:
        add(a.replace("\\pi", "Real.pi"), "pi_real")
        add(a.replace("\\pi", "π"), "pi_unicode")

    m_int = re.match(r"^\[\s*([^,;]+)\s*[,;]\s*([^\]]+)\s*\]$", a)
    if m_int:
        lo, hi = m_int.group(1).strip(), m_int.group(2).strip()
        add(f"Icc {lo} {hi}", "interval_icc")
        add(f"Set.Icc {lo} {hi}", "interval_set_icc")
        add(f"Set.Ioo {lo} {hi}", "interval_set_ioo")

    return out


def translate_to_lean(answer: str) -> list[str]:
    return [expr for expr, _rule in translate_to_lean_candidates(answer)]


# ---------------------------------------------------------------------------
# Occurrence helpers
# ---------------------------------------------------------------------------
def code_spans_excluding_comments(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    start = 0
    n = len(text)
    while i < n:
        if text.startswith("/-", i):
            if start < i:
                spans.append((start, i))
            j = text.find("-/", i + 2)
            i = n if j < 0 else j + 2
            start = i
            continue
        if text.startswith("--", i):
            if start < i:
                spans.append((start, i))
            j = text.find("\n", i + 2)
            i = n if j < 0 else j + 1
            start = i
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    return spans


def find_occurrences_outside_comments(text: str, sub: str) -> list[int]:
    if not sub:
        return []
    out: list[int] = []
    sub_starts_alnum = sub[0].isalnum()
    sub_ends_alnum = sub[-1].isalnum()
    for s, e in code_spans_excluding_comments(text):
        start = s
        while True:
            i = text.find(sub, start, e)
            if i < 0:
                break
            before = text[i - 1] if i > 0 else ""
            after = text[i + len(sub)] if i + len(sub) < len(text) else ""
            # Reject matches embedded inside a longer alnum run, e.g. "9"
            # falsely matching inside "196" -- these aren't real occurrences
            # of the (whole) answer token, just accidental substrings.
            if not ((sub_starts_alnum and before.isalnum()) or
                    (sub_ends_alnum and after.isalnum())):
                out.append(i)
            start = i + len(sub)
    return out


def is_in_theorem_body(text: str, pos: int) -> bool:
    m_start = re.search(r"\b(theorem|lemma|example)\b", text)
    m_end = re.search(r":=\s*by\b", text)
    if not m_start or not m_end:
        return False
    return m_start.start() <= pos < m_end.start()


def filter_in_theorem_body(text: str, positions: list[int]) -> list[int]:
    return [p for p in positions if is_in_theorem_body(text, p)]


def mark_occurrence(text: str, sub: str, pos: int, marker: str = "__DISAMBIG_TARGET__") -> str:
    return text[:pos] + marker + text[pos + len(sub):]


def format_occurrences(formal_problem: str, sub: str, indices: list[int], ctx: int = 40) -> str:
    lines = []
    for i, pos in enumerate(indices):
        l = max(0, pos - ctx)
        r = min(len(formal_problem), pos + len(sub) + ctx)
        snippet = formal_problem[l:pos] + "[" + sub + "]" + formal_problem[pos + len(sub):r]
        snippet = snippet.replace("\n", " ")
        lines.append(f"  [{i}] ...{snippet}...")
    return "\n".join(lines)


def safe_direct_candidate(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    if any(tok in a for tok in ["\\", "√", "π", "∞", "≤", "≥", "≠", "∈"]):
        return False
    if re.search(r"\b(and|or|text|solution|answer|choice)\b", a, re.IGNORECASE):
        return False
    if any(ch in a for ch in [",", ";", "{", "}", "[", "]"]):
        return False
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", a):
        return True
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", a):
        return True
    allowed_words = {"sqrt", "Real.sqrt", "Nat.sqrt", "Int.sqrt", "abs", "Real.pi"}
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*", a)
    if words and not all(w in allowed_words for w in words):
        return False
    return bool(re.fullmatch(r"[0-9A-Za-z_'.:+\-*/^()\s]+", a))


# ---------------------------------------------------------------------------
# Disambiguation prompt (for multi-occurrence cases)
# ---------------------------------------------------------------------------
DISAMBIG_PROMPT = r"""You are given a Lean 4 formal problem statement. The answer string `{answer}` appears at multiple positions. Decide which occurrence is the *answer slot* — i.e., the position where the answer would be replaced if the candidate answer changes.

Return ONLY a single integer: the 0-based index of the correct occurrence.

Formal problem:
{formal_problem}

Each occurrence shown with surrounding context (40 chars each side):
{occurrences}

Which occurrence (0..{n_minus_1}) is the answer slot? Reply with just the integer."""


# ---------------------------------------------------------------------------
# Core R1 pipeline
# ---------------------------------------------------------------------------
def run_r1(
    ds: Dataset,
    gen_fn=None,
    strict_direct_gate: bool = True,
    safe_direct_candidates_only: bool = True,
    no_translate_route: bool = True,
    base_shortcircuit: bool = True,
    save_new_formal: bool = True,
    method_name: str = "hybrid",
) -> list[dict[str, Any]]:
    """Run R1 rule-based routing on the dataset.

    Args:
        ds: Dataset with formal_problem, original_answer, candidate_answers, etc.
        gen_fn: Optional function(prompt, max_tokens) -> str for disambiguation.
                If None, multi-occurrence cases fall through to R2.
        strict_direct_gate: Only match occurrences inside theorem body.
        safe_direct_candidates_only: Only paste safe (Lean-like) candidates directly.
        no_translate_route: Skip LaTeX→Lean translate route.
        base_shortcircuit: Short-circuit when candidate == original_answer.
        save_new_formal: Store reconstructed Lean statements.
        method_name: Method name for output rows.

    Returns:
        List of per-problem result dicts with candidate_results[] and needs_r2 flag.
    """
    MARKER = "__DISAMBIG_TARGET__"

    all_lean_codes = []
    all_lean_meta = []
    per_row = []

    eval_rows = list(ds)

    for row_idx, row in enumerate(tqdm(eval_rows, desc="R1")):
        pi = row.get("problem_index", row_idx)
        formal = row.get("formal_problem", "") or ""
        original_answer = str(row.get("original_answer", ""))
        gt = str(row.get("gt_answer", row.get("answer", "")))
        candidates = [str(c) for c in (row.get("candidate_answers", row.get("candidates", [])) or [])]
        problem = row.get("problem", row.get("informal_problem", ""))

        route = "unknown"
        marked_formal = None

        if not formal or not original_answer or not candidates:
            per_row.append({
                "problem_index": pi,
                "problem": problem,
                "method_name": method_name,
                "route": "skip::no_base",
                "needs_r2": False,
                "candidates": candidates,
                "candidate_answers": candidates,
                "gt_answer": gt,
                "original_answer": original_answer,
                "formal_problem": formal,
                "candidate_results": [
                    {"candidate": c, "is_gt": c == gt, "lean_passed": False, "route": "skip::no_base"}
                    for c in candidates
                ],
            })
            continue

        # ---- Step A: direct locate ----
        if original_answer in formal:
            occ_raw = find_occurrences_outside_comments(formal, original_answer)
            occ = filter_in_theorem_body(formal, occ_raw) if strict_direct_gate else occ_raw

            if len(occ) == 1:
                route = "rule::direct_unique"
                marked_formal = mark_occurrence(formal, original_answer, occ[0], MARKER)
            elif len(occ) > 1 and gen_fn is not None:
                route = "rule::direct_multi+disambig"
                d_prompt = DISAMBIG_PROMPT.format(
                    answer=original_answer, formal_problem=formal,
                    occurrences=format_occurrences(formal, original_answer, occ),
                    n_minus_1=len(occ) - 1,
                )
                raw = gen_fn(d_prompt, 64)
                m = re.search(r"\d+", raw)
                pick = int(m.group(0)) if m else 0
                pick = max(0, min(pick, len(occ) - 1))
                marked_formal = mark_occurrence(formal, original_answer, occ[pick], MARKER)

        # ---- Step B: translate locate ----
        if route == "unknown" and not no_translate_route:
            for cand_expr in translate_to_lean(original_answer):
                if cand_expr == original_answer:
                    continue
                if cand_expr and cand_expr in formal:
                    occ_raw = find_occurrences_outside_comments(formal, cand_expr)
                    occ = filter_in_theorem_body(formal, occ_raw) if strict_direct_gate else occ_raw

                    if len(occ) == 1:
                        route = "rule::translate_unique"
                        marked_formal = mark_occurrence(formal, cand_expr, occ[0], MARKER)
                        break
                    elif len(occ) > 1 and gen_fn is not None:
                        route = "rule::translate_multi+disambig"
                        d_prompt = DISAMBIG_PROMPT.format(
                            answer=cand_expr, formal_problem=formal,
                            occurrences=format_occurrences(formal, cand_expr, occ),
                            n_minus_1=len(occ) - 1,
                        )
                        raw = gen_fn(d_prompt, 64)
                        m = re.search(r"\d+", raw)
                        pick = int(m.group(0)) if m else 0
                        pick = max(0, min(pick, len(occ) - 1))
                        marked_formal = mark_occurrence(formal, cand_expr, occ[pick], MARKER)
                        break

        # Immutable gate for the whole row: only "unknown" routes (R1 couldn't
        # locate the answer at all) skip every candidate here. A per-candidate
        # safe_direct_fallback (below) must NOT cascade into this gate -- it
        # only affects the one unsafe candidate, not later candidates in the
        # same row (matches run_eval_ours_hybrid.py's per-candidate handling).
        row_needs_r2 = (route == "unknown")
        any_candidate_needs_r2 = row_needs_r2

        # ---- Candidate-level substitution ----
        cand_results = []
        for cand_i, cand in enumerate(candidates):
            cr: dict[str, Any] = {
                "candidate": cand,
                "is_gt": cand == gt,
                "lean_passed": False,
                "route": route if not row_needs_r2 else "pending_r2",
            }

            if row_needs_r2:
                cand_results.append(cr)
                continue

            if base_shortcircuit and cand == original_answer:
                cr["lean_passed"] = True
                cr["route"] = route + "+base_cached"
                if save_new_formal:
                    cr["new_formal"] = formal
                cand_results.append(cr)
                continue

            if safe_direct_candidates_only and route.startswith("rule::direct") and not safe_direct_candidate(cand):
                cr["route"] = "pending_r2+safe_direct_fallback"
                cr["needs_r2"] = True
                any_candidate_needs_r2 = True
                cand_results.append(cr)
                continue

            if route.startswith("rule::translate"):
                cand_translations = translate_to_lean(cand)
                new_cand_block = cand_translations[0] if cand_translations else cand
            else:
                new_cand_block = cand

            new_formal = marked_formal.replace(MARKER, new_cand_block, 1)
            if save_new_formal:
                cr["new_formal"] = new_formal
            all_lean_codes.append(new_formal)
            all_lean_meta.append((row_idx, cand_i))
            cand_results.append(cr)

        per_row.append({
            "problem_index": pi,
            "problem": problem,
            "method_name": method_name,
            "route": route,
            "needs_r2": any_candidate_needs_r2,
            "candidates": candidates,
            "candidate_answers": candidates,
            "gt_answer": gt,
            "original_answer": original_answer,
            "formal_problem": formal,
            "candidate_results": cand_results,
        })

    # ---- Lean verification ----
    if all_lean_codes:
        print(f"Lean-verifying {len(all_lean_codes)} R1 reconstructions...")
        lean_results = verify_lean(all_lean_codes)
        for (row_i, cand_i), lr in zip(all_lean_meta, lean_results):
            per_row[row_i]["candidate_results"][cand_i]["lean_passed"] = bool(lr.get("passed"))

    route_counts = Counter(r["route"] for r in per_row)
    print(f"\nR1 route distribution:")
    for rt, n in route_counts.most_common():
        print(f"  {rt}: {n}")
    n_r2 = sum(1 for r in per_row if r["needs_r2"])
    print(f"\nR1 handled: {len(per_row) - n_r2}/{len(per_row)}, needs R2: {n_r2}")

    return per_row


def compute_summary(results: list[dict[str, Any]]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    n_r1_handled = sum(1 for r in results if not r.get("needs_r2"))
    n_any_passed = sum(
        1 for r in results
        if any(c.get("lean_passed") for c in r["candidate_results"])
    )
    n_gt_passed = sum(
        1 for r in results
        if any(c.get("lean_passed") and c.get("is_gt") for c in r["candidate_results"])
    )

    return {
        "num_problems": n,
        "r1_handled": n_r1_handled,
        "needs_r2": n - n_r1_handled,
        "any_candidate_passed": n_any_passed,
        "any_candidate_passed_rate": n_any_passed / n,
        "gt_candidate_passed": n_gt_passed,
        "gt_candidate_passed_rate": n_gt_passed / n,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input dataset (HF repo or JSONL path)")
    parser.add_argument("output", help="Output JSONL path")
    parser.add_argument("--split", default="test")
    parser.add_argument("--method-name", default="hybrid")
    parser.add_argument("--no-translate-route", action="store_true", default=True)
    parser.add_argument("--enable-translate-route", action="store_true")
    parser.add_argument("--no-strict-gate", action="store_true")
    parser.add_argument("--no-safe-direct", action="store_true")
    parser.add_argument("--no-base-shortcircuit", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--disambig-model", default="Qwen/Qwen3-8B",
        help="Base model (no LoRA) used to disambiguate multi-occurrence "
             "direct/translate locates, matching run_eval_ours_hybrid.py's "
             "Step A.2/B.2. Set --no-disambig to skip loading it.",
    )
    parser.add_argument(
        "--no-disambig", action="store_true",
        help="Skip loading the disambiguation model; multi-occurrence rows "
             "fall straight through to R2 instead of being resolved here.",
    )
    parser.add_argument("--max-new-tokens-disambig", type=int, default=64)
    return parser.parse_args()


def build_disambig_gen_fn(model_name: str):
    """Load a base model (no LoRA/adapter) and return a gen_fn(prompt, max_tokens) -> str.

    Mirrors run_eval_ours_hybrid.py's disambiguation call: greedy decoding,
    chat template with thinking disabled.
    """
    print(f"Loading disambiguation model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    def gen_fn(prompt: str, max_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outs = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        return tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)

    return gen_fn


def main() -> None:
    args = parse_args()

    source_path = Path(args.input)
    if source_path.suffix in (".jsonl", ".json"):
        ds = load_dataset("json", data_files={args.split: str(source_path)}, split=args.split)
    else:
        ds = load_dataset(_ensure_hf_namespace(args.input), split=args.split)

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Filter to rows with a base formal_problem
    ds = ds.filter(lambda x: bool(x.get("formal_problem")))
    print(f"Input: {len(ds)} problems with base formal_problem")

    gen_fn = None
    if not args.no_disambig:
        gen_fn = build_disambig_gen_fn(args.disambig_model)

    results = run_r1(
        ds,
        gen_fn=gen_fn,
        strict_direct_gate=not args.no_strict_gate,
        safe_direct_candidates_only=not args.no_safe_direct,
        no_translate_route=args.no_translate_route and not args.enable_translate_route,
        base_shortcircuit=not args.no_base_shortcircuit,
        method_name=args.method_name,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = compute_summary(results)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
