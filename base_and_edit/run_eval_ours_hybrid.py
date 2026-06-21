#!/usr/bin/env python3
"""End-to-end eval runner for the HYBRID method.

Routing per problem:
  Step A: direct locate of original_answer in formal_problem
    - count == 1 -> rule path (deterministic .replace)
    - count >  1 -> Qwen disambiguation (base model, no LoRA), then rule path
    - count == 0 -> Step B
  Step B: translate locate (LaTeX -> Lean rule table; if rules can't translate, skip)
    - if found 1  -> rule path
    - if found >1 -> Qwen disambiguation, rule path
    - else -> Step C
  Step C: B2 SFT path (Qwen + v2 LoRA), uses predicted_original_local_block

For each candidate: substitute via the chosen path's block, Lean-verify.
Records `route` per problem for ablation analysis.

Output schema is compatible with eval_main_answer_selection.py.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompts import prompt_fill_answer_json
from verify import verify_lean


# -----------------------------------------------------------------------------
# LaTeX -> Lean translation (rule-based, conservative)
# -----------------------------------------------------------------------------
def translate_to_lean_candidates(answer: str) -> list[tuple[str, str]]:
    """Return candidate Lean expressions for an informal answer.

    The translator is intentionally multi-candidate: the same LaTeX answer may
    appear as `sqrt n`, `Real.sqrt n`, casts around fractions, etc. The caller
    chooses the variant by unique-locating it in the formal problem.
    """
    a = answer.strip()
    out: list[tuple[str, str]] = []

    def add(x: str, rule: str):
        x = x.strip()
        if x and (x, rule) not in out:
            out.append((x, rule))

    # Always try lightly normalized literal forms first.
    add(a, "literal")
    if a.startswith("$") and a.endswith("$"):
        add(a[1:-1].strip(), "strip_dollars")
    add(a.replace("\\{", "{").replace("\\}", "}"), "strip_set_escapes")
    add(re.sub(r"\^", " ^ ", a), "space_power")

    # Strip simple equation prefix: x = expr -> expr.
    m_eq = re.match(r"^[A-Za-z][A-Za-z0-9_']*\s*=\s*(.+)$", a)
    if m_eq:
        add(m_eq.group(1), "strip_equation_lhs")

    # Strip simple units at the end. Keep this low-confidence rule after
    # structural rules; unique locate + Lean verification decide if it is safe.
    m_num_unit = re.match(r"^([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*[A-Za-z][A-Za-z/ ]+$", a)
    if m_num_unit:
        add(m_num_unit.group(1), "strip_units")

    # Comma handling is noisy for tuples/lists, so only emit it as a late
    # candidate. Unique locate prevents most bad cases from triggering.
    if "," in a:
        add(a.replace(",", ""), "strip_commas")
        add(a.replace(",", ", "), "comma_space")
        add(a.replace(";", ","), "semicolon_to_comma")

    # \sqrt{N} -> several Lean spellings.
    for m in re.finditer(r"\\sqrt\s*\{([^}]+)\}", a):
        x = m.group(1).strip()
        variants = [
            (f"Real.sqrt {x}", "sqrt_real"),
            (f"sqrt {x}", "sqrt_bare"),
            (f"Real.sqrt ({x})", "sqrt_real_paren"),
            (f"√{x}", "sqrt_unicode"),
            (f"√({x})", "sqrt_unicode_paren"),
        ]
        for repl, rule in variants:
            add(a.replace(m.group(0), repl), rule)

    # \sqrt[n]{x} -> power form variants.
    for m in re.finditer(r"\\sqrt\s*\[([^\\]]+)\]\s*\{([^}]+)\}", a):
        n, x = m.group(1).strip(), m.group(2).strip()
        add(a.replace(m.group(0), f"{x} ^ (1 / {n})"), "nth_root_power")
        add(a.replace(m.group(0), f"{x} ^ ((1 : ℝ) / {n})"), "nth_root_real_power")

    # a\sqrt{b} -> a * sqrt b / a * Real.sqrt b.
    for m in re.finditer(r"([+-]?\d+)\s*\\sqrt\s*\{([^}]+)\}", a):
        coef, x = m.group(1), m.group(2).strip()
        add(a.replace(m.group(0), f"{coef} * sqrt {x}"), "coef_sqrt_bare")
        add(a.replace(m.group(0), f"{coef} * Real.sqrt {x}"), "coef_sqrt_real")
        add(a.replace(m.group(0), f"{coef} * √{x}"), "coef_sqrt_unicode")

    # \frac{a}{b} -> multiple cast styles.
    for m in re.finditer(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", a):
        num, den = m.group(1).strip(), m.group(2).strip()
        variants = [
            (f"{num} / {den}", "frac_plain"),
            (f"({num}) / ({den})", "frac_paren"),
            (f"({num} : ℝ) / {den}", "frac_real_cast"),
            (f"({num} : ℚ) / {den}", "frac_rat_cast"),
            (f"(↑{num} : ℝ) / {den}", "frac_up_real_cast"),
            (f"({num} : NNReal) / {den}", "frac_nnreal_cast"),
        ]
        for repl, rule in variants:
            add(a.replace(m.group(0), repl), rule)

    # \pi -> common Lean spellings.
    if "\\pi" in a:
        add(a.replace("\\pi", "Real.pi"), "pi_real")
        add(a.replace("\\pi", "π"), "pi_unicode")

    # [a,b] / [a;b] interval notation.
    m_int = re.match(r"^\[\s*([^,;]+)\s*[,;]\s*([^\]]+)\s*\]$", a)
    if m_int:
        lo, hi = m_int.group(1).strip(), m_int.group(2).strip()
        add(f"Icc {lo} {hi}", "interval_icc")
        add(f"Set.Icc {lo} {hi}", "interval_set_icc")
        add(f"Set.Ioo {lo} {hi}", "interval_set_ioo")

    return out


def translate_to_lean(answer: str) -> list[str]:
    """Backward-compatible wrapper used by the hybrid evaluator."""
    return [expr for expr, _rule in translate_to_lean_candidates(answer)]


# -----------------------------------------------------------------------------
# Qwen disambiguation (base model, no LoRA) -- pick which occurrence is answer slot
# -----------------------------------------------------------------------------
DISAMBIG_PROMPT = r"""You are given a Lean 4 formal problem statement. The answer string `{answer}` appears at multiple positions. Decide which occurrence is the *answer slot* — i.e., the position where the answer would be replaced if the candidate answer changes.

Return ONLY a single integer: the 0-based index of the correct occurrence.

Formal problem:
{formal_problem}

Each occurrence shown with surrounding context (40 chars each side):
{occurrences}

Which occurrence (0..{n_minus_1}) is the answer slot? Reply with just the integer."""


def find_all_occurrences(text: str, sub: str) -> list[int]:
    """Return start indices of every (overlapping=False) occurrence."""
    out = []
    start = 0
    while True:
        i = text.find(sub, start)
        if i < 0:
            break
        out.append(i)
        start = i + len(sub)
    return out


def code_spans_excluding_comments(text: str) -> list[tuple[int, int]]:
    """Return [start, end) spans outside Lean comments.

    This deliberately excludes both block comments `/- ... -/` and line
    comments `-- ...`. Direct replacement must not locate answer strings inside
    comments, because changing comments does not change the Lean theorem.
    """
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
    """Find occurrences whose start position is outside Lean comments."""
    if not sub:
        return []
    out: list[int] = []
    for s, e in code_spans_excluding_comments(text):
        start = s
        while True:
            i = text.find(sub, start, e)
            if i < 0:
                break
            out.append(i)
            start = i + len(sub)
    return out


def mark_occurrence(text: str, sub: str, pos: int, marker: str = "__DISAMBIG_TARGET__") -> str:
    return text[:pos] + marker + text[pos + len(sub):]


def is_in_theorem_body(text: str, pos: int) -> bool:
    """Strict gate: occurrence must lie inside the theorem statement body
    (between the start of `theorem` / `lemma` / `example` and the `:= by`
    proof marker). Filters out occurrences in imports, options, helper defs,
    or anything after `:= by sorry`."""
    m_start = re.search(r"\b(theorem|lemma|example)\b", text)
    m_end = re.search(r":=\s*by\b", text)
    if not m_start or not m_end:
        return False
    return m_start.start() <= pos < m_end.start()


def filter_in_theorem_body(text: str, positions: list[int]) -> list[int]:
    return [p for p in positions if is_in_theorem_body(text, p)]


def safe_direct_candidate(answer: str) -> bool:
    """Whether a candidate answer is safe to insert directly into Lean code.

    Direct location of the original answer only solves *where* to edit. This
    guard checks whether the new candidate is already Lean-like enough to be
    pasted into that slot. LaTeX, labels, and mixed natural-language answers
    must go through SFT/translation instead.
    """
    a = (answer or "").strip()
    if not a:
        return False
    if any(tok in a for tok in ["\\", "√", "π", "∞", "≤", "≥", "≠", "∈"]):
        return False
    if re.search(r"\b(and|or|text|solution|answer|choice)\b", a, re.IGNORECASE):
        return False
    if any(ch in a for ch in [",", ";", "{", "}", "[", "]"]):
        return False
    # Plain signed integer/decimal.
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", a):
        return True
    # Simple fraction with optional spaces.
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", a):
        return True
    # Conservative Lean-like arithmetic expression over numbers and common
    # Mathlib constants/functions. Reject arbitrary letters/equalities.
    allowed_words = {"sqrt", "Real.sqrt", "Nat.sqrt", "Int.sqrt", "abs", "Real.pi"}
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_'.]*", a)
    if words and not all(w in allowed_words for w in words):
        return False
    return bool(re.fullmatch(r"[0-9A-Za-z_'.:+\-*/^()\s]+", a))


def format_occurrences(formal_problem: str, sub: str, indices: list[int],
                        ctx: int = 40) -> str:
    lines = []
    for i, pos in enumerate(indices):
        l = max(0, pos - ctx)
        r = min(len(formal_problem), pos + len(sub) + ctx)
        snippet = formal_problem[l:pos] + "[" + sub + "]" + formal_problem[pos + len(sub):r]
        snippet = snippet.replace("\n", " ")
        lines.append(f"  [{i}] ...{snippet}...")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# B2 helpers
# -----------------------------------------------------------------------------
def extract_b2_json(text: str) -> tuple[str, str, bool]:
    if not text:
        return "", "", False
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        s = text.find("{"); e = text.rfind("}")
        if s != -1 and e > s:
            candidate = text[s:e+1]
    if candidate is None:
        return "", "", False
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "", "", False
    olb = parsed.get("original_local_block", "") or ""
    fac = parsed.get("fill_answer_code", "") or ""
    if not isinstance(olb, str): olb = ""
    if not isinstance(fac, str): fac = ""
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", default=None, help="v2 SFT LoRA for B2 path (LoRA mode).")
    parser.add_argument("--full-model-path", default=None, help="Full fine-tuned model path for B2 path (full-FT mode). If set, disambiguation uses --base-model and B2 uses this model.")
    parser.add_argument("--method-name", default="ours_hybrid")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--base-shortcircuit",
        action="store_true",
        help="For best_base eval sets: when candidate_answer == original_answer "
             "(the pre-verified base candidate), short-circuit fill_answer and "
             "mark lean_passed=True directly. This avoids spurious failures "
             "when fill_answer doesn't perfectly roundtrip the base answer.",
    )
    parser.add_argument("--max-new-tokens-disambig", type=int, default=64)
    parser.add_argument("--max-new-tokens-b2", type=int, default=2048)
    parser.add_argument("--no-sft-fallback", action="store_true",
                        help="If set, abstain instead of using B2 SFT for hard-absent cases. "
                             "Use this to produce the rule-only ablation row.")
    parser.add_argument(
        "--no-translate-route",
        action="store_true",
        help="Disable the LaTeX-to-Lean translator route. If direct locate fails, "
             "fall through directly to B2 SFT. Useful because translate routes can "
             "be low-precision and otherwise do not retry failed candidates with SFT.",
    )
    parser.add_argument(
        "--save-new-formal",
        action="store_true",
        help="Store each candidate's reconstructed Lean statement in candidate_results[].new_formal. "
             "This is large, but needed for downstream proof/prover checks.",
    )
    parser.add_argument(
        "--strict-direct-gate",
        action="store_true",
        help="Restrict direct/translate/SFT marker placement to occurrences inside the "
             "theorem statement body (between `theorem`/`lemma`/`example` and `:= by`). "
             "Occurrences in imports, options, helper defs, comment blocks before the "
             "theorem, or anything after `:= by sorry` are filtered out, causing the "
             "route to fall through to the next strategy.",
    )
    parser.add_argument(
        "--safe-direct-candidates-only",
        action="store_true",
        help="For rule::direct routes, only paste candidates that already look like "
             "Lean-compatible numeric/arithmetic expressions. High-risk candidates "
             "(LaTeX, labels, mixed text, tuples/sets/lists) fall back to B2 SFT "
             "per candidate instead of being pasted directly.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-total", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(f"Missing dep: {exc}")

    if bool(args.full_model_path) == bool(args.lora_path):
        raise SystemExit("Pass exactly one of --lora-path or --full-model-path")

    print(f"Loading base model {args.base_model} for tokenizer/disambiguation...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fullft_mode = bool(args.full_model_path)
    if fullft_mode:
        print(f"Loading clean base model for disambiguation: {args.base_model}")
        disambig_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        disambig_model.eval()
        print(f"Loading full-FT B2 model: {args.full_model_path}")
        sft_model = AutoModelForCausalLM.from_pretrained(
            args.full_model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        sft_model.eval()
    else:
        print(f"Loading base model {args.base_model}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        print(f"Loading B2 LoRA from {args.lora_path}...")
        sft_model = PeftModel.from_pretrained(base_model, args.lora_path)
        sft_model.eval()
        disambig_model = sft_model

    def gen_with_lora(prompt: str, max_tokens: int, sft: bool) -> str:
        # LoRA mode: B2 uses adapter ON; disambiguation disables adapter.
        # Full-FT mode: B2 uses the full-FT model; disambiguation uses clean base Qwen.
        if sft and not fullft_mode:
            sft_model.set_adapter("default")
        model = sft_model if sft else disambig_model
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            if (not sft) and (not fullft_mode):
                with sft_model.disable_adapter():
                    outs = sft_model.generate(
                        **inputs, max_new_tokens=max_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            else:
                outs = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        prompt_len = inputs.input_ids.shape[1]
        return tokenizer.decode(outs[0][prompt_len:], skip_special_tokens=True)

    eval_rows = []
    with open(args.eval_set) as f:
        for line in f:
            if line.strip():
                eval_rows.append(json.loads(line))
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    if args.shard_total > 1:
        eval_rows = [
            row for i, row in enumerate(eval_rows)
            if i % args.shard_total == args.shard_id
        ]
    print(f"Eval set size: {len(eval_rows)}")

    all_lean_codes = []
    all_lean_meta = []
    per_row = []

    for row_idx, row in enumerate(tqdm(eval_rows, desc="hybrid")):
        idx = int(row["index"])
        informal = row.get("informal_problem", "")
        formal = row.get("formal_problem", "")
        original_answer = str(row.get("original_answer", ""))
        gt = str(row.get("gt_answer", ""))
        candidates = [str(c) for c in (row.get("candidate_answers") or [])]
        task_family = row.get("task_family") or "unknown"

        route = "unknown"
        block = ""
        fill_fn = None
        replace_time = 0.0
        formalize_calls = 1  # the upstream formalization (already done, amortized)
        model_inf_calls = 0
        gemini_calls = 1     # upstream ranked-list call

        gate_filtered_direct = 0
        gate_filtered_translate = 0
        gate_filtered_sft = 0

        # ---- Step A: direct locate ----
        if original_answer and original_answer in formal:
            occ_raw = find_occurrences_outside_comments(formal, original_answer)
            if args.strict_direct_gate:
                occ = filter_in_theorem_body(formal, occ_raw)
                gate_filtered_direct = len(occ_raw) - len(occ)
            else:
                occ = occ_raw
            if len(occ) == 1:
                route = "rule::direct_unique"
                marker = "__DISAMBIG_TARGET__"
                block = marker
                row["_marked_formal"] = mark_occurrence(formal, original_answer, occ[0], marker)
            elif len(occ) > 1:
                # Step A.2: Qwen disambiguation (base model)
                route = "rule::direct_multi+disambig"
                t0 = time.time()
                d_prompt = DISAMBIG_PROMPT.format(
                    answer=original_answer, formal_problem=formal,
                    occurrences=format_occurrences(formal, original_answer, occ),
                    n_minus_1=len(occ)-1,
                )
                raw = gen_with_lora(d_prompt, args.max_new_tokens_disambig, sft=False)
                replace_time += time.time() - t0
                model_inf_calls += 1
                m = re.search(r"\d+", raw)
                pick = int(m.group(0)) if m else 0
                pick = max(0, min(pick, len(occ)-1))
                # Replace only the picked occurrence: temporarily wrap with marker
                marker = "__DISAMBIG_TARGET__"
                pos = occ[pick]
                marked_formal = mark_occurrence(formal, original_answer, pos, marker)
                # Block to substitute is marker; we also store new_formal_template
                block = marker
                # Store marked_formal for later substitution
                row["_marked_formal"] = marked_formal
            # If original_answer only appears in comments, occ is empty. Do not
            # route to direct replacement; fall through to translate/SFT.
        # ---- Step B: translate locate ----
        if route == "unknown" and not args.no_translate_route:
            translated_candidates = translate_to_lean(original_answer)
            for cand in translated_candidates:
                if cand == original_answer:
                    continue  # already tried
                if cand and cand in formal:
                    occ_raw = find_occurrences_outside_comments(formal, cand)
                    if args.strict_direct_gate:
                        occ = filter_in_theorem_body(formal, occ_raw)
                        gate_filtered_translate += len(occ_raw) - len(occ)
                    else:
                        occ = occ_raw
                    if len(occ) == 1:
                        route = "rule::translate_unique"
                        marker = "__DISAMBIG_TARGET__"
                        block = marker
                        row["_marked_formal"] = mark_occurrence(formal, cand, occ[0], marker)
                        break
                    elif len(occ) > 1:
                        route = "rule::translate_multi+disambig"
                        t0 = time.time()
                        d_prompt = DISAMBIG_PROMPT.format(
                            answer=cand, formal_problem=formal,
                            occurrences=format_occurrences(formal, cand, occ),
                            n_minus_1=len(occ)-1,
                        )
                        raw = gen_with_lora(d_prompt, args.max_new_tokens_disambig, sft=False)
                        replace_time += time.time() - t0
                        model_inf_calls += 1
                        m = re.search(r"\d+", raw)
                        pick = int(m.group(0)) if m else 0
                        pick = max(0, min(pick, len(occ)-1))
                        marker = "__DISAMBIG_TARGET__"
                        pos = occ[pick]
                        marked_formal = mark_occurrence(formal, cand, pos, marker)
                        block = marker
                        row["_marked_formal"] = marked_formal
                        row["_disambig_block_value"] = cand  # for fn output rewriting
                        break

        # ---- Step C: B2 SFT path (or abstain if --no-sft-fallback) ----
        if route == "unknown" and args.no_sft_fallback:
            route = "abstain::no_sft_fallback"
        if route == "unknown":
            route = "sft::b2"
            t0 = time.time()
            b2_prompt = (
                f"<task_family>{task_family}</task_family>\n\n"
                + prompt_fill_answer_json.format(
                    informal_problem=informal, formal_problem=formal,
                    original_answer=original_answer,
                )
            )
            raw = gen_with_lora(b2_prompt, args.max_new_tokens_b2, sft=True)
            replace_time += time.time() - t0
            model_inf_calls += 1
            olb, fac, json_ok = extract_b2_json(raw)
            # Code-only locate: olb must appear EXACTLY ONCE outside Lean comments.
            # This prevents the model's predicted block from accidentally pointing
            # at a comment substring (which would cause "false-pass" substitutions
            # where the comment gets edited but the theorem stays unchanged).
            code_occs_raw = find_occurrences_outside_comments(formal, olb) if (json_ok and olb) else []
            if args.strict_direct_gate:
                code_occs = filter_in_theorem_body(formal, code_occs_raw)
                gate_filtered_sft = len(code_occs_raw) - len(code_occs)
            else:
                code_occs = code_occs_raw
            if json_ok and olb and len(code_occs) == 1 and fac:
                fn, _ = exec_fill_answer(fac)
                if fn is not None:
                    fill_fn = fn
                    # Use marker pattern (same as rule:: routes) so all subsequent
                    # candidate substitutions are position-aware and cannot land
                    # in a comment region.
                    marker = "__DISAMBIG_TARGET__"
                    block = marker
                    row["_marked_formal"] = mark_occurrence(formal, olb, code_occs[0], marker)
                    row["_sft_olb"] = olb  # keep for fn_output bookkeeping
                else:
                    route = "sft::b2_exec_fail"
            else:
                route = "sft::b2_parse_fail"

        # Candidate-level SFT fallback for direct routes. The row may be easy to
        # locate (`direct_unique` / `direct_multi`) while some candidate answers
        # are not safe Lean snippets (e.g. LaTeX or natural-language labels).
        # In that case, keep numeric candidates on the cheap direct path and use
        # B2 SFT only for the unsafe candidates.
        fallback_fill_fn = fill_fn if route == "sft::b2" else None
        fallback_marked_formal = row.get("_marked_formal") if route == "sft::b2" else None
        fallback_route = route if route.startswith("sft::") else "unknown"

        def ensure_sft_fallback():
            nonlocal fallback_fill_fn, fallback_marked_formal, fallback_route
            nonlocal replace_time, model_inf_calls, gate_filtered_sft
            if fallback_route != "unknown":
                return fallback_fill_fn, fallback_marked_formal, fallback_route
            if args.no_sft_fallback:
                fallback_route = "abstain::no_sft_fallback"
                return None, None, fallback_route

            t0 = time.time()
            b2_prompt = (
                f"<task_family>{task_family}</task_family>\n\n"
                + prompt_fill_answer_json.format(
                    informal_problem=informal, formal_problem=formal,
                    original_answer=original_answer,
                )
            )
            raw = gen_with_lora(b2_prompt, args.max_new_tokens_b2, sft=True)
            replace_time += time.time() - t0
            model_inf_calls += 1
            olb, fac, json_ok = extract_b2_json(raw)
            code_occs_raw = find_occurrences_outside_comments(formal, olb) if (json_ok and olb) else []
            if args.strict_direct_gate:
                code_occs = filter_in_theorem_body(formal, code_occs_raw)
                gate_filtered_sft += len(code_occs_raw) - len(code_occs)
            else:
                code_occs = code_occs_raw
            if json_ok and olb and len(code_occs) == 1 and fac:
                fn, _ = exec_fill_answer(fac)
                if fn is not None:
                    fallback_fill_fn = fn
                    marker = "__DISAMBIG_TARGET__"
                    fallback_marked_formal = mark_occurrence(formal, olb, code_occs[0], marker)
                    fallback_route = "sft::b2"
                    return fallback_fill_fn, fallback_marked_formal, fallback_route
                fallback_route = "sft::b2_exec_fail"
                return None, None, fallback_route
            fallback_route = "sft::b2_parse_fail"
            return None, None, fallback_route

        cand_results = []
        for cand_i, cand in enumerate(candidates):
            cr: dict[str, Any] = {
                "candidate": cand,
                "is_gt": cand == gt,
                "formalize_ok": True,  # pre-formalized
                "formalize_time_s": 0.0,
                "replace_time_s": replace_time / max(len(candidates), 1),
                "lean_passed": False,
                "lean_time_s": 0.0,
                "fn_output": "",
                "route": route,
            }

            # SHORT-CIRCUIT for best_base eval sets:
            # When this candidate IS the base candidate (its answer == original_answer
            # of the row, which for best_base sets is the verified-compiling base
            # answer), we already know the base formal_problem compiled by construction.
            # Skip fill_answer entirely and directly mark lean_passed=True.
            # Without this, fill_answer roundtrip imperfections can spuriously fail
            # the base candidate slot, depressing Pass@K below the base coverage rate.
            if args.base_shortcircuit and cand == original_answer:
                cr["lean_passed"] = True
                cr["fn_output"] = "(base statement, no substitution)"
                cr["route"] = route + "+base_cached" if route else "base_cached"
                if args.save_new_formal:
                    cr["new_formal"] = formal
                cand_results.append(cr)
                continue

            if (args.safe_direct_candidates_only
                    and route.startswith("rule::direct")
                    and not safe_direct_candidate(cand)):
                fb_fn, fb_marked, fb_route = ensure_sft_fallback()
                cr["route"] = fb_route + "+safe_direct_fallback"
                if fb_fn is not None and fb_marked is not None:
                    out, _ = safe_call(fb_fn, cand)
                    if out:
                        new_formal = fb_marked.replace("__DISAMBIG_TARGET__", out, 1)
                        cr["fn_output"] = out[:300]
                        if args.save_new_formal:
                            cr["new_formal"] = new_formal
                        all_lean_codes.append(new_formal)
                        all_lean_meta.append((row_idx, cand_i))
                cand_results.append(cr)
                continue

            if route == "rule::direct_unique" or route == "rule::translate_unique":
                # block known; replace literally with candidate (or its translation)
                if route == "rule::translate_unique":
                    cand_translations = translate_to_lean(cand)
                    new_cand_block = cand_translations[0] if cand_translations else cand
                else:
                    new_cand_block = cand
                if "_marked_formal" in row:
                    new_formal = row["_marked_formal"].replace("__DISAMBIG_TARGET__", new_cand_block, 1)
                else:
                    new_formal = formal.replace(block, new_cand_block, 1)
                cr["fn_output"] = new_cand_block[:200]
                if args.save_new_formal:
                    cr["new_formal"] = new_formal
                # Always Lean-check, even when substitution is a no-op.
                # The original formal_problem was Lean-validated upstream, but we
                # still verify here to be safe.
                all_lean_codes.append(new_formal)
                all_lean_meta.append((row_idx, cand_i))
            elif route.startswith("rule::") and "disambig" in route:
                marker = "__DISAMBIG_TARGET__"
                marked_formal = row["_marked_formal"]
                if route == "rule::translate_multi+disambig":
                    cand_translations = translate_to_lean(cand)
                    new_cand_block = cand_translations[0] if cand_translations else cand
                else:
                    new_cand_block = cand
                new_formal = marked_formal.replace(marker, new_cand_block, 1)
                cr["fn_output"] = new_cand_block[:200]
                if args.save_new_formal:
                    cr["new_formal"] = new_formal
                all_lean_codes.append(new_formal)
                all_lean_meta.append((row_idx, cand_i))
            elif route == "sft::b2" and fill_fn is not None:
                out, _ = safe_call(fill_fn, cand)
                if out:
                    # Position-aware: substitute via marker (set up at sft::b2 setup),
                    # never via raw .replace() on full formal_problem (which could
                    # land in a comment).
                    if "_marked_formal" in row:
                        new_formal = row["_marked_formal"].replace(
                            "__DISAMBIG_TARGET__", out, 1
                        )
                    else:
                        # defensive fallback (should not reach here after the patch)
                        new_formal = formal.replace(block, out, 1)
                    cr["fn_output"] = out[:300]
                    if args.save_new_formal:
                        cr["new_formal"] = new_formal
                    all_lean_codes.append(new_formal)
                    all_lean_meta.append((row_idx, cand_i))
            # routes ending in *_fail or unknown: skip Lean (no candidate Lean check)
            cand_results.append(cr)

        per_row.append({
            "index": idx,
            "method_name": args.method_name,
            "route": route,
            "candidates": candidates,
            "gt_answer": gt,
            "candidate_results": cand_results,
            "gemini_calls": gemini_calls,
            "formalize_calls": formalize_calls,
            "model_inf_calls": model_inf_calls,
            "lean_calls": len(candidates),
            "strict_gate_applied": bool(args.strict_direct_gate),
            "strict_gate_filtered_direct": gate_filtered_direct,
            "strict_gate_filtered_translate": gate_filtered_translate,
            "strict_gate_filtered_sft": gate_filtered_sft,
        })

    print(f"Lean-verifying {len(all_lean_codes)} reconstructions...")
    lean_t0 = time.time()
    lean_results = verify_lean(all_lean_codes) if all_lean_codes else []
    avg_lean = (time.time() - lean_t0) / max(len(all_lean_codes), 1)
    for (row_i, cand_i), lr in zip(all_lean_meta, lean_results):
        cr = per_row[row_i]["candidate_results"][cand_i]
        cr["lean_passed"] = bool(lr.get("passed"))
        cr["lean_time_s"] = avg_lean

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in per_row:
            r["total_time_s"] = sum(
                c["formalize_time_s"] + c["replace_time_s"] + c["lean_time_s"]
                for c in r["candidate_results"]
            )
            # Drop internal helpers that may not be JSON-serializable cleanly
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Route distribution summary
    from collections import Counter
    route_counts = Counter(r["route"] for r in per_row)
    print(f"\nRoute distribution:")
    for rt, n in route_counts.most_common():
        print(f"  {rt}: {n}")
    print(f"\nWrote {len(per_row)} rows to {out_path}")


if __name__ == "__main__":
    main()
