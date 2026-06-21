#!/usr/bin/env python3
"""Evaluate trained SFT fill_answer model.

Generates fill_answer from the trained model on the test set, executes the
function on ranked_answers from the problem pool, substitutes into
formal_problem using original_local_block, Lean-verifies each reconstruction,
reports pass rates.

Supports two output schemas (auto-detected):
    v1:    assistant target = raw `def fill_answer...` Python code
    json:  assistant target = JSON with {original_local_block, fill_answer_code}

For v1, original_local_block is recovered from --gold-jsonl (Phase 1 labels).
For json, model's predicted original_local_block is used.

Usage:
    python -m base_and_edit.eval_sft_fill_answer \\
      --base-model Qwen/Qwen3-4B-Instruct-2507 \\
      --lora-path outputs/sft/fill_answer_qwen3_4b_b2strict_v1 \\
      --dataset data/base_and_edit/sft_fill_answer_v1 \\
      --gold-jsonl outputsgold_structured_multi.jsonl \\
                   outputsgold_short_label.jsonl \\
      --problem-pool outputsproblem_with_answer1_10000_gemini3_pro.jsonl \\
      --output-jsonl outputseval_sft_v1.jsonl \\
      --summary-json outputseval_sft_v1_summary.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from verify import verify_lean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True,
                        help="Path to SFT-trained LoRA adapter directory.")
    parser.add_argument("--dataset", required=True,
                        help="DatasetDict path (must have a 'test' split).")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gold-jsonl", nargs="+", required=True,
                        help="Phase 1 gold collection JSONLs (for original_local_block fallback).")
    parser.add_argument("--problem-pool", required=True, nargs="+",
                        help="One or more JSONLs with ranked_answers per index. "
                             "Pass multiple to cover both 9k and 70k pool sources.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--num-test-answers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def extract_fill_answer_function(text: str) -> str | None:
    if not text:
        return None
    fenced = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    body = fenced.group(1).strip() if fenced else text.strip()
    if "def fill_answer" not in body:
        return None
    if not fenced:
        m = re.search(
            r"((?:^import [^\n]+\n|^from [^\n]+\n)*\s*def fill_answer\b.+)",
            body, re.MULTILINE | re.DOTALL,
        )
        body = m.group(1).strip() if m else body
    body = re.sub(r"\n```.*$", "", body, flags=re.DOTALL)
    return body.strip()


def extract_json_output(text: str) -> tuple[str | None, str | None]:
    """Parse model output as JSON {original_local_block, fill_answer_code}. Returns (block, code) or (None, None)."""
    if not text:
        return None, None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None, None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, None
    olb = parsed.get("original_local_block")
    fac = parsed.get("fill_answer_code")
    if isinstance(olb, str) and isinstance(fac, str):
        return olb, fac
    return None, None


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


import os as _os
try:
    from func_timeout import func_timeout, FunctionTimedOut
    _SAFE_CALL_TIMEOUT_S = float(_os.environ.get("SAFE_CALL_TIMEOUT_S", "5"))
except ImportError:
    func_timeout = None
    FunctionTimedOut = type("FunctionTimedOut", (Exception,), {})
    _SAFE_CALL_TIMEOUT_S = None


def safe_call(fn, arg: str) -> tuple[str, str]:
    """Call a model-generated function with bounded wall-clock time.

    Mining hot loop runs untrusted, model-generated code. Without a timeout,
    pathological generations (infinite loops or massive allocations) leak the
    parent's RSS and OOM the whole shard. We wrap with func_timeout so each
    invocation is hard-capped (default 5s, overridable via SAFE_CALL_TIMEOUT_S).
    """
    try:
        if func_timeout is not None:
            out = func_timeout(_SAFE_CALL_TIMEOUT_S, fn, args=(arg,))
        else:
            out = fn(arg)
        return (str(out) if out is not None else ""), ""
    except FunctionTimedOut:
        return "", f"TimeoutError: exceeded {_SAFE_CALL_TIMEOUT_S}s"
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def extract_prompt_tag(prompt: str, tag: str) -> str:
    """Extract a prompt tag that appears as a standalone XML-like block.

    The prompt instructions mention strings like "<formal_problem>" in prose.
    A loose regex would start there and capture the instruction text too, which
    makes Lean see English before `import Mathlib`. Match only standalone tags.
    """
    pattern = rf"(?ms)^<{re.escape(tag)}>\s*\n?(.*?)\n?</{re.escape(tag)}>\s*$"
    m = re.search(pattern, prompt)
    return m.group(1).strip() if m else ""


def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit(f"Missing dep: {exc}")

    print(f"Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Loading LoRA from {args.lora_path}...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    ds = load_from_disk(args.dataset)
    if isinstance(ds, dict) or hasattr(ds, "keys") and args.split in ds:
        ds = ds[args.split]

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Test set size: {len(ds)}")

    gold_by_idx: dict[int, dict[str, Any]] = {}
    for fp in args.gold_jsonl:
        with open(fp) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold_by_idx[int(r["index"])] = r

    pool: dict[int, dict[str, Any]] = {}
    for pool_path in args.problem_pool:
        with open(pool_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    idx_pool = int(r["index"])
                    if idx_pool not in pool:
                        pool[idx_pool] = r
    print(f"Pool indices loaded: {len(pool)} from {len(args.problem_pool)} file(s)")

    enriched_rows: list[dict[str, Any]] = []
    batch_codes: list[str] = []
    batch_meta: list[dict[str, Any]] = []

    for ex_i, ex in enumerate(ds):
        idx = int(ex["index"])
        task_family = ex.get("task_family", "unknown")
        user_prompt = ex["messages"][0]["content"]
        gold_assistant = ex["messages"][1]["content"]

        # Generate
        messages = [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else None,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        raw = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        # Try JSON schema first, fall back to v1 raw function
        json_block, json_fn = extract_json_output(raw)
        used_schema = "json" if (json_block and json_fn) else "v1"
        if used_schema == "json":
            function_code = json_fn
            original_block = json_block
        else:
            function_code = extract_fill_answer_function(raw)
            # v1: get original_local_block from gold collection
            original_block = ""
            gold = gold_by_idx.get(idx, {})
            for lbl in gold.get("labels", []):
                if lbl.get("usable") and lbl.get("original_local_block"):
                    original_block = lbl["original_local_block"]
                    break

        # Recover formal_problem.
        # Priority: (1) parse from the user prompt's <formal_problem> tag (works for
        # any dataset, including new scale-up problems not in gold_jsonl).
        # Fallback: (2) reconstruct from Phase 1 gold collection labels.
        formal = ""
        formal = extract_prompt_tag(user_prompt, "formal_problem")
        if not formal or (original_block and original_block not in formal):
            gold = gold_by_idx.get(idx, {})
            for lbl in gold.get("labels", []):
                if lbl.get("usable"):
                    nf = lbl.get("new_formal_problem") or ""
                    rb = lbl.get("rewritten_local_block") or ""
                    ob = lbl.get("original_local_block") or ""
                    if nf and rb and ob:
                        recovered = nf.replace(rb, ob, 1)
                        if original_block in recovered:
                            formal = recovered
                            break

        # Get test answers from pool. Also parse original_answer from the user
        # prompt as a fallback (works for problems not in any pool file).
        pool_row = pool.get(idx, {})
        ranked = pool_row.get("ranked_answers") or []
        original_answer = str(ranked[0]) if ranked else ""
        if not original_answer:
            original_answer = extract_prompt_tag(user_prompt, "original_answer")
        test_answers = [str(a) for a in ranked[1 : 1 + args.num_test_answers]]

        per_row = {
            "index": idx,
            "task_family": task_family,
            "schema": used_schema,
            "raw_output": raw[:1000],
            "function_code": (function_code or "")[:2000],
            "original_local_block": original_block[:500],
            "test_answers": test_answers,
            "function_exec_ok": False,
            "exec_error": "",
            "results": [],
        }

        if not function_code:
            per_row["exec_error"] = "no function extracted"
            enriched_rows.append(per_row)
            continue
        if not original_block:
            per_row["exec_error"] = "no original_local_block"
            enriched_rows.append(per_row)
            continue
        if not formal or original_block not in formal:
            per_row["exec_error"] = "could not recover formal_problem with original_block"
            enriched_rows.append(per_row)
            continue

        fn, err = exec_fill_answer(function_code)
        if fn is None:
            per_row["exec_error"] = err
            enriched_rows.append(per_row)
            continue
        per_row["function_exec_ok"] = True

        for new_answer in test_answers:
            out, call_err = safe_call(fn, new_answer)
            res = {
                "new_answer": new_answer,
                "fn_output": out[:300] if out else "",
                "fn_call_error": call_err,
                "lean_passed": False,
                "lean_reason": "",
            }
            per_row["results"].append(res)
            if out:
                new_formal = formal.replace(original_block, out, 1)
                if new_formal != formal:
                    batch_codes.append(new_formal)
                    batch_meta.append({"row_index": len(enriched_rows), "ans_index": len(per_row["results"]) - 1})

        enriched_rows.append(per_row)
        if (ex_i + 1) % 5 == 0:
            print(f"  generated {ex_i + 1}/{len(ds)}")

    print(f"Lean-verifying {len(batch_codes)} reconstructions...")
    if batch_codes:
        lean_results = verify_lean(batch_codes)
        for meta, lr in zip(batch_meta, lean_results):
            res = enriched_rows[meta["row_index"]]["results"][meta["ans_index"]]
            res["lean_passed"] = bool(lr.get("passed"))
            errs = lr.get("errors") or []
            res["lean_reason"] = json.dumps(errs, ensure_ascii=False)[:300] if errs else ""

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in enriched_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(enriched_rows)
    by_family: dict[str, dict[str, int]] = {}
    total_calls = 0
    pass_calls = 0
    n_exec_ok = 0
    n_any_pass = 0
    n_all_pass = 0
    for r in enriched_rows:
        fam = r["task_family"]
        bf = by_family.setdefault(fam, {"n": 0, "exec_ok": 0, "any_pass": 0, "all_pass": 0,
                                          "total_calls": 0, "pass_calls": 0})
        bf["n"] += 1
        if r["function_exec_ok"]:
            n_exec_ok += 1
            bf["exec_ok"] += 1
        passes = sum(1 for x in r["results"] if x["lean_passed"])
        bf["total_calls"] += len(r["results"])
        bf["pass_calls"] += passes
        total_calls += len(r["results"])
        pass_calls += passes
        if passes > 0:
            n_any_pass += 1
            bf["any_pass"] += 1
        if r["results"] and passes == len(r["results"]):
            n_all_pass += 1
            bf["all_pass"] += 1

    summary = {
        "n_test": n,
        "function_exec_ok": n_exec_ok,
        "any_lean_pass": n_any_pass,
        "all_lean_pass": n_all_pass,
        "per_call_lean_pass": pass_calls,
        "per_call_total": total_calls,
        "per_call_lean_pass_rate": pass_calls / total_calls if total_calls else 0.0,
        "any_pass_rate": n_any_pass / n if n else 0.0,
        "all_pass_rate": n_all_pass / n if n else 0.0,
        "by_family": by_family,
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
