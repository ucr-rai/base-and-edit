#!/usr/bin/env python3
"""Unified pipeline to build the LeanScribe SFT training data from scratch.

Stages:
  Iterative from-scratch order:
    pool → seed → package → train seed-only model
                          → strict → package → train bootstrap model
                                             → hard → final → train final LeanScribe

  pool     Collect math problems from NuminaMath, generate multiple candidate
           answers (via Gemini), and autoformalize each problem into Lean.

  seed     Phase 1 gold (no trained model needed): use Gemini to locate the
           answer-encoding substring in each Lean statement (e.g. "3 * Real.sqrt 13"
           for the answer "3√13"), then synthesize a fill_answer function that can
           rewrite that substring for any new answer, and verify the result compiles
           in Lean.

  package  Combine all available gold sources into one SFT dataset. Missing
           sources are skipped gracefully. Used iteratively — first with only
           seed gold, then again with seed + strict gold, each time producing a
           training set for a stronger model.

  strict   Use the current model to generate many candidate fill_answer functions
           via rejection sampling, keep only those that pass Lean verification
           >= phase2_min_pass times (strict filtering, default ≥4).

  hard     Mine problems the current model still fails on by throwing heavy
           compute at them (32 samples per problem via bootstrap + vLLM). Keep
           anything that passes Lean even once (lenient filtering, ≥1).

  final    Aggregate all gold (seed + strict + hard), deduplicate, and build the
           final SFT dataset for training the production model.

"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from datasets import Dataset, load_dataset
from prompts import prompt_block_rewrite, prompt_fill_answer_json, prompt_fill_answer_synthesis
import yaml

from base_and_edit.eval_sft_fill_answer import extract_json_output
from base_and_edit.run_eval_ours_hybrid import translate_to_lean_candidates
from verify import verify_lean


STAGES = ["pool", "seed", "package", "strict", "hard", "final"]


#Shared helpers

def load_hf_dataset_indexed(repo_id: str) -> dict[int, dict[str, Any]]:
    return {int(r["index"]): r for r in load_dataset(repo_id, split="train")}


def write_hf_dataset(rows: list[dict], repo_id: str) -> None:
    Dataset.from_list(rows).push_to_hub(repo_id, private=True)
    print(f"  Pushed {len(rows)} rows -> Hugging Face dataset at {repo_id}")


def write_summary(summary: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  Wrote summary -> {p}")


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


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


def looks_like_prose(s: str) -> bool:
    _PROSE_INDICATORS = re.compile(
        r"\b(wait|let'?s|let us|note that|however|actually|since|because|the maximum|"
        r"provide|assuming|both possibilities|we have|we get|we need|should be|"
        r"if they|when they|note|observe)\b",
        re.IGNORECASE,
    )
    s = (s or "").strip()
    if len(s) < 15:
        return False
    return bool(_PROSE_INDICATORS.search(s))


def _load_ranked_answers(repo_id: str) -> dict[int, list[str]]:
    """Load a ranked-answers HF dataset and return {index: deduped_answers}."""
    result: dict[int, list[str]] = {}
    for r in load_dataset(repo_id, split="train"):
        deduped = list(dict.fromkeys(str(a).strip() for a in r["ranked_answers"]))
        deduped = [v for v in deduped if v]
        result[int(r["index"])] = deduped
    return result


def _build_rs_input_from_hard_cases(
    hard_rows: list[dict],
    answers_by_idx: dict[int, dict[str, Any]],
    *,
    rs_n: int,
    num_test_answers: int = 4,
) -> tuple[list[dict], int]:
    """Build duplicated rejection sampling input rows from hard cases + candidate answers.

    Returns (input_rows, n_problems).
    """
    input_rows: list[dict] = []
    for r in hard_rows:
        idx = int(r["index"])
        answer_row = answers_by_idx.get(idx)
        if not answer_row:
            continue
        ranked = [str(a) for a in answer_row["ranked_answers"]]
        original = str(r["answer"]).strip()
        test_answers = [a for a in ranked if a.strip() and a != original][:num_test_answers]
        if len(test_answers) < num_test_answers:
            continue
        base_row = {
            "index": idx,
            "informal_problem": r["problem"],
            "formal_problem": r["formal_problem"],
            "original_answer": original,
            "test_answers": test_answers,
            "task_family": r["problem_type"],
            "hard_reason": r["hard_reason"],
        }
        for cand_id in range(rs_n):
            input_rows.append({**base_row, "candidate_id": cand_id})
    return input_rows


def _run_rs_flow(
    *,
    input_rows: list[dict],
    out: str,
    prefix: str,
    args: argparse.Namespace,
    temperature: float,
) -> str:
    """Write input, run gen_data.py, return output repo ID."""
    input_repo = f"{out}_{prefix}_rs_input"
    output_repo = f"{out}_{prefix}_rs_output"
    write_hf_dataset(input_rows, input_repo)
    _call_gen_data(
        base_model=args.base_model, lora_path=args.lora_path,
        input_repo=input_repo, output_repo=output_repo,
        temperature=temperature, max_tokens=2048,
        enable_thinking=False,
    )
    return output_repo


def _run_hard_case_rs(
    *,
    hard_rows: list[dict],
    answers_by_idx: dict[int, dict[str, Any]],
    out: str,
    prefix: str,
    args: argparse.Namespace,
    rs_n: int,
    temperature: float,
) -> list[dict]:
    """Shared rejection-sampling pipeline for hard-case mining (bootstrap & vLLM runs).

    Returns gold rows.
    """
    input_rows = _build_rs_input_from_hard_cases(hard_rows, answers_by_idx, rs_n=rs_n)
    print(f"  Prepared {len(input_rows)} rows ({rs_n} candidates per problem)")
    output_repo = _run_rs_flow(
        input_rows=input_rows, out=out, prefix=prefix, args=args, temperature=temperature,
    )
    print("  Postprocessing...")
    return _postprocess_fill_answer_rs(
        output_repo, parse_fn=_parse_json_output, gold_criteria_fn=_mining_gold_criteria,
    )


def _call_gen_data(
    *, base_model: str, lora_path: str,
    input_repo: str, output_repo: str,
    temperature: float = 0.8, top_p: float | None = 0.95,
    max_tokens: int = 2048, enable_thinking: bool = False,
    batch_size: int = 20480,
) -> None:
    """Write a temp model config, invoke gen_data.py, clean up."""
    config: dict[str, Any] = {
        "r2_fill_answer": {
            "rs": {
                "model": base_model,
                "lora_path": lora_path,
                "post_processing": "none",
                "prompt": "prompts/r2_fill_answer.txt",
                "prompt_file": True,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "sampling_params": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }
        }
    }
    if top_p is not None:
        config["r2_fill_answer"]["rs"]["sampling_params"]["top_p"] = top_p

    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        cmd = [
            sys.executable, "gen_data.py",
            f"{config_path}:r2_fill_answer:rs",
            input_repo, output_repo,
            "--output_key", "r2_fill_answer",
            "--splits", "train",
            "--batch_size", str(batch_size),
        ]
        print(f"  Running gen_data.py ({base_model}+LoRA)...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"  gen_data.py failed (rc={result.returncode})")
    finally:
        Path(config_path).unlink(missing_ok=True)


def _evaluate_candidate(
    r: dict[str, Any],
    cand_id: int,
    formal: str,
    original_answer: str,
    test_answers: list[str],
    parse_fn,
    problem_idx: int,
    batch_codes: list[str],
    batch_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_output = r["r2_fill_answer"]
    olb, fac = parse_fn(raw_output)
    cand: dict[str, Any] = {
        "candidate_id": cand_id, "predicted_original_local_block": olb,
        "fill_answer_code": fac, "exec_ok": False,
        "original_roundtrip_ok": False, "outputs": [],
    }
    if (olb and fac) and formal.count(olb) == 1:
        fn, _ = exec_fill_answer(fac)
        if fn is not None:
            cand["exec_ok"] = True
            o_out, o_err = safe_call(fn, original_answer)
            cand["original_output"] = o_out[:1000] if o_out else ""
            cand["original_roundtrip_ok"] = bool(o_out) and not o_err and o_out == olb
            for ta in test_answers:
                ta_out, ta_err = safe_call(fn, ta)
                cand["outputs"].append({
                    "new_answer": ta, "fn_output": ta_out[:1000] if ta_out else "",
                    "fn_call_error": ta_err, "lean_passed": False,
                })
                if ta_out and not ta_err:
                    new_formal = formal.replace(olb, ta_out, 1)
                    if new_formal != formal:
                        batch_codes.append(new_formal)
                        batch_meta.append({
                            "problem_idx": problem_idx, "cand_id": cand_id,
                            "out_idx": len(cand["outputs"]) - 1,
                        })
    return cand


def _postprocess_fill_answer_rs(
    raw_rows: str | list[dict[str, Any]],
    *,
    parse_fn,
    gold_criteria_fn,
) -> list[dict[str, Any]]:
    """Shared postprocess for fill-answer rejection sampling outputs.

    Groups duplicated rows by index, parses each candidate, execs fill_answer,
    checks roundtrip, batches Lean verification, and extracts gold via
    *gold_criteria_fn*.
    """
    if isinstance(raw_rows, str):
        raw_rows = load_dataset(raw_rows, split="train")
    # Group by index (rows were duplicated for rejection sampling)
    by_index: dict[int, list[dict]] = {}
    for row in raw_rows:
        idx = int(row["index"])
        by_index.setdefault(idx, []).append(row)

    batch_codes: list[str] = []
    batch_meta: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    for idx in sorted(by_index.keys()):
        rows = by_index[idx]
        first = rows[0]
        formal = first["formal_problem"]
        original_answer = first["original_answer"]
        # HF datasets may serialize lists as JSON strings
        ta_raw = first["test_answers"]
        test_answers = json.loads(ta_raw) if isinstance(ta_raw, str) else list(ta_raw)

        candidates = [
            _evaluate_candidate(
                r=r, cand_id=cand_id, formal=formal,
                original_answer=original_answer, test_answers=test_answers,
                parse_fn=parse_fn, problem_idx=len(problems),
                batch_codes=batch_codes, batch_meta=batch_meta,
            )
            for cand_id, r in enumerate(rows)
        ]

        prob = {k: first.get(k, "") for k in _GOLD_PROBLEM_KEYS}
        prob.update(index=idx, formal_problem=formal, original_answer=original_answer,
                    test_answers=test_answers, hard_reason=first.get("hard_reason", ""),
                    candidates=candidates)
        problems.append(prob)

    # Batch Lean verification
    print(f"  Lean-verifying {len(batch_codes)} substitutions...")
    if batch_codes:
        lean_results = verify_lean(batch_codes)
        for meta, lr in zip(batch_meta, lean_results):
            cand = problems[meta["problem_idx"]]["candidates"][meta["cand_id"]]
            cand["outputs"][meta["out_idx"]]["lean_passed"] = bool(lr.get("passed"))

    return gold_criteria_fn(problems)


# --- Parse functions for postprocess ---

def _parse_json_object(raw: str) -> tuple[str, str]:
    parsed = extract_json_object(raw) or {}
    olb = parsed.get("original_local_block") or ""
    fac = parsed.get("fill_answer_code") or ""
    return (olb if isinstance(olb, str) else ""), (fac if isinstance(fac, str) else "")


def _parse_json_output(raw: str) -> tuple[str, str]:
    """Parse JSON-schema model output into (original_local_block, fill_answer_code)."""
    block, fn_code = extract_json_output(raw)
    return block or "", fn_code or ""


# --- Gold criteria functions ---

_GOLD_PROBLEM_KEYS = ("index", "informal_problem", "formal_problem", "original_answer",
                      "task_family", "test_answers")


def _gold_row(p: dict, cand: dict, **extra) -> dict:
    """Build a gold row from a problem dict and a candidate dict."""
    row = {k: p.get(k, "") for k in _GOLD_PROBLEM_KEYS}
    row["original_local_block"] = cand["predicted_original_local_block"]
    row["fill_answer_code"] = cand["fill_answer_code"]
    row.update(extra)
    return row

def _strict_gold_criteria(problems: list[dict]) -> list[dict]:
    """Strict gold: pick best candidate per problem with n_lean_pass >= min_pass."""
    min_pass = 2
    gold: list[dict] = []
    for p in problems:
        best_cand = None
        best_pass = -1
        for cand in p["candidates"]:
            if not cand.get("exec_ok") or not cand.get("original_roundtrip_ok"):
                continue
            n_pass = sum(1 for o in cand["outputs"] if o.get("lean_passed"))
            if n_pass > best_pass:
                best_pass = n_pass
                best_cand = cand
        if best_cand and best_pass >= min_pass:
            gold.append(_gold_row(p, best_cand, n_lean_pass=best_pass))
    return gold


def _mining_gold_criteria(problems: list[dict], min_distinct: int = 2) -> list[dict]:
    """Hard gold: require ALL test answers pass + distinct outputs >= min_distinct."""
    gold: list[dict] = []
    seen: set[int] = set()
    for p in problems:
        idx = p["index"]
        if idx in seen:
            continue
        for cand in p["candidates"]:
            if not cand.get("exec_ok") or not cand.get("original_roundtrip_ok"):
                continue
            outputs = cand["outputs"]
            if not outputs or not all(o.get("lean_passed") for o in outputs):
                continue
            test_outputs = [o["fn_output"] for o in outputs]
            if len(set(test_outputs)) < min_distinct:
                continue
            seen.add(idx)
            gold.append(_gold_row(
                p, cand, hard_reason=p.get("hard_reason", ""),
                test_outputs=test_outputs,
                n_lean_pass=sum(1 for o in outputs if o.get("lean_passed")),
            ))
            break
    return gold


# --- Hook entry points for gen_data.py ---

def fill_answer_strict_postprocess(ds_output):
    """Postprocess hook for strict gold rejection sampling."""
    return _postprocess_fill_answer_rs(
        list(ds_output), parse_fn=_parse_json_object,
        gold_criteria_fn=_strict_gold_criteria,
    )

def fill_answer_strict_summary(results):
    return {"num_gold": len(results)}

def fill_answer_mining_postprocess(ds_output):
    """Postprocess hook for hard-case gold mining."""
    return _postprocess_fill_answer_rs(
        list(ds_output), parse_fn=_parse_json_output,
        gold_criteria_fn=_mining_gold_criteria,
    )

def fill_answer_mining_summary(results):
    return {"num_gold": len(results)}


#Shared SFT dataset builder (used by package and final)


def _build_sft_dataset(
    *,
    out: str,
    hard_gold_path: str,
    hard_source_prefix: str,
    sft_output_dir: str,
    pool_path: str | None = None,
    phase2_gold_path: str | None = None,
    test_size: float = 0.1,
    seed: int = 42,
    require_unseen: bool = False,
    phase2_min_pass: int = 4,
    upsample_short_label: int = 2,
    upsample_hard: int = 1,
) -> None:
    """Build an SFT dataset from phase1 + phase2 + hard gold.

    Shared by 'package' and 'final' stages.
    """
    if phase2_gold_path is None:
        phase2_gold_path = f"{out}_phase2_gold_combined"
    if pool_path is None:
        pool_path = f"{out}_problem_with_answer1"

    pool = load_hf_dataset_indexed(pool_path)

    examples: list[dict[str, Any]] = []
    sft_seen: set[int] = set()
    skipped: Counter = Counter()

    def add_example(idx: int, informal: str, formal: str, original_answer: str,
                    task_family: str, olb: str, fac: str, source: str) -> None:
        body = prompt_fill_answer_json.format(
            informal_problem=informal, formal_problem=formal,
            original_answer=original_answer,
        )
        if task_family and task_family != "unknown":
            body = f"<task_family>{task_family}</task_family>\n\n{body}"
        examples.append({
            "messages": [
                {"role": "user", "content": body},
                {"role": "assistant", "content": json.dumps(
                    {"original_local_block": olb, "fill_answer_code": fac}, ensure_ascii=False,
                )},
            ],
            "index": idx, "task_family": task_family, "source": source,
        })

    def add_gold_from_path(
        path: str,
        source_label_base: str,
        dup_tag: str,
        missing_tag: str,
        is_phase2: bool = False,
        upsample: int = 1,
    ) -> int:
        rows = load_dataset(path, split="train")

        count = 0
        for row in rows:
            idx = int(row["index"])
            if idx in sft_seen:
                skipped[dup_tag] += 1
                continue
            if is_phase2 and int(row.get("n_lean_pass", 0)) < phase2_min_pass:
                skipped["p2_below_min_pass"] += 1
                continue

            informal = row["informal_problem"]
            formal = row["formal_problem"]
            original_answer = row["original_answer"]
            task_family = row["task_family"]
            olb = row["original_local_block"]
            fac = row["fill_answer_code"]

            if not all([informal, formal, original_answer, olb, fac]):
                skipped[missing_tag] += 1
                continue

            if is_phase2:
                src_label = "phase2"
            else:
                combined = row.get("combined_source")
                src_label = f"{source_label_base}:{combined}" if combined else source_label_base

            for _ in range(max(upsample, 1)):
                add_example(idx, informal, formal, original_answer, task_family, olb, fac, src_label)
            sft_seen.add(idx)
            count += 1
        return count

    # --- Phase 1 gold ---
    print("  === Phase 1 gold ===")
    lean_rows = {int(r["index"]): r for r in load_dataset(f"{out}_gold_lean_check", split="train")}
    p1_count = 0
    for p in [f"{out}_gold_structured_multi", f"{out}_gold_short_label"]:
        p_rows = load_dataset(p, split="train")
        for row in p_rows:
            idx = int(row["index"])
            if idx in sft_seen:
                continue
            lean_row = lean_rows.get(idx)
            if lean_row is None:
                skipped["p1_no_lean_row"] += 1; continue
            if not row.get("function_exec_ok"):
                skipped["p1_not_gold"] += 1; continue
            orig = lean_row.get("function_on_original_result")
            if not orig or not orig.get("passed"):
                skipped["p1_not_gold"] += 1; continue
            train_results = lean_row.get("function_on_training_results", [])
            if not train_results or not all(r.get("passed") for r in train_results):
                skipped["p1_not_gold"] += 1; continue
            if require_unseen:
                unseen = lean_row.get("function_on_unseen_results", [])
                if not unseen or not any(r.get("passed") for r in unseen):
                    skipped["p1_not_gold"] += 1; continue
            function_code = row.get("synthesis_function") or ""
            if not function_code.strip():
                skipped["p1_missing_function"] += 1; continue
            pool_row = pool.get(idx, {})
            informal = pool_row.get("problem", "")
            original_answer = row["original_answer"]
            task_family = lean_row["task_family"]
            original_local_block = ""
            recovered_formal = ""
            for lbl in row["labels"]:
                if lbl["usable"] and lbl["original_local_block"]:
                    original_local_block = lbl["original_local_block"]
                    nf = lbl["new_formal_problem"]
                    rb = lbl["rewritten_local_block"]
                    if nf and rb:
                        recovered_formal = nf.replace(rb, original_local_block, 1)
                    break
            formal = pool_row.get("formal_problem") or recovered_formal
            if not all([informal, formal, original_answer, original_local_block]):
                skipped["p1_missing_field"] += 1; continue
            add_example(idx, informal, formal, original_answer, task_family,
                        original_local_block, function_code, "phase1")
            sft_seen.add(idx)
            p1_count += 1
    print(f"    phase1 gold added: {p1_count}")

    # --- Phase 2 strict gold ---
    print("  === Phase 2 strict gold ===")
    print(f"    phase2 gold added: {add_gold_from_path(
        path=phase2_gold_path,
        source_label_base='phase2',
        dup_tag='p2_dup',
        missing_tag='p2_missing_field',
        is_phase2=True,
    )}")

    # --- Hard gold ---
    print(f"  === Hard gold ({hard_source_prefix}) ===")
    print(f"    hard gold added: {add_gold_from_path(
        path=hard_gold_path,
        source_label_base=hard_source_prefix,
        dup_tag='hard_dup',
        missing_tag='hard_missing_field',
        upsample=upsample_hard,
    )}")

    if not examples:
        raise SystemExit("No gold examples after filtering — check input files.")

    print(f"  Total: {len(examples)} examples, {len(sft_seen)} unique indices")
    print(f"  Skipped: {dict(skipped.most_common())}")

    # Upsample short_label
    if upsample_short_label > 1:
        original_count = len(examples)
        short_label_examples = [e for e in examples
                                if e["task_family"] == "short_label_or_symbolic_choice"]
        for _ in range(upsample_short_label - 1):
            examples.extend(short_label_examples)
        print(f"  After upsample_short_label x{upsample_short_label}: "
              f"{len(examples)} (was {original_count})")

    # Build dataset (train/test split)
    dsdict = Dataset.from_list(examples).train_test_split(test_size=test_size, seed=seed)
    print(f"  Split: train={len(dsdict['train'])} test={len(dsdict['test'])}")
    dsdict.push_to_hub(sft_output_dir, private=True)

    write_summary({
        "total_examples": len(examples),
        "unique_indices": len(sft_seen),
        "n_train": len(dsdict["train"]), "n_test": len(dsdict["test"]),
        "by_source": dict(Counter(e["source"] for e in examples).most_common()),
        "by_task_family": dict(Counter(e["task_family"] for e in examples).most_common()),
        "skipped": dict(skipped.most_common()),
    }, f"{sft_output_dir}/build_summary.json")
    print(f"  SFT output: {sft_output_dir}")


def _sft_kwargs(args: argparse.Namespace) -> dict:
    """Extract common SFT dataset builder kwargs from CLI args."""
    return dict(
        test_size=args.test_size,
        require_unseen=args.require_unseen_pass,
        phase2_min_pass=args.phase2_min_pass,
        upsample_short_label=args.upsample_short_label,
        upsample_hard=args.upsample_hard,
    )


# package: Build the bootstrap SFT dataset


def stage_package(out: str, args: argparse.Namespace) -> None:
    """Package whatever gold exists so far into a bootstrap SFT dataset."""
    print("\n=== package: Build bootstrap SFT dataset ===")
    _build_sft_dataset(
        out=out,
        hard_gold_path=f"{out}_bootstrap_hard_gold",
        hard_source_prefix="bootstrap_hard",
        sft_output_dir=args.sft_output_dir or f"{out}_sft_fill_answer_bootstrap",
        **_sft_kwargs(args),
    )


# pool: Build the problem pool


def stage_pool(out: str, args: argparse.Namespace) -> None:
    """Pair NuminaMath problems with answers and autoformalize into Lean."""
    print("\n=== pool: Build the problem pool ===")

    pool_out = f"{out}_problem_with_answer1"

    dataset_name = "AI-MO/NuminaMath-CoT"
    print(f"  Fetching dataset from HF ({dataset_name})...")
    ds = load_dataset(dataset_name, split="train")

    rows = []
    for rec in load_dataset(f"{out}_gemini_answers", split="train"):
        idx = int(rec["index"])
        ranked = [str(x).strip() for x in (rec.get("ranked_answers") or []) if str(x).strip()]
        if not ranked:
            continue
        ex = ds[idx]
        rows.append({
            **ex,
            "index": idx,
            "ranked_answers": ranked,
            "generated_answer": ranked[0],
            "problem_with_answer1": ex["problem"].strip() + "\n\nAnswer: " + ranked[0],
        })

    write_hf_dataset(rows, pool_out)

    # --- Step 2: autoformalize via gen_data.py ---
    formal_out = f"{out}_formal_problem_pool"
    cmd = [
        sys.executable, "gen_data.py",
        "config.yaml:autoformalizers:kimina",
        pool_out,
        formal_out,
        "--input_key", "problem_with_answer1",
        "--output_key", "formal_problem",
        "--splits", "train",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"  gen_data.py formalization failed (rc={result.returncode})")

    print(f"  pool outputs: {pool_out}, {formal_out}")


# seed: Phase 1 gold — labeling + fill-answer synthesis


def _call_gen_data_gemini(
    model_name: str,
    input_repo: str,
    output_repo: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> None:
    """Write a temp Gemini config, invoke gen_data.py, clean up."""
    config: dict[str, Any] = {
        "gemini_run": {
            "model": model_name,
            "post_processing": "none",
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }
    }
    fd, config_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(config, f)
        cmd = [
            sys.executable, "gen_data.py",
            f"{config_path}:gemini_run",
            input_repo,
            output_repo,
            "--input_key", "_prompt",
            "--output_key", "raw_output",
            "--splits", "train",
        ]
        print(f"  Running gen_data.py for Gemini: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"  gen_data.py Gemini call failed (rc={result.returncode})")
    finally:
        Path(config_path).unlink(missing_ok=True)


def stage_seed(out: str, args: argparse.Namespace) -> None:
    """Call Gemini to produce rewrite labels, synthesize fill_answer functions, and Lean-verify them."""
    print("\n=== seed: Phase 1 gold — labeling, synthesis & Lean verification ===")

    model_name = args.gemini_model or "gemini-2.5-pro"
    pool = load_hf_dataset_indexed(f"{out}_problem_with_answer1")
    k_answers = 3
    num_problems = 10000

    # Pick problems
    picked: list[dict[str, Any]] = []
    for idx in pool.keys():
        if len(picked) >= num_problems:
            break
        row = pool.get(idx)
        if not row:
            continue
        ranked = row.get("ranked_answers") or []
        if len(ranked) < k_answers + 1:
            continue
        new_answers_check = [str(a) for a in ranked[1 : 1 + k_answers]]
        if any(looks_like_prose(a) for a in new_answers_check):
            continue
        picked.append(row)

    print(f"  Picked {len(picked)} problems for synthesis.")

    # 1. Generate labeling prompts in batch
    label_inputs: list[dict[str, Any]] = []
    for row in picked:
        idx = int(row["index"])
        informal = row["informal_problem"]
        formal = row["formal_problem"]
        ranked = row["ranked_answers"]
        original_answer = str(ranked[0]).strip()
        new_answers = [str(a).strip() for a in ranked[1 : 1 + k_answers]]

        for new_answer in new_answers:
            prompt = prompt_block_rewrite.format(
                informal_problem=informal, formal_problem=formal,
                original_answer=original_answer, new_answer=new_answer,
            )
            label_inputs.append({
                "index": idx,
                "new_answer": new_answer,
                "_prompt": prompt,
            })

    print(f"  Running batch labeling for {len(label_inputs)} prompts via gen_data.py...")
    label_in_repo = f"{out}_seed_label_input"
    label_out_repo = f"{out}_seed_label_output"
    write_hf_dataset(label_inputs, label_in_repo)

    _call_gen_data_gemini(
        model_name=model_name,
        input_repo=label_in_repo,
        output_repo=label_out_repo,
        temperature=0.0,
        max_tokens=2048,
    )

    # Load labeling outputs and group them by index
    label_outputs = load_dataset(label_out_repo, split="train")
    outputs_by_index: dict[int, list[dict[str, Any]]] = {}
    for r in label_outputs:
        outputs_by_index.setdefault(int(r["index"]), []).append(r)

    # 2. Build synthesis prompts in batch
    results: list[dict[str, Any]] = []
    synth_inputs: list[dict[str, Any]] = []
    per_row_by_index: dict[int, dict[str, Any]] = {}

    for row in picked:
        idx = int(row["index"])
        informal = row["informal_problem"]
        formal = row["formal_problem"]
        ranked = row["ranked_answers"]
        original_answer = str(ranked[0]).strip()
        new_answers = [str(a).strip() for a in ranked[1 : 1 + k_answers]]

        per_row: dict[str, Any] = {
            "index": idx,
            "original_answer": original_answer,
            "new_answers": new_answers,
            "labels": [],
            "usable_labels": 0,
            "synthesis_raw": "",
            "synthesis_function": "",
            "function_exec_ok": False,
            "exec_error": "",
            "n_consistent": 0,
            "all_k_consistent": False,
            "call_details": [],
        }
        per_row_by_index[idx] = per_row

        problem_label_outputs = outputs_by_index.get(idx, [])
        raw_by_ans = {r["new_answer"]: r.get("raw_output") or "" for r in problem_label_outputs}

        usable_labels: list[dict[str, Any]] = []
        for new_answer in new_answers:
            raw = raw_by_ans.get(new_answer, "")
            lbl: dict[str, Any] = {
                "new_answer": new_answer, "usable": False,
                "original_local_block": "", "rewritten_local_block": "",
                "normalized_answer_structure": "", "new_formal_problem": "",
            }
            parsed = extract_json_object(raw)
            if parsed:
                orig = parsed.get("original_local_block") or ""
                rewr = parsed.get("rewritten_local_block") or ""
                norm = parsed.get("normalized_answer_structure") or ""
                if isinstance(orig, str) and isinstance(rewr, str):
                    lbl["original_local_block"] = orig
                    lbl["rewritten_local_block"] = rewr
                    lbl["normalized_answer_structure"] = (
                        norm if isinstance(norm, str) else json.dumps(norm)
                    )
                    if (orig and orig in formal and formal.count(orig) == 1
                            and rewr and rewr != orig):
                        lbl["usable"] = True
                        lbl["new_formal_problem"] = formal.replace(orig, rewr, 1)

            per_row["labels"].append(lbl)
            if lbl["usable"]:
                usable_labels.append(lbl)

        per_row["usable_labels"] = len(usable_labels)
        if len(usable_labels) < 2:
            results.append(per_row)
            continue

        # Synthesize fill_answer function
        lines = []
        for k, ul in enumerate(usable_labels, start=1):
            lines.append(f"Example {k}:")
            lines.append(f"  fill_answer({ul['new_answer']!r})")
            lines.append(f"  must return exactly:")
            lines.append(f"  {ul['rewritten_local_block']!r}")
            lines.append("")
        worked_examples = "\n".join(lines).rstrip()

        synth_prompt = prompt_fill_answer_synthesis.format(
            informal_problem=informal, formal_problem=formal,
            original_answer=original_answer, worked_examples=worked_examples,
        )
        synth_inputs.append({
            "index": idx,
            "_prompt": synth_prompt,
        })

    print(f"  Running batch synthesis for {len(synth_inputs)} prompts via gen_data.py...")
    synth_outputs_by_index: dict[int, str] = {}
    if synth_inputs:
        synth_in_repo = f"{out}_seed_synth_input"
        synth_out_repo = f"{out}_seed_synth_output"
        write_hf_dataset(synth_inputs, synth_in_repo)

        _call_gen_data_gemini(
            model_name=model_name,
            input_repo=synth_in_repo,
            output_repo=synth_out_repo,
            temperature=0.0,
            max_tokens=8192,
        )

        for r in load_dataset(synth_out_repo, split="train"):
            synth_outputs_by_index[int(r["index"])] = r.get("raw_output") or ""

    # Postprocess and evaluate synthesized functions
    for idx, per_row in per_row_by_index.items():
        if per_row["usable_labels"] < 2:
            continue

        synth_raw = synth_outputs_by_index.get(idx, "")
        per_row["synthesis_raw"] = synth_raw

        function_code = extract_fill_answer_function(synth_raw)
        if not function_code:
            results.append(per_row)
            continue
        per_row["synthesis_function"] = function_code

        fn, exec_err = exec_fill_answer(function_code)
        if fn is None:
            per_row["exec_error"] = exec_err
            results.append(per_row)
            continue
        per_row["function_exec_ok"] = True

        usable_labels = [lbl for lbl in per_row["labels"] if lbl["usable"]]
        n_consistent = 0
        call_details = []
        for ul in usable_labels:
            call_result = {"new_answer": ul["new_answer"],
                           "expected": ul["rewritten_local_block"],
                           "got": "", "match": False, "call_error": ""}
            try:
                got = fn(ul["new_answer"])
                call_result["got"] = str(got) if got is not None else ""
                call_result["match"] = (call_result["got"] == ul["rewritten_local_block"])
            except Exception as exc:
                call_result["call_error"] = f"{type(exc).__name__}: {exc}"
            if call_result["match"]:
                n_consistent += 1
            call_details.append(call_result)

        per_row["n_consistent"] = n_consistent
        per_row["all_k_consistent"] = (n_consistent == len(usable_labels))
        per_row["call_details"] = call_details
        results.append(per_row)

    output_repo = f"{out}_gold_structured_multi"
    summary_path = f"{out}/gold_structured_multi_summary.json"
    write_hf_dataset(results, output_repo)

    total_calls = sum(r["usable_labels"] for r in results)
    consistent_calls = sum(r["n_consistent"] for r in results)
    summary = {
        "num_problems_attempted": len(picked),
        "num_with_synthesis": sum(1 for r in results if r["usable_labels"] >= 2),
        "num_function_exec_ok": sum(1 for r in results if r["function_exec_ok"]),
        "num_all_k_consistent": sum(1 for r in results if r["all_k_consistent"]),
        "total_usable_labels": total_calls,
        "consistent_calls": consistent_calls,
        "consistent_call_rate": consistent_calls / total_calls if total_calls else 0.0,
        "model": model_name,
    }
    write_summary(summary, summary_path)

    # Run verification
    print("\n  Running Phase 1 Lean verification...")
    gold_multi = f"{out}_gold_structured_multi"
    gold_short = f"{out}_gold_short_label"
    lean_out = f"{out}_gold_lean_check"
    num_unseen = 2

    gold_sources = [
        (gold_multi, "structured_multi_answer_block"),
        (gold_short, "short_label_or_symbolic_choice"),
    ]
    rows: list[dict[str, Any]] = []
    for p, family in gold_sources:
        p_rows = load_dataset(p, split="train")
        for row in p_rows:
            row["task_family"] = family
            rows.append(row)
    print(f"  Loaded {len(rows)} probe rows.")

    batch_codes: list[str] = []
    batch_meta: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    for row in rows:
        idx = int(row["index"])
        formal = ""
        original_block = ""
        usable_labels = []
        for lbl in row.get("labels", []):
            if not lbl.get("usable"):
                continue
            usable_labels.append(lbl)
            if not formal and lbl.get("new_formal_problem"):
                pool_row = pool.get(idx, {})
                formal = pool_row.get("formal_problem", "")
            if not original_block:
                original_block = lbl.get("original_local_block", "")

        per_row = {
            "index": idx,
            "task_family": row.get("task_family", "unknown"),
            "usable_labels": len(usable_labels),
            "function_exec_ok": bool(row.get("function_exec_ok")),
            "label_lean_results": [],
            "function_on_original_result": None,
            "function_on_training_results": [],
            "function_on_unseen_results": [],
        }

        for lbl in usable_labels:
            new_formal = lbl.get("new_formal_problem") or ""
            if not new_formal:
                per_row["label_lean_results"].append({
                    "new_answer": lbl["new_answer"], "passed": False,
                    "complete": False, "reason": "no reconstruction",
                })
                continue
            batch_codes.append(new_formal)
            batch_meta.append({
                "row_index": len(enriched_rows), "category": "label",
                "new_answer": lbl["new_answer"],
            })
            per_row["label_lean_results"].append({
                "new_answer": lbl["new_answer"], "passed": False, "complete": False, "reason": "",
            })

        if row.get("function_exec_ok") and row.get("synthesis_function") and original_block:
            fn, exec_err = exec_fill_answer(row["synthesis_function"])
            if fn is not None:
                pool_row = pool.get(idx, {})
                ranked = pool_row.get("ranked_answers", []) or []
                original_answer = row.get("original_answer", "") or (
                    str(ranked[0]) if ranked else ""
                )
                # Recover pool_formal
                pool_formal = ""
                for lbl in usable_labels:
                    cf = lbl.get("new_formal_problem") or ""
                    if cf and lbl.get("rewritten_local_block") and lbl.get("original_local_block"):
                        pool_formal = cf.replace(
                            lbl["rewritten_local_block"], lbl["original_local_block"], 1
                        )
                        if original_block in pool_formal:
                            break
                if not pool_formal or original_block not in pool_formal:
                    pool_formal = pool_row.get("formal_problem", "") or formal

                # Original answer
                if original_answer:
                    o_out, o_err = safe_call(fn, original_answer)
                    orig_res = {
                        "original_answer": original_answer,
                        "fn_output": o_out[:300] if o_out else "",
                        "passed": False, "complete": False, "reason": o_err,
                    }
                    per_row["function_on_original_result"] = orig_res
                    if o_out:
                        if o_out == original_block:
                            orig_res.update(passed=True, complete=True,
                                            reason="exact_match_original_block")
                        else:
                            new_f = pool_formal.replace(original_block, o_out, 1)
                            if new_f and new_f != pool_formal:
                                batch_codes.append(new_f)
                                batch_meta.append({
                                    "row_index": len(enriched_rows),
                                    "category": "fn_original",
                                    "new_answer": original_answer,
                                })

                def _check_answers(answers, bucket_key, category):
                    for ans in answers:
                        o_out, o_err = safe_call(fn, ans)
                        new_f = pool_formal.replace(original_block, o_out, 1) if o_out else ""
                        res = {"new_answer": ans, "fn_output": o_out[:300],
                               "passed": False, "complete": False, "reason": o_err}
                        per_row[bucket_key].append(res)
                        if o_out and new_f != pool_formal:
                            batch_codes.append(new_f)
                            batch_meta.append({
                                "row_index": len(enriched_rows),
                                "category": category, "new_answer": ans,
                            })

                _check_answers([lbl["new_answer"] for lbl in usable_labels],
                               "function_on_training_results", "fn_train")
                trained = {lbl["new_answer"] for lbl in usable_labels}
                unseen = [a for a in ranked[1:] if a not in trained][:num_unseen]
                _check_answers(unseen, "function_on_unseen_results", "fn_unseen")

        enriched_rows.append(per_row)

    print(f"  Lean-verifying {len(batch_codes)} reconstructed statements...")
    if batch_codes:
        lean_results = verify_lean(batch_codes)
        for meta, lr in zip(batch_meta, lean_results):
            target = enriched_rows[meta["row_index"]]
            if meta["category"] == "fn_original":
                entry = target.get("function_on_original_result")
                if entry is None or entry.get("passed"):
                    continue
                entry["passed"] = bool(lr.get("passed"))
                entry["complete"] = bool(lr.get("complete"))
                continue
            bucket = {
                "label": "label_lean_results",
                "fn_train": "function_on_training_results",
                "fn_unseen": "function_on_unseen_results",
            }[meta["category"]]
            for entry in target[bucket]:
                if entry["new_answer"] == meta["new_answer"] and not entry.get("passed"):
                    entry["passed"] = bool(lr.get("passed"))
                    entry["complete"] = bool(lr.get("complete"))
                    break

    write_hf_dataset(enriched_rows, lean_out)

    def _pass_count(rows_list, bucket):
        total = passed = 0
        for r in rows_list:
            for e in r.get(bucket, []):
                total += 1
                if e.get("passed"):
                    passed += 1
        return passed, total

    lp, lt = _pass_count(enriched_rows, "label_lean_results")
    tp, tt = _pass_count(enriched_rows, "function_on_training_results")
    up, ut = _pass_count(enriched_rows, "function_on_unseen_results")
    summary = {
        "num_rows": len(enriched_rows),
        "label_lean_pass": lp, "label_lean_total": lt,
        "function_on_training_pass": tp, "function_on_training_total": tt,
        "function_on_unseen_pass": up, "function_on_unseen_total": ut,
    }
    write_summary(summary, f"{out}/gold_lean_check_summary.json")
    print(f"  seed done. Output: {lean_out}")


# strict: Phase 2 gold — rejection sampling with strict filtering

def stage_strict(out: str, args: argparse.Namespace) -> None:
    """Phase 2 rejection sampling: prepare input → gen_data.py → extract gold."""
    print("\n=== strict: Phase 2 gold — rejection sampling ===")

    gold_multi = f"{out}_gold_structured_multi"
    gold_short = f"{out}_gold_short_label"
    pool_path = f"{out}_problem_with_answer1"

    # --- Step 1: prep_phase2_input ---
    print("  Step 1: Preparing phase 2 input...")
    pool = load_hf_dataset_indexed(pool_path)

    num_test_answers = 4
    num_candidates = 8
    input_rows: list[dict[str, Any]] = []
    for p in [gold_multi, gold_short]:
        p_rows = load_dataset(p, split="train")
        for row in p_rows:
            idx = int(row["index"])
            pool_row = pool.get(idx)
            if not pool_row:
                continue
            ranked = pool_row.get("ranked_answers") or []
            test_answers = [str(a) for a in ranked[1 : 1 + num_test_answers]]
            if not test_answers:
                continue
            original_local_block = ""
            for lbl in row.get("labels", []):
                if lbl.get("usable") and lbl.get("original_local_block"):
                    original_local_block = lbl["original_local_block"]
                    break
            if not original_local_block:
                continue
            base_row = {
                "index": idx,
                "informal_problem": pool_row.get("problem", ""),
                "formal_problem": row["formal_problem"],
                "original_answer": row["original_answer"],
                "original_local_block": original_local_block,
                "test_answers": test_answers,
                "task_family": row["task_family"],
            }
            for cand_id in range(num_candidates):
                input_rows.append({**base_row, "candidate_id": cand_id})

    n_problems = len(input_rows) // num_candidates if input_rows else 0
    print(f"  Prepared {len(input_rows)} rows ({n_problems} problems × {num_candidates} candidates)")

    output_repo = _run_rs_flow(
        input_rows=input_rows, out=out, prefix="strict_rs", args=args,
        temperature=0.8,
    )

    print("  Postprocessing...")
    gold = _postprocess_fill_answer_rs(
        output_repo, parse_fn=_parse_json_object,
        gold_criteria_fn=_strict_gold_criteria,
    )

    phase2_combined = f"{out}_phase2_gold_combined"
    write_hf_dataset(gold, phase2_combined)
    write_summary({"num_problems": n_problems, "num_gold": len(gold)},
                  f"{out}/phase2_summary.json")
    print(f"  strict output: {phase2_combined}")


# hard: Hard-case gold — mining + rejection sampling


def _locate_answer_expr(formal_problem: str, answer: str) -> tuple[str, str]:
    """Helper to locate where the answer uniquely occurs inside the formal problem."""
    loc_status = "none"
    original_expr = ""
    hits = []
    for expr, rule in translate_to_lean_candidates(answer):
        n = formal_problem.count(expr) if expr else 0
        if n > 0:
            hits.append((expr, rule, n))
    if hits:
        unique_hits = [(e, r) for e, r, n in hits if n == 1]
        if len(unique_hits) == 1:
            loc_status = "unique"
            original_expr = unique_hits[0][0]
        else:
            loc_status = "multi"
    return loc_status, original_expr


def _generate_translator_jobs(formal_problem: str, original_expr: str, test_answers: list[str]) -> list[str]:
    """Helper to generate alternative formal statements for the test answers."""
    translator_jobs = []
    for ta in test_answers:
        for new_expr, rule in translate_to_lean_candidates(ta):
            if not new_expr:
                continue
            new_formal = formal_problem.replace(original_expr, new_expr, 1)
            if new_formal != formal_problem:
                translator_jobs.append(new_formal)
    return translator_jobs


def stage_hard(out: str, args: argparse.Namespace) -> None:
    """Mine hard cases and rejection-sample gold fill-answer pairs (Bootstrap & vLLM)."""
    print("\n=== hard: Hard-case gold mining ===")

    # --- Step 1: mine hard cases (no inference) ---
    print("  Step 1: Mining hard cases...")

    pool_repo = f"{out}_formal_problem_pool"
    answers_jsonl = f"{out}_gemini_flash_answers_70k_min2"
    min_test_answers = 4
    max_test_answers = 4

    print(f"  Loading pool from: {pool_repo}")
    ranked_by_idx = _load_ranked_answers(answers_jsonl)
    print(f"  Ranked answers for {len(ranked_by_idx)} indices.")

    cls_counts: Counter[str] = Counter()
    pending_verify_codes: list[str] = []
    pending_verify_meta: list[tuple[int, int]] = []
    pending_rows: list[dict] = []

    for row in load_dataset(pool_repo, split="train"):
        if not row.get("passed", False):
            continue
        idx = int(row["index"])
        answer = str(row["answer"]).strip()
        formal_problem = row["formal_problem"]
        ranked = ranked_by_idx.get(idx, [])
        test_answers = [a for a in ranked if a != answer][:max_test_answers]
        if len(test_answers) < min_test_answers:
            cls_counts["skip_no_test_answers"] += 1
            continue
        if answer in formal_problem:
            cls_counts["direct_skip"] += 1
            continue

        loc_status, original_expr = _locate_answer_expr(formal_problem, answer)

        translator_jobs = []
        if loc_status == "unique" and original_expr:
            translator_jobs = _generate_translator_jobs(formal_problem, original_expr, test_answers)

        mined_row = {
            "index": idx,
            "problem": row["problem"],
            "answer": answer,
            "ranked_answers": ranked,
            "test_answers": test_answers,
            "formal_problem": formal_problem,
            "problem_type": row.get("problem_type", "unknown"),
            "translator_loc_status": loc_status,
            "translator_verified": False,
            "hard_reason": (
                "translator_not_located" if loc_status == "none" else
                "translator_multi_unverified" if loc_status == "multi" else
                "translator_located_unverified"
            ),
        }
        row_pos = len(pending_rows)
        pending_rows.append(mined_row)

        for code in translator_jobs:
            pending_verify_meta.append((row_pos, 0))
            pending_verify_codes.append(code)

    if pending_verify_codes:
        print(f"  Translator verification: {len(pending_verify_codes)} codes...")
        lean_results = verify_lean(pending_verify_codes)
        for (row_pos, _), lr in zip(pending_verify_meta, lean_results):
            if lr.get("passed"):
                pending_rows[row_pos]["translator_verified"] = True

    hard_cases_config = f"{out}_hard_cases"
    hard_rows_out = [r for r in pending_rows if not r["translator_verified"]]
    write_hf_dataset(hard_rows_out, hard_cases_config)
    print(f"  Hard cases: {len(hard_rows_out)}")

    # --- Step 2: run bootstrap rejection sampling ---
    print("  Step 2: Running bootstrap rejection sampling...")
    answers_by_idx = load_hf_dataset_indexed(answers_jsonl)
    gold_rows_bootstrap = _run_hard_case_rs(
        hard_rows=hard_rows_out, answers_by_idx=answers_by_idx,
        out=out, prefix="bootstrap_rs", args=args, rs_n=16, temperature=0.8,
    )
    bootstrap_gold = f"{out}_bootstrap_hard_gold"
    write_hf_dataset(gold_rows_bootstrap, bootstrap_gold)
    write_summary({
        "n_hard_cases": len(hard_rows_out),
        "n_gold": len(gold_rows_bootstrap),
    }, f"{out}/bootstrap_hard_gold_summary.json")
    print(f"  Bootstrap hard gold repo: {bootstrap_gold}")

    # --- Step 3: run vLLM rejection sampling ---
    print("  Step 3: Running vLLM rejection sampling...")
    gold_rows_vllm = _run_hard_case_rs(
        hard_rows=hard_rows_out, answers_by_idx=answers_by_idx,
        out=out, prefix="vllm_rs", args=args, rs_n=args.rs_n, temperature=args.rs_temp,
    )
    mining_gold = f"{out}_mining_hard_gold"
    write_hf_dataset(gold_rows_vllm, mining_gold)
    write_summary({
        "n_gold": len(gold_rows_vllm),
        "rs_n": args.rs_n, "rs_temp": args.rs_temp,
    }, f"{out}/mining_hard_gold_summary.json")
    print(f"  vLLM mining gold output: {mining_gold}")



# final: Aggregate all gold + build final SFT dataset


def stage_final(out: str, args: argparse.Namespace) -> None:
    """Aggregate bootstrap+phase2+mining gold, then build the HF SFT dataset."""
    print("\n=== final: Aggregate gold + build final SFT dataset ===")

    # --- Step 1: aggregate final gold ---
    print("  Step 1: Aggregating final gold...")
    bootstrap_hard_gold = f"{out}_bootstrap_hard_gold"
    phase2_gold = f"{out}_phase2_gold_combined"
    mining_gold = f"{out}_mining_hard_gold"

    seen_indices: set[int] = set()
    combined: list[dict] = []
    by_source: Counter = Counter()
    n_dup_skipped = 0

    def _collect_layer(path: str, source_tag: str, by_source_tag: str | None = None) -> tuple[int, int]:
        nonlocal n_dup_skipped
        rows = load_dataset(path, split="train")
        kept = 0
        for r in rows:
            idx = int(r["index"])
            if idx in seen_indices:
                n_dup_skipped += 1; continue
            row = dict(r)
            row["combined_source"] = source_tag
            combined.append(row)
            seen_indices.add(idx)
            by_source[by_source_tag or source_tag] += 1
            kept += 1
        return kept, len(rows)

    _collect_layer(bootstrap_hard_gold, "bootstrap_hard_gold")
    _collect_layer(f"{out}_bootstrap_hard_gold_phase2_extracted", "phase2_extracted")
    _collect_layer(mining_gold, "mining_vllm")

    final_gold_path = f"{out}_final_hard_gold"
    write_hf_dataset(combined, final_gold_path)
    write_summary({
        "n_total_unique_indices": len(combined),
        "n_dup_skipped": n_dup_skipped,
        "by_source": dict(by_source),
    }, f"{out}/final_hard_gold_summary.json")
    print(f"  Aggregated: {len(combined)} unique indices, {n_dup_skipped} dups skipped")

    # --- Step 2: build the final SFT dataset ---
    print("  Step 2: Building final SFT dataset...")
    sft_out = args.sft_output_dir or f"{out}_sft_fill_answer_final"
    _build_sft_dataset(
        out=out,
        hard_gold_path=final_gold_path,
        hard_source_prefix="final_hard",
        sft_output_dir=sft_out,
        phase2_gold_path=phase2_gold,
        **_sft_kwargs(args),
    )


#CLI


STAGE_FUNCS = {
    "package": stage_package,
    "pool": stage_pool,
    "seed": stage_seed,
    "strict": stage_strict,
    "hard": stage_hard,
    "final": stage_final,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "stages", nargs="*", choices=STAGES, default=[],
        help="Stage(s) to run (e.g. pool seed strict).",
    )
    p.add_argument(
        "--output-dir", default="outputs",
        help="Root output directory (default: outputs).",
    )

    seed_grp = p.add_argument_group("seed options")
    seed_grp.add_argument("--gemini-model", default=None,
                          help="Gemini model for synthesis (default: gemini-2.5-pro).")

    rs_grp = p.add_argument_group("strict/hard options (GPU stages)")
    rs_grp.add_argument("--base-model", default="Qwen/Qwen3-8B",
                        help="Base model for rejection sampling (e.g. Qwen/Qwen3-8B).")
    rs_grp.add_argument("--lora-path", default=None,
                        help="LoRA checkpoint for rejection sampling.")

    hard_grp = p.add_argument_group("hard options")
    hard_grp.add_argument("--rs-n", type=int, default=32,
                          help="Number of rejection samples per prompt (default: 32).")
    hard_grp.add_argument("--rs-temp", type=float, default=0.7,
                          help="Sampling temperature (default: 0.7).")

    sft_grp = p.add_argument_group("final/package options (SFT dataset build)")
    sft_grp.add_argument("--sft-output-dir", default=None,
                         help="Output dir for SFT dataset (default: <output-dir>/sft_fill_answer_final).")
    sft_grp.add_argument("--test-size", type=float, default=0.1,
                         help="Fraction of data for test split (default: 0.1).")
    sft_grp.add_argument("--seed", type=int, default=42, help="Random seed.")
    sft_grp.add_argument("--require-unseen-pass", action="store_true",
                         help="Require unseen answers to pass Lean for phase 1 gold.")
    sft_grp.add_argument("--upsample-short-label", type=int, default=2,
                         help="Upsample factor for short_label examples (default: 2).")
    sft_grp.add_argument("--upsample-hard", type=int, default=1,
                         help="Upsample factor for hard gold examples (default: 1).")
    sft_grp.add_argument("--phase2-min-pass", type=int, default=4,
                         help="Min Lean passes for phase 2 gold (default: 4).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    for stage in args.stages:
        STAGE_FUNCS[stage](args.output_dir, args)


if __name__ == "__main__":
    main()
