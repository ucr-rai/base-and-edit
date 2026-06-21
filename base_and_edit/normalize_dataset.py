#!/usr/bin/env python3
"""Normalize competition datasets to {index, problem, answer} schema and push to HF.

Produces a clean HF dataset that downstream steps consume with
``--input_key problem`` and the standard ``answer`` column.

Usage:
    python -m base_and_edit.normalize_dataset                # normalize and push all datasets
    python -m base_and_edit.normalize_dataset --dataset amc83  # normalize and push one dataset
"""

import argparse

from datasets import Dataset, DatasetDict, load_dataset


def normalize_aime2024() -> list[dict]:
    rows = []
    d = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    for i, r in enumerate(d):
        rows.append({
            "index": i,
            "problem": r["Problem"],
            "answer": str(r["Answer"]).strip(),
            "source_id": r.get("ID", ""),
        })
    return rows


def normalize_aime2025() -> list[dict]:
    rows = []
    base_idx = 0
    for cfg in ["AIME2025-I", "AIME2025-II"]:
        d = load_dataset("opencompass/AIME2025", cfg, split="test")
        for i, r in enumerate(d):
            rows.append({
                "index": base_idx + i,
                "problem": r["question"],
                "answer": str(r["answer"]).strip(),
                "source_id": f"{cfg}-{i}",
            })
        base_idx += len(d)
    return rows


def normalize_amc83() -> list[dict]:
    rows = []
    d = load_dataset("AI-MO/aimo-validation-amc", split="train")
    for i, r in enumerate(d):
        ans = r["answer"]
        if isinstance(ans, float) and ans.is_integer():
            ans_str = str(int(ans))
        else:
            ans_str = str(ans).strip()
        rows.append({
            "index": i,
            "problem": r["problem"],
            "answer": ans_str,
            "source_id": str(r.get("id", i)),
        })
    return rows


def normalize_hmmt2025feb() -> list[dict]:
    rows = []
    d = load_dataset("MathArena/hmmt_feb_2025", split="train")
    for i, r in enumerate(d):
        rows.append({
            "index": i,
            "problem": r["problem"],
            "answer": str(r["answer"]).strip(),
            "source_id": str(r.get("problem_idx", i)),
        })
    return rows


def normalize_olympiadbench_math_en() -> list[dict]:
    rows = []
    d = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    keep_idx = 0
    for r in d:
        if r["is_multiple_answer"]:
            continue
        if r["answer_type"] not in ("Numerical", "Expression"):
            continue
        ans_list = r["final_answer"] or []
        if not ans_list:
            continue
        ans = str(ans_list[0]).strip()
        if not ans:
            continue
        rows.append({
            "index": keep_idx,
            "problem": r["question"],
            "answer": ans,
            "source_id": str(r.get("id", keep_idx)),
        })
        keep_idx += 1
    return rows


def normalize_math500() -> list[dict]:
    rows = []
    d = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i, r in enumerate(d):
        rows.append({
            "index": i,
            "problem": r["problem"],
            "answer": str(r["answer"]).strip(),
            "source_id": r.get("unique_id", ""),
        })
    return rows


REGISTRY = {
    "math500": (normalize_math500, "math500_normalized"),
    "aime2024": (normalize_aime2024, "aime2024_normalized"),
    "aime2025": (normalize_aime2025, "aime2025_normalized"),
    "amc83": (normalize_amc83, "amc83_normalized"),
    "hmmt2025feb": (normalize_hmmt2025feb, "hmmt2025feb_normalized"),
    "olympiadbench_math_en": (normalize_olympiadbench_math_en, "olympiadbench_math_en_normalized"),
}


def run_one(name: str) -> None:
    fn, repo_name = REGISTRY[name]
    rows = fn()
    print(f"  {name}: {len(rows)} rows → {repo_name}")
    ds = Dataset.from_list(rows)
    DatasetDict({"test": ds}).push_to_hub(repo_name, private=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(REGISTRY.keys()),
                    help="Normalize one dataset (default: all).")
    args = ap.parse_args()

    targets = [args.dataset] if args.dataset else list(REGISTRY.keys())
    print(f"Normalizing {len(targets)} dataset(s):")
    for name in targets:
        run_one(name)
    print("Done.")


if __name__ == "__main__":
    main()
