#!/usr/bin/env python3
"""Pre/post-processing for ranked answer generation on NuminaMath.

Provides preprocess() and postprocess() functions that plug into gen_data.py's
DataThread infrastructure. Can also be run standalone for quick tests.
"""

import os
import re
import sys
from typing import Any

from datasets import Dataset, load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

KEEP_COLUMNS = ["problem", "solution", "answer", "problem_type"]


# ---------------------------------------------------------------------------
# Dataset cleaning (NuminaMath-specific)
# ---------------------------------------------------------------------------

def _is_yes(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() == "yes"


def _should_keep_numina_row(example: dict[str, Any]) -> bool:
    answer = str(example.get("answer", "")).strip().lower()
    return (
        answer != "proof"
        and _is_yes(example.get("problem_is_valid"))
        and _is_yes(example.get("solution_is_valid"))
    )


def clean_numina_dataset(ds: Dataset, num_proc: int = 1) -> Dataset:
    columns = set(ds.column_names)
    filter_columns = {"answer", "problem_is_valid", "solution_is_valid"}
    if filter_columns.issubset(columns):
        kwargs = {}
        if num_proc and num_proc > 1:
            kwargs["num_proc"] = num_proc
        ds = ds.filter(
            _should_keep_numina_row,
            desc="Filtering valid non-proof rows",
            **kwargs,
        )
        remove_columns = [c for c in ds.column_names if c not in KEEP_COLUMNS]
        if remove_columns:
            ds = ds.remove_columns(remove_columns)
        return ds

    if set(KEEP_COLUMNS).issubset(columns):
        return ds

    if {"problem", "answer"}.issubset(columns):
        return ds

    raise KeyError(
        "Dataset does not match raw NuminaMath columns or the cleaned schema. "
        f"Columns: {ds.column_names}"
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_ranked_prompt(prompt_template: str, problem: str, num_answers: int) -> str:
    prompt = prompt_template.format(problem=problem)
    prompt = re.sub(r"Give\s+5\s+possible final answers", f"Give {num_answers} possible final answers", prompt)
    prompt = re.sub(r"Exactly\s+5\s+lines\.", f"Exactly {num_answers} lines.", prompt)
    return prompt


# ---------------------------------------------------------------------------
# Answer parsing & evaluation
# ---------------------------------------------------------------------------

def normalize_answer(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("\\!", "")
    value = value.replace("$", "")
    value = value.replace("\\,", "")
    value = re.sub(r"\s+", "", value)
    return value


def parse_ranked_answers(raw: str, expected_count: int) -> list[str]:
    text = raw.strip()

    tag_match = re.search(r"<ranked_answers>(.*?)</ranked_answers>", text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1)
    else:
        marker = "### RANKED ANSWERS"
        marker_pos = text.rfind(marker)
        if marker_pos != -1:
            text = text[marker_pos + len(marker):]

    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("text\n"):
            text = text[5:]
        elif text.lower().startswith("markdown\n"):
            text = text[9:]

    answers = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\d+\s*[\)\.\:\-]\s*", "", cleaned)
        cleaned = re.sub(r"^\-\s*", "", cleaned)
        if cleaned:
            answers.append(cleaned)

    return answers[:expected_count]


# ---------------------------------------------------------------------------
# preprocess / postprocess — the main API for gen_data.py integration
# ---------------------------------------------------------------------------

def preprocess(ds: Dataset, prompt_path: str, num_answers: int, num_proc: int = 1) -> Dataset:
    """Clean dataset and add a ``_prompt`` column with ranked-answer prompts.

    The returned dataset is ready to be passed to gen_data.py's generation
    pipeline (DataThread reads ``_prompt`` or uses ``input_key``).
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    ds = clean_numina_dataset(ds, num_proc=num_proc)

    def add_prompt(example):
        example["_prompt"] = _build_ranked_prompt(prompt_template, example["problem"], num_answers)
        return example

    return ds.map(add_prompt)


def postprocess(example: dict, raw_output: str, num_answers: int) -> dict:
    """Parse model output into ranked answers and evaluate against gold.

    Adds fields: ranked_answers, parser_ok, exact_count_ok,
    top1_exact_match, any_exact_match.
    """
    ranked_answers = parse_ranked_answers(raw_output, expected_count=num_answers)
    parser_ok = len(ranked_answers) > 0
    exact_count_ok = len(ranked_answers) == num_answers

    top1_exact_match = None
    any_exact_match = None
    if parser_ok and "answer" in example:
        normalized_gold = normalize_answer(example["answer"])
        top1_exact_match = normalize_answer(ranked_answers[0]) == normalized_gold
        any_exact_match = any(
            normalize_answer(c) == normalized_gold for c in ranked_answers
        )

    example["ranked_answers"] = ranked_answers
    example["parser_ok"] = parser_ok
    example["exact_count_ok"] = exact_count_ok
    example["top1_exact_match"] = top1_exact_match
    example["any_exact_match"] = any_exact_match
    return example


def compute_summary(ds: Dataset) -> dict:
    n = len(ds)
    if n == 0:
        return {}
    summary = {
        "num_examples": n,
        "parser_ok": sum(1 for x in ds if x.get("parser_ok")),
        "exact_count_ok": sum(1 for x in ds if x.get("exact_count_ok")),
    }
    top1_values = [x["top1_exact_match"] for x in ds if x.get("top1_exact_match") is not None]
    any_values = [x["any_exact_match"] for x in ds if x.get("any_exact_match") is not None]
    if top1_values:
        summary["top1_exact_match"] = sum(top1_values)
        summary["top1_accuracy"] = sum(top1_values) / len(top1_values)
    if any_values:
        summary["any_exact_match"] = sum(any_values)
        summary["any_accuracy"] = sum(any_values) / len(any_values)
    return summary
