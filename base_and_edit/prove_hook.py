"""Prove hook for gen_data.py: preprocess/postprocess for prover runs.

Preprocess: renames `new_formal` → `formal_statement` so that gen_data.py's
generation code can use it as the input key for proof generation.

Postprocess: adds `single_tactic_complete` / `single_tactic_passed` aliases
for downstream compatibility with eval_prover_answer_selection.py.
"""

from datasets import Dataset
from typing import Any


def preprocess(ds: Dataset) -> Dataset:
    """Rename new_formal → formal_statement for the prover prompt."""
    def _rename(example):
        example["formal_statement"] = example.get("new_formal") or ""
        return example
    return ds.map(_rename)


def postprocess(ds: Dataset, ds_input: Dataset = None) -> list[dict[str, Any]]:
    """Add single_tactic_* aliases expected by downstream scripts."""
    results = []
    for row in ds:
        row = dict(row)
        row["single_tactic_passed"] = bool(row.get("passed"))
        row["single_tactic_complete"] = bool(row.get("complete"))
        results.append(row)
    return results


def compute_summary(results: list[dict[str, Any]]) -> dict:
    n = len(results)
    n_passed = sum(1 for r in results if r.get("complete"))
    return {
        "n_rows": n,
        "n_complete": n_passed,
        "complete_rate": n_passed / n if n else 0.0,
    }
