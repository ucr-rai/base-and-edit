#!/usr/bin/env python3
"""Build problem_with_answer1 for Numina using ranked generated answers."""

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from datasets import Dataset
from huggingface_hub import HfApi


DEFAULT_DATASET = "data/base_and_edit/clean_train"
DEFAULT_SPLIT = "train"
DEFAULT_ANSWERS_JSONL = "outputsgemini_answers.jsonl"
DEFAULT_OUTPUT_JSONL = "outputsproblem_with_answer1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append ranked answer 1 to each Numina problem."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--answers-jsonl", default=DEFAULT_ANSWERS_JSONL)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--hub-split", default="train")
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def load_source_dataset(dataset_name: str, split: str):
    if Path(dataset_name).exists():
        dataset_obj = load_from_disk(dataset_name)
        if isinstance(dataset_obj, DatasetDict):
            if split not in dataset_obj:
                raise KeyError(
                    f"Split '{split}' not found in dataset saved at {dataset_name}. "
                    f"Available splits: {list(dataset_obj.keys())}"
                )
            return dataset_obj[split]
        return dataset_obj
    return load_dataset(dataset_name, split=split)


def resolve_hub_repo(explicit_repo: str | None, suffix: str) -> str:
    if explicit_repo:
        return explicit_repo
    username = HfApi().whoami()["name"]
    return f"{username}/{suffix}"


def main() -> None:
    args = parse_args()

    ds = load_source_dataset(args.dataset, args.split)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.answers_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            answer_record = json.loads(line)
            idx = int(answer_record["index"])
            ranked_answers = answer_record.get("ranked_answers") or []
            ranked_answers = [str(x).strip() for x in ranked_answers if str(x).strip()]
            if not ranked_answers:
                continue
            answer1 = ranked_answers[0]

            ex = ds[idx]
            rows.append(
                {
                    "index": idx,
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                    "answer": ex["answer"],
                    "problem_type": ex["problem_type"],
                    "ranked_answers": ranked_answers,
                    "generated_answer": answer1,
                    "problem_with_answer1": ex["problem"].strip() + "\n\nAnswer: " + answer1,
                }
            )

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output_jsonl}")

    if args.push_to_hub and rows:
        repo_id = resolve_hub_repo(args.hub_repo, "base_and_edit_problem_with_answer1")
        Dataset.from_list(rows).push_to_hub(
            repo_id,
            split=args.hub_split,
            private=True,
            token=args.hf_token,
        )
        print(f"Pushed problem_with_answer1 dataset to {repo_id} [{args.hub_split}]")


if __name__ == "__main__":
    main()
