"""Generate data by calling GPT in parallel."""

import argparse
import json
import tempfile
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.exceptions import DatasetNotFoundError, DatasetGenerationError
from utils import save_results_to_dataset_repo, load_results_from_dataset_repo, clear_repo, push_dataset_to_hub, resolve_hf_repo
from huggingface_hub import HfApi
from pilot_study_2025.answer_corruption import add_corrupted_answer_columns
from verify import verify, add_lean_results
from generation import DataThread, resolve_model_config
from base_and_edit.generate_answers import compute_summary as compute_ranked_summary
from base_and_edit.run_eval_brute_force import preprocess as af_brute_force_preprocess, postprocess as af_brute_force_postprocess, compute_summary as af_brute_force_summary
from base_and_edit.run_eval_ours_R2 import preprocess as r2_preprocess, postprocess as r2_postprocess, compute_summary as r2_summary
from base_and_edit.prove_hook import preprocess as prove_preprocess, postprocess as prove_postprocess, compute_summary as prove_summary
from base_and_edit.build_fill_answer_data import fill_answer_strict_postprocess, fill_answer_strict_summary, fill_answer_mining_postprocess, fill_answer_mining_summary


def load_input_dataset(input_source, split):
    if isinstance(input_source, str):
        source_path = Path(input_source)
        if source_path.suffix == ".jsonl" or source_path.suffix == ".json":
            return load_dataset("json", data_files={split: str(source_path)}, split=split)
        return load_dataset(resolve_hf_repo(input_source), split=split)
    return input_source


def maybe_push_dataset(args, split, dataset, commit_message):
    if getattr(args, "no_push_to_hub", False):
        print(f"Skipping HF push for {args.output} [{split}]: {commit_message}", flush=True)
        return
    push_dataset_to_hub(args.output, split, dataset, commit_message)


def maybe_save_results(args, stats):
    if getattr(args, "no_push_to_hub", False):
        print(f"Skipping HF results save for {args.output}", flush=True)
        return
    save_results_to_dataset_repo(stats, args.output, "results.jsonl")


def write_dataset_jsonl(dataset, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    print(f"Wrote local JSONL -> {path} ({len(dataset)} rows)", flush=True)


def _save_summary(args, summary):
    summary_filename = "summary.json"
    if not getattr(args, "no_push_to_hub", False):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / summary_filename
            summary_path.write_text(json.dumps(summary, indent=2))
            HfApi().upload_file(
                path_or_fileobj=str(summary_path),
                path_in_repo=summary_filename,
                repo_id=args.output,
                repo_type="dataset",
                commit_message=f"Add {summary_filename}",
            )
        print(f"Uploaded {summary_filename} to {args.output}")
    if args.local_output_jsonl:
        local_summary = Path(args.local_output_jsonl).with_suffix(".summary.json")
        local_summary.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary -> {local_summary}")
    print(f"Summary: {summary}")


def write_stats_json(stats, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote local stats JSON -> {path}", flush=True)


def load_existing_progress(output_repo, split, num_examples, verify, status_key):
    """
    Load existing progress from output repository.

    Returns:
        tuple: (all_samples_per_example, ds_all_output, start_sample_idx, need_verification_for_last_sample)
    """
    try:
        ds_existing = load_dataset(output_repo)
        if split not in ds_existing:
            return [[] for _ in range(num_examples)], None, 0, False

        ds_existing_split = ds_existing[split]
        if len(ds_existing_split) == 0 or "samples" not in ds_existing_split.column_names:
            return [[] for _ in range(num_examples)], None, 0, False

        num_processed_samples = len(ds_existing_split[0].get("samples", []))
        if num_processed_samples == 0:
            return [[] for _ in range(num_examples)], None, 0, False

        all_samples_per_example = ds_existing_split["samples"]
        ds_all_output = ds_existing_split.remove_columns(["samples"])

        if verify and status_key in ds_all_output.column_names:
            has_null_status = any(example.get(status_key) is None for example in ds_all_output)
            if has_null_status:
                print(f"Resuming from sample {num_processed_samples} (generation done, verification needed)")
                return all_samples_per_example, ds_all_output, num_processed_samples - 1, True
            else:
                print(f"Resuming from sample {num_processed_samples + 1} (found {num_processed_samples} fully processed samples)")
                return all_samples_per_example, ds_all_output, num_processed_samples, False
        else:
            print(f"Resuming from sample {num_processed_samples + 1} (found {num_processed_samples} already processed samples)")
            return all_samples_per_example, ds_all_output, num_processed_samples, False

    except (DatasetNotFoundError, DatasetGenerationError):
        print(f"No existing dataset found at {output_repo}, starting from scratch")
        return [[] for _ in range(num_examples)], None, 0, False


def generate_sample(ds, args, sample_idx, num_examples, push_interval, split, data_threads):
    """Generate a single sample for all problems in the dataset."""
    print(f"\nGenerating sample {sample_idx + 1}/{args.num_samples}")
    current_offset = 0
    ds_sample_output = None

    while current_offset < num_examples:
        batch_examples = min(push_interval, num_examples - current_offset)

        with tqdm(total=batch_examples, desc=f"Sample {sample_idx + 1}/{args.num_samples}") as progress_bar:
            num_examples_each = (batch_examples + args.njobs - 1) // args.njobs
            threads_to_use = []
            for i in range(args.njobs):
                start = num_examples_each * i
                if start >= batch_examples:
                    break
                end = min(num_examples_each * (i + 1), batch_examples)
                start += current_offset
                end += current_offset

                thread = data_threads[i]
                thread.ds = ds.select(range(start, end))
                thread.ds_output = None
                thread.progress_bar = progress_bar
                threads_to_use.append(thread)

            if len(threads_to_use) == 1:
                threads_to_use[0].run()
            else:
                for thread in threads_to_use:
                    thread.start()
                for thread in threads_to_use:
                    thread.join()
            ds_batch_output = concatenate_datasets([thread.ds_output for thread in threads_to_use])

        if ds_sample_output is None:
            ds_sample_output = ds_batch_output
        else:
            ds_sample_output = concatenate_datasets([ds_sample_output, ds_batch_output])

        current_offset += batch_examples

    return ds_sample_output


def process_one_sample(
    sample_idx, ds, ds_all_output, all_samples_per_example, num_examples,
    need_verification_for_last_sample, start_sample_idx, status_key, args,
    push_interval, split, data_threads, sample_stats
):
    """Process one sample: generation, verification, and pushing to hub."""
    if need_verification_for_last_sample and sample_idx == start_sample_idx:
        print(f"Skipping generation for sample {sample_idx + 1}, proceeding directly to verification")
        failed_indexes = [i for i in range(num_examples)
                          if ds_all_output[i].get(status_key) is None]
        failed_idx_map = {orig_idx: i for i, orig_idx in enumerate(failed_indexes)}
        ds_sample_output = ds_all_output.select(failed_indexes)
    else:
        if ds_all_output is None:
            ds_to_generate = ds
            failed_indexes = None
        else:
            failed_indexes = [i for i in range(num_examples)
                              if not ds_all_output[i].get(status_key, False)]

            ds_to_generate = ds.select(failed_indexes)
            print(f"Re-generating {len(failed_indexes)} failed examples")

        ds_sample_output = generate_sample(
            ds_to_generate, args, sample_idx,
            len(ds_to_generate), push_interval, split, data_threads
        )

        if ds_all_output is None:
            for i, example in enumerate(ds_sample_output):
                all_samples_per_example[i].append(example["_output_raw"])
        else:
            failed_idx_map = {orig_idx: i for i, orig_idx in enumerate(failed_indexes)}
            for i in range(num_examples):
                if i in failed_idx_map:
                    new_example = ds_sample_output[failed_idx_map[i]]
                    all_samples_per_example[i].append(new_example["_output_raw"])
                else:
                    all_samples_per_example[i].append(None)

    if args.verify:
        if not (need_verification_for_last_sample and sample_idx == start_sample_idx):
            ds_sample_output = ds_sample_output.map(lambda example: {**example, status_key: None})

            if ds_all_output is None:
                ds_all_output = ds_sample_output
            else:
                ds_all_output = ds_all_output.map(
                    lambda example, idx: ds_sample_output[failed_idx_map[idx]]
                                         if idx in failed_idx_map else example,
                    with_indices=True
                )

            ds_current = ds_all_output.map(
                lambda _, idx: {"samples": all_samples_per_example[idx]}, with_indices=True
            )

            commit_message = f"Update {split} split after sample {sample_idx + 1}/{args.num_samples} (before verification)"
            maybe_push_dataset(args, split, ds_current, commit_message)
            if not getattr(args, "no_push_to_hub", False):
                print(f"Pushed unverified sample {sample_idx + 1} to {args.output}")

        ret_lean = verify(ds_sample_output, output_key=args.output_key)
        ds_sample_output = add_lean_results(ds_sample_output, ret_lean)
        num_passed = sum(1 for example in ds_sample_output if example.get(status_key, False))
        num_total = len(ds_sample_output)

    if failed_indexes is None:
        ds_all_output = ds_sample_output
    else:
        ds_all_output = ds_all_output.map(
            lambda example, idx: ds_sample_output[failed_idx_map[idx]]
            if idx in failed_idx_map and ds_sample_output[failed_idx_map[idx]].get(status_key, False) else example,
            with_indices=True
        )


    if args.verify:
        overall_passed = sum(1 for example in ds_all_output if example.get(status_key, False))
        overall_total = len(ds_all_output)
        overall_pass_rate = overall_passed / overall_total if overall_total > 0 else 0.0

        sample_stats.append({
            "sample_idx": sample_idx,
            "num_passed": num_passed,
            "num_total": num_total,
            "pass_rate": num_passed / num_total if num_total > 0 else 0.0,
            "overall_pass_rate": overall_pass_rate
        })
        print(f"Sample {sample_idx + 1}: {num_passed}/{num_total} passed ({100 * num_passed / num_total:.2f}%), overall: {overall_passed}/{overall_total} ({100 * overall_pass_rate:.2f}%)")

        ds_current = ds_all_output.map(
            lambda _, idx: {"samples": all_samples_per_example[idx]},
            with_indices=True
        )

        commit_message = f"Update {split} split after sample {sample_idx + 1}/{args.num_samples}: {num_passed}/{num_total} passed"
        maybe_push_dataset(args, split, ds_current, commit_message)

        current_stats = {split: sample_stats}
        maybe_save_results(args, current_stats)
        if not getattr(args, "no_push_to_hub", False):
            print(f"Pushed sample {sample_idx + 1} results to {args.output}")
    else:
        ds_current = ds_all_output
        commit_message = f"Update {split} split after sample {sample_idx + 1}/{args.num_samples}"
        maybe_push_dataset(args, split, ds_current, commit_message)

    return ds_all_output


def generate(input, output, split, args=None):
    ds = load_input_dataset(input, split)
    num_examples = len(ds) if not args.num_examples else min(args.num_examples, len(ds))
    ds = ds.select(range(num_examples))

    if args.hook == "af_brute_force":
        ds = af_brute_force_preprocess(ds)
        num_examples = len(ds)
        print(f"After preprocess (af_brute_force): {num_examples} rows")
    elif args.hook == "r2":
        ds = r2_preprocess(ds)
        num_examples = len(ds)
        print(f"After preprocess (r2): {num_examples} rows needing R2")
    elif args.hook == "prove":
        ds = prove_preprocess(ds)
        num_examples = len(ds)
        print(f"After preprocess (prove): {num_examples} rows")

    if args.retry:
        ds_last = load_dataset(args.retry)
        if split == "test":
            split_last = "test"
        else:
            assert split in ["train", "train_base"]
            # train_base is used in iterations other than the first one
            split_last = "train_base" if split == "train_base" and "train_base" in ds_last else "train"
        ds_last = ds_last[split_last]
        retry_indexes = [i for i in range(min(num_examples, len(ds_last)))
                         if not ds_last[i].get("passed", False)]
        ds = ds.select(retry_indexes)
        num_examples = len(ds)

    resolved_model = resolve_model_config(args.model)["model"]
    if "/" in resolved_model:
        args.njobs = 1

    if getattr(args, "corrupt_answer", False) and args.output_key == "formal_problem":
        if "answer" not in ds.column_names:
            print("WARNING: --corrupt_answer is set but dataset has no 'answer' column; skipping corruption.")
        else:
            ds = ds.map(add_corrupted_answer_columns)

    push_interval = args.push_interval or num_examples

    print(f"Generating {num_examples} examples with push_interval {push_interval}")

    ds_all_output = None
    start_sample_idx = 0

    if not args.verify:
        assert args.num_samples == 1

    if args.verify:
        if args.output_key == "formal_proof":
            status_key = "complete"
        elif args.output_key == "formal_problem":
            status_key = "passed"
        else:
            raise NotImplementedError(f"Unsupported output_key for verification: {args.output_key}")
    else:
        status_key = "passed"

    if args.resume:
        (all_samples_per_example, ds_all_output,
         start_sample_idx, need_verification_for_last_sample) = load_existing_progress(
             args.output, split, num_examples, args.verify, status_key)
    else:
        all_samples_per_example = [[] for _ in range(num_examples)]
        need_verification_for_last_sample = False

    sample_stats = []
    if args.resume:
        sample_stats = load_results_from_dataset_repo(args.output, split)

    if start_sample_idx < args.num_samples:
        data_threads = []
        for i in range(args.njobs):
            thread = DataThread(None, output, args.input_key, args.output_key,
                                args.model, i, None, args.batch_size)
            data_threads.append(thread)

        for sample_idx in range(start_sample_idx, args.num_samples):
            ds_all_output = process_one_sample(
                sample_idx, ds, ds_all_output, all_samples_per_example, num_examples,
                need_verification_for_last_sample, start_sample_idx, status_key, args,
                push_interval, split, data_threads, sample_stats
            )

        if args.verify:
            print("\n" + "="*60)
            print("SAMPLE STATISTICS SUMMARY")
            print("="*60)
            for stat in sample_stats:
                print(f"Sample {stat['sample_idx'] + 1}: {stat['num_passed']}/{stat['num_total']} passed ({100 * stat['pass_rate']:.2f}%)")
            print("="*60 + "\n")

    if args.retry:
        retry_idx_to_output_idx = {retry_idx: i for i, retry_idx in enumerate(retry_indexes)}
        # The initial dataset may not have the "messages" column (unnecessary),
        # which may cause issues when merging the datasets.
        if "messages" in ds_last:
            ds_last = ds_last.remove_columns("messages")
        if "messages" in ds_all_output:
            ds_all_output = ds_all_output.remove_columns("messages")
        def update_example(example, idx):
            if idx in retry_idx_to_output_idx:
                return ds_all_output[retry_idx_to_output_idx[idx]]
            return example
        ds_all_output = ds_last.map(update_example, with_indices=True)

    return ds_all_output, sample_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--input_key")
    parser.add_argument("--output_key")
    parser.add_argument("--retry", type=str, help="Retry a previous generation and only handle failed cases.")
    parser.add_argument("--resume", action="store_true", help="Resume from the last processed sample in the output dataset.")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--njobs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=20480, help="Batch size for vLLM models")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--push_interval", type=int, default=None, help="Push to HuggingFace after every X examples")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no_push_to_hub", action="store_true", help="Do not clear or push HF output repos; use with --local_output_jsonl.")
    parser.add_argument("--local_output_jsonl", default=None, help="Write final output dataset to a local JSONL file.")
    parser.add_argument("--local_stats_json", default=None, help="Write sample statistics to a local JSON file.")
    parser.add_argument(
        "--corrupt_answer",
        action="store_true",
        help="Add `answer_new`/`if_change` and use wrong answers for generation (only for output_key=formal_problem).",
    )
    parser.add_argument(
        "--hook",
        default=None,
        choices=["af_brute_force", "r2", "prove", "fill_answer_strict", "fill_answer_mining"],
        help="Pre/post-process hook for the dataset.",
    )
    args = parser.parse_args()

    def _ensure_hf_namespace(name):
        if "/" not in name and not Path(name).suffix:
            username = HfApi().whoami()["name"]
            return f"{username}/{name}"
        return name

    args.input = _ensure_hf_namespace(args.input)
    if not args.no_push_to_hub:
        args.output = _ensure_hf_namespace(args.output)
    print(f"Input: {args.input}, Output: {args.output}")

    if ("/" not in args.model and ":" not in args.model) and not args.push_interval:
        args.push_interval = 50 * args.njobs
    if args.output_key in ["formal_proof", "formal_problem"]:
        args.verify = True
    if not args.resume and not args.no_push_to_hub:
        clear_repo(args.output, args.splits)
    elif args.no_push_to_hub:
        print(f"Skipping HF clear for {args.output}", flush=True)

    for split in args.splits:
        ds_output, stats = generate(args.input, args.output, split, args=args)
        if args.local_stats_json:
            stats_path = args.local_stats_json.format(split=split) if "{split}" in args.local_stats_json else args.local_stats_json
            write_stats_json({split: stats}, stats_path)
        if args.output_key == "ranked_answers":
            summary = compute_ranked_summary(ds_output)
            _save_summary(args, summary)

        if args.hook == "af_brute_force":
            results = af_brute_force_postprocess(ds_output)
            summary = af_brute_force_summary(results)
            ds_output = Dataset.from_list(results)
            maybe_push_dataset(args, split, ds_output, f"Post-processed {split} (af_brute_force)")
            _save_summary(args, summary)

        elif args.hook == "r2":
            ds_r1 = load_input_dataset(args.input, split)
            results = r2_postprocess(ds_output, ds_r1)
            summary = r2_summary(results)
            ds_output = Dataset.from_list(results)
            maybe_push_dataset(args, split, ds_output, f"Post-processed {split} (r2)")
            _save_summary(args, summary)

        elif args.hook == "prove":
            results = prove_postprocess(ds_output)
            summary = prove_summary(results)
            ds_output = Dataset.from_list(results)
            maybe_push_dataset(args, split, ds_output, f"Post-processed {split} (prove)")
            _save_summary(args, summary)

        elif args.hook == "fill_answer_strict":
            results = fill_answer_strict_postprocess(ds_output)
            summary = fill_answer_strict_summary(results)
            ds_output = Dataset.from_list(results)
            maybe_push_dataset(args, split, ds_output, f"Post-processed {split} (fill_answer_strict)")
            _save_summary(args, summary)

        elif args.hook == "fill_answer_mining":
            results = fill_answer_mining_postprocess(ds_output)
            summary = fill_answer_mining_summary(results)
            ds_output = Dataset.from_list(results)
            maybe_push_dataset(args, split, ds_output, f"Post-processed {split} (fill_answer_mining)")
            _save_summary(args, summary)

        if args.local_output_jsonl:
            out_path = args.local_output_jsonl.format(split=split) if "{split}" in args.local_output_jsonl else args.local_output_jsonl
            write_dataset_jsonl(ds_output, out_path)
