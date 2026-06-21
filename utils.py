import re
import json
import tempfile
import os
from typing import List
import torch
from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError, DatasetGenerationError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.models.utils import ChatMlSpecialTokens
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError


def resolve_hf_repo(name: str) -> str:
    """Prefix with the current HF user if no namespace is given."""
    if "/" not in name:
        name = f"{HfApi().whoami()['name']}/{name}"
    return name
from tenacity import retry, stop_after_attempt, wait_exponential


api_price = {
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1": (2., 8.),
    "gpt-5": (1.25, 10.),
    "gpt-5-mini": (0.25, 2.),
}


def parse_informal_steps(output: str) -> List[str]:
    """
    Extract all content within <step> tags from LLM output.

    Args:
        output (str): The raw text output from an LLM containing <step> tags

    Returns:
        List[str]: A list of strings, where each string is the content of a <step> tag
    """
    # Use regex to find all matches of content between <step> and </step> tags
    step_pattern = r"<step>(.*?)</step>"

    # re.DOTALL flag makes the dot (.) match newlines as well
    steps = re.findall(step_pattern, output, re.DOTALL)

    # Strip whitespace from each extracted step
    steps = [step.strip() for step in steps]

    return steps


def remove_lean_comments(code, verbose=False):
    lines = code.split('\n')

    lines_ = []
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        if line_stripped.startswith('--'):
            if verbose:
                print('Removing line:', line)
            i += 1
            continue
        if line_stripped.startswith('/-') and line_stripped.endswith('-/'):
            if verbose:
                print('Removing line:', line)
            i += 1
            continue
        if line_stripped.startswith('/-'):
            matched = False
            for j in range(i + 1, len(lines)):
                if lines[j].strip().endswith('-/'):
                    if verbose:
                        print('Removing lines:')
                        print(lines[i:j+1])
                    i = j + 1
                    matched = True
                    break
            if matched:
                continue
        lines_.append(line)
        i += 1

    return '\n'.join(lines_)


def clean_up_lean(code):
    code = code.replace("```lean", "")
    code = code.replace("```", "")
    code = code.replace("<think>", "")
    code = code.replace("</think>", "")
    code = code.replace("<header>", "")
    code = code.replace("</header>", "")
    code = code.replace("set option", "set_option")
    while "  := by" in code:
        code = code.replace("  := by", " := by")
    code = code.strip()
    return code


def simplify_lean_header(header):
    lines = header.split("\n")
    lines_simplified = []
    lib_common = []
    variables = {}
    for line in lines:
        tokens = line.split()
        # Merge imports
        if line.startswith("import ") and len(tokens) == 2 and "." in tokens[1]:
            lib = tokens[1].split(".")[0]
            lib_common.append(lib)
        else:
            # Remove duplicate variable lines
            if line.startswith("variable"):
                if line in variables:
                    continue
                else:
                    variables[line] = True
            lines_simplified.append(line)
    header_common = []
    for lib in set(lib_common):
        header_common.append(f"import {lib}")
    return "\n".join(header_common + lines_simplified)


def merge_lora_model(base_model_name, lora_model, merged_model_name):
    print(f"Merging LoRA model: base model {base_model_name}, lora model {lora_model}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(lora_model)
    if tokenizer_base.chat_template != tokenizer.chat_template:
        print("Chat template was changed. Updating the base model.")
        # The vocabulary of the base model needs to be updated before it can
        # be merged with the LoRA model.
        base_model, tokenizer_base = setup_chat_format(base_model, tokenizer_base)
    merged_model = PeftModel.from_pretrained(base_model, lora_model)
    merged_model = merged_model.merge_and_unload()
    merged_model = merged_model.to(torch.bfloat16)
    merged_model.push_to_hub(merged_model_name, private=True)
    tokenizer.push_to_hub(merged_model_name, private=True)


def maybe_merge_lora_model(model):
    """Merge the LoRA model if the merged model does not exist."""
    api = HfApi()

    model_info = api.model_info(model)
    is_lora = "adapter_config.json" in [sib.rfilename for sib in model_info.siblings]
    if not is_lora:
        return model

    model_merged = f"{model}_merged"
    try:
        model_info_merged = api.model_info(model_merged)
    except RepositoryNotFoundError:
        merge_lora_model(
            model_info.card_data["base_model"],
            model, model_merged
        )
    return model_merged


def setup_chat_format(model, tokenizer, resize_to_multiple_of=None):
    """
    Originally from trl.models.utils.
    Changed to only set EOS, but not BOS or PAD.
    """

    chat_format = ChatMlSpecialTokens()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.add_special_tokens({
        "additional_special_tokens": [chat_format.bos_token, chat_format.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_results_from_dataset_repo(repo_id, split, filename="results.jsonl"):
    repo_id = resolve_hf_repo(repo_id)
    """
    Load previous results from a HuggingFace dataset repository.

    Args:
        repo_id: HuggingFace dataset repository ID
        split: The split to load results for
        filename: Name of the file to load (default: "results.jsonl")

    Returns:
        List of previous sample statistics, or empty list if not found
    """
    try:
        api = HfApi()
        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = os.path.join(tmpdir, filename)
            api.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=tmpdir
            )
            previous_results = []
            with open(local_file, 'r') as f:
                for line in f:
                    sample_result = json.loads(line.strip())
                    if sample_result.get("split") == split:
                        previous_results.append(sample_result)
            if previous_results:
                print(f"Loaded {len(previous_results)} previous sample statistics")
            return previous_results
    except Exception as e:
        print(f"Could not load previous results: {e}")
    return []


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def _upload_file_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
    """Upload file to HuggingFace with retry logic for connection errors."""
    return api.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message
    )


def save_results_to_dataset_repo(results_dict, repo_id, filename="results.jsonl"):
    """
    Save results dictionary as a JSONL file to a HuggingFace dataset repository.
    """
    repo_id = resolve_hf_repo(repo_id)
    api = HfApi()

    existing_results = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = os.path.join(tmpdir, filename)
            api.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=tmpdir
            )
            with open(local_file, 'r') as f:
                for line in f:
                    existing_results.append(json.loads(line.strip()))
    except Exception:
        pass

    existing_by_split_and_idx = {}
    for result in existing_results:
        key = (result["split"], result["sample_idx"])
        existing_by_split_and_idx[key] = result

    for split, sample_stats in results_dict.items():
        for stat in sample_stats:
            stat["split"] = split
            key = (split, stat["sample_idx"])
            existing_by_split_and_idx[key] = stat

    with tempfile.TemporaryDirectory() as tmpdir:
        results_file = os.path.join(tmpdir, filename)
        with open(results_file, 'w') as f:
            for result in sorted(existing_by_split_and_idx.values(), key=lambda x: (x["split"], x["sample_idx"])):
                f.write(json.dumps(result) + '\n')

        _upload_file_with_retry(
            api=api,
            path_or_fileobj=results_file,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {filename}"
        )
        print(f"\nResults saved to {repo_id}/{filename}")


def clear_repo(repo_id, splits):
    """Clear a HuggingFace dataset repository by pushing empty datasets and removing result files."""
    repo_id = resolve_hf_repo(repo_id)
    ds_final = {split: Dataset.from_dict({}) for split in splits}
    DatasetDict(ds_final).push_to_hub(repo_id, private=True, commit_message="Clear")

    api = HfApi()
    for result_file in ["results.jsonl", "results.json"]:
        try:
            api.delete_file(
                path_in_repo=result_file,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Remove {result_file}"
            )
            print(f"Removed {result_file} from {repo_id}")
        except Exception:
            pass

def push_dataset_to_hub(output_repo, split, ds_current, commit_message):
    """Push dataset to HuggingFace hub."""
    output_repo = resolve_hf_repo(output_repo)
    try:
        ds_to_push = load_dataset(output_repo)
    except (DatasetNotFoundError, DatasetGenerationError):
        ds_to_push = {}
    if isinstance(ds_to_push, DatasetDict):
        ds_to_push[split] = ds_current
        ds_to_push.push_to_hub(output_repo, private=True, commit_message=commit_message)
    else:
        DatasetDict({split: ds_current}).push_to_hub(output_repo, private=True, commit_message=commit_message)
