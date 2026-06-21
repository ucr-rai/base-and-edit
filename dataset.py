import argparse
import json
from datasets import load_dataset, concatenate_datasets
from utils import remove_lean_comments, simplify_lean_header
from prompts import prompt_informal2formal_with_problem
from huggingface_hub import hf_hub_download


def load_test_data(name, num_examples=-1):
    split = "test" if num_examples == -1 else f"test[:{num_examples}]"
    if name == "herald":
        return load_dataset("zhouxingshi/Herald_proofs_processed", split=split)
    elif name == "lean-workbook":
        return load_dataset("zhouxingshi/Lean-workbook-proofs_processed", split=split)
    elif name == "minif2f":
        return load_dataset("zhouxingshi/minif2f_processed", split=split)
    else:
        raise NameError(name)


def construct_prompt_for_rl(example):
    assert len(example["messages"]) == 2
    prefix  = example["messages"][1]["content"]
    keyword = ":= by"
    if keyword not in prefix:
        print(prefix)
    assert keyword in prefix
    prefix = prefix[:prefix.find(keyword)+len(keyword)]
    return {
        "prompt_rl": [
            example["messages"][0],
            {"role": "assistant", "content": prefix}
        ]
    }


def process_herald():
    dataset = load_dataset("FrenzyMath/Herald_proofs", split="train")
    print("Initial number of examples:", len(dataset))

    def filter_example(example):
        # Invalid example
        if "Unable to analyze" in example["header"]:
            return False
        # Too complicated
        if "theorem " in example["header"]:
            return False
        if "def " in example["header"]:
            return False
        return True

    data = dataset.filter(filter_example)
    print("Filtered dataset:", len(data))

    data = data.map(
        lambda example: {
            "messages": [
                {
                    "role": "user",
                    "content": prompt_informal2formal_with_problem.format(
                        problem=example["informal_theorem"],
                        proof=example["informal_proof"],
                    )
                },
                {
                    "role": "assistant",
                    "content": (simplify_lean_header(example["header"])
                                + "\n" + remove_lean_comments(example["formal_proof"])),
                },
            ]
        },
        num_proc=16
    )

    print("Example:")
    print(json.dumps(data[0]["messages"], indent=4))

    data = data.train_test_split(test_size=0.1)
    data.push_to_hub("zhouxingshi/Herald_proofs_processed", private=True)


def get_lean_workbook_id2problem():
    original_path = hf_hub_download(
        repo_id="internlm/Lean-Workbook",
        filename="lean_workbook.json",
        repo_type="dataset"
    )
    with open(original_path) as file:
        data_original = json.loads(file.read())

    id2problem = {}
    for example in data_original:
        id = example["formal_statement"].split()[1]
        assert id.startswith("lean_workbook")
        id2problem[id] = {
            "informal": example["natural_language_statement"],
            "formal": example["formal_statement"],
        }
    return id2problem


def process_lean_workbook():
    id2problem = get_lean_workbook_id2problem()
    data = concatenate_datasets([
        load_dataset("zhouxingshi/Lean-workbook-proofs_GPT_10k", split="train"),
        load_dataset("zhouxingshi/Lean-workbook-proofs_GPT_10k-15k", split="train"),
        load_dataset("zhouxingshi/Lean-workbook-proofs_GPT_15k-20k", split="train"),
        load_dataset("zhouxingshi/Lean-workbook-proofs_GPT_20k-25k", split="train"),
        load_dataset("zhouxingshi/Lean-workbook-proofs_GPT_25k-30k", split="train"),
    ])
    data = data.map(lambda example:
                    {"informal_problem": id2problem[example["problem_id"]]["informal"]})
    data = data.train_test_split(test_size=0.02)
    data.push_to_hub("zhouxingshi/Lean-workbook-proofs_processed", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        choices=["herald", "lean-workbook", "minif2f"])
    args = parser.parse_args()

    if not args.dataset or args.dataset == "herald":
        process_herald()
    if not args.dataset or args.dataset == "lean-workbook":
        process_lean_workbook()
