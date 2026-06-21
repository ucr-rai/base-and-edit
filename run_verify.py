import argparse

from datasets import load_dataset, DatasetDict
from datasets.exceptions import DatasetNotFoundError
from verify import verify, start_kimina_server, stop_kimina_server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output")
    parser.add_argument("--output_key", type=str, default="formal_proof")
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    args = parser.parse_args()

    print(f"Running verification for {args.output}")

    server_process, lean_port = start_kimina_server()

    try:
        ds_results = load_dataset(f"{args.output}_lean")
    except DatasetNotFoundError:
        ds_results = {}

    commit_messages = []
    for split in args.splits:
        ds = load_dataset(args.output, split=split)
        result = verify(ds, output_key=args.output_key, lean_port=lean_port)
        ds_results[split] = result
        cnt_passed = sum(item["passed"] for item in result)
        rate_passed = cnt_passed * 1. / len(result)
        commit_messages.append(f"{split}: {len(ds)} examples, pass rate {rate_passed:.4f}")

    ds_results = DatasetDict(ds_results)
    ds_results.push_to_hub(f"{args.output}_lean", commit_message=",".join(commit_messages), private=True)

    stop_kimina_server(server_process)
