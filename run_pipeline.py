import argparse
import os
import portpicker
import subprocess
from datetime import datetime
from huggingface_hub import HfApi


def get_training_batch_size(base_model, batch_size):
    if batch_size:
        return batch_size
    if "-7B" in base_model or '-8B' in base_model:
        return 1
    elif "-4B" in base_model or "-3B" in base_model:
        return 1
    elif "-1.5B" in base_model or "-1.7B" in base_model:
        return 2
    elif "-0.5B" in base_model or "-0.6B" in base_model:
        return 8
    else:
        return None


def shorten_eval_data_name(eval_data):
    eval_data_name = os.path.basename(eval_data)
    if "_" in eval_data_name:
        eval_data_name = eval_data_name[:eval_data_name.rfind("_")]
    return eval_data_name


def get_sketcher_commands(stage, args, unknown_args_str, test=False):
    """Training the sketcher."""

    init_model = args.models[0].split("/")[-1] if "/" in args.models[0] else args.models[0]

    if args.eval_data:
        output_data = f"{args.models[-1]}_eval_{shorten_eval_data_name(args.eval_data)}"
        if stage == "generate":
            command = ("python gen_data.py "
                       f"{args.models[-1]} {args.eval_data} {output_data} "
                       "--input_key informal_proof --output_key formal_proof --splits test "
                       f"--num_samples {args.eval_num_samples} ")
        else:
            raise NotImplementedError
    else:
        data_prefix = args.informal_proof_data[:args.informal_proof_data.find("_")]
        data_names = [
            f"{args.username}/{args.informal_proof_data}",
            f"{args.username}/{data_prefix}_{args.informal_proof_model}_{init_model}_v1_{args.num_examples_cs}_base{args.suffix}"
        ] + [f"{model}_infer" if "@" not in model else f"{model[:model.index('@')]}_infer"
             for model in args.models[1:]]
        for i in range(1, len(data_names) - 1):
            data_names[i] += "_built"

        splits = "test" if test else "train_base"

        if stage == "generate":
            # FIXME add formal_problem for legacy lean workbook data
            command = ("python gen_data.py "
                       f"{args.models[-1]} {data_names[-2]} {data_names[-1]} "
                       "--input_key informal_proof --output_key formal_proof ")
            if len(args.models) == 1:
                command += f"--num_examples {args.num_examples_cs} --njobs {args.njobs} --splits train "
            else:
                command += f"--splits {splits} "
            if not test and len(args.models) > 1:
                assert data_names[-2].endswith("_built")
                command += f"--retry {data_names[-2][:-6]} "
        elif stage == "build":
            command = f"python build_dataset.py {data_names[-1]} --informal_proof_data {args.username}/{args.informal_proof_data} "
            if len(args.models) > 1:
                command += f"--cont {data_names[-2]}"
        elif stage == "train":
            command = get_training_command(data_names[-1], args)
        else:
            raise NotImplementedError

    if args.add_extra_args or stage == "train":
        command += f"{unknown_args_str} "

    return command



def get_replacement_commands(stage, args, unknown_args_str):
    if stage == "replacement_train":
        dataset = f"{args.base_dataset}_built{args.built_suffix}"
        output_dir = os.path.join(os.path.expanduser("~/checkpoints"), args.hub_model_id)
        num_gpus = args.slurm_num_gpus if args.slurm else (os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1)
        command = (
            "ACCELERATE_LOG_LEVEL=info accelerate launch "
            "--config_file recipes/deepspeed_zero3.yaml "
            f"--num_processes={num_gpus} --main_process_port={portpicker.pick_unused_port()} "
            f"run_sft.py recipes/sft.yaml --dataset={dataset} --model_name_or_path={args.base_model} "
            f"--hub_model_id={args.hub_model_id} --output_dir={output_dir} ")
        batch_size = get_training_batch_size(args.base_model, args.batch_size)
        if batch_size:
            command += f"--per_device_eval_batch_size={batch_size} --per_device_train_batch_size={batch_size} "
    elif stage == "replacement_build":
        # FIXME hard-coded for now
        command = (f"python build_dataset.py {args.base_dataset} --type replacement ")
    else:
        raise NotImplementedError

    if args.add_extra_args or "train" in stage:
        command += f"{unknown_args_str} "

    return command


def get_training_command(dataset, args):
    dataset += f"_built{args.built_suffix}"
    output_dir = os.path.join(os.path.expanduser("~/checkpoints"), args.hub_model_id)
    num_gpus = args.slurm_num_gpus if args.slurm else (os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1)
    command = (
        "ACCELERATE_LOG_LEVEL=info accelerate launch "
        "--config_file recipes/deepspeed_zero3.yaml "
        f"--num_processes={num_gpus} --main_process_port={portpicker.pick_unused_port()} "
        f"run_sft.py recipes/sft.yaml --dataset={dataset} --model_name_or_path={args.base_model} "
        f"--hub_model_id={args.hub_model_id} --output_dir={output_dir} ")
    batch_size = get_training_batch_size(args.base_model, args.batch_size)
    if batch_size:
        command += f"--per_device_eval_batch_size={batch_size} --per_device_train_batch_size={batch_size} "
    return command


def get_formal_statement_commands(args):
    """Prepare test data and generate formal statements from informal problem statements."""
    output_name = f"{args.username}/{os.path.basename(args.eval_data)}_{args.models[0]}_formal-problem"
    command = (f"python gen_data.py {args.models[0]} {args.eval_data} {output_name} "
                   f"--input_key problem --output_key formal_problem --splits test ")
    commands = [command] + [f"{command} --retry {output_name}" for _ in range(args.retry)]
    return commands


def gen_generate_informal_proof_commands(args, unknown_args_str):
    """Generate informal proofs."""
    output_name = (f"{args.username}/{os.path.basename(args.eval_data)}_"
                   f"{os.path.basename(args.models[0])}_informal-proof")
    command = (f"python gen_data.py {args.models[0]} {args.eval_data} {output_name} "
               f"--input_key informal_problem --output_key informal_proof --splits test")
    if args.add_extra_args:
        command += f"{unknown_args_str} "
    return command


def submit_slurm_job(command, args, dependency_job_id=None):
    """Submit a SLURM job and capture job ID."""

    if "build_dataset.py" in command:
        resources = f"--mem={args.slurm_mem}"
    else:
        resources = f"--cpus-per-task={args.slurm_num_cpus} --gres=gpu:{args.slurm_num_gpus} --mem={args.slurm_mem}"

    if args.slurm_always_gpu and not "--gres=gpu:" in resources:
        resources += " --gres=gpu:1 "

    slurm_cmd = (f"sbatch -p {args.slurm_partition} {resources} --time={args.slurm_hours}:00:00 "
                 f"--output=slurm/slurm-%j.out --error=slurm/slurm-%j.out ")
    if args.slurm_account:
        slurm_cmd += f"--account {args.slurm_account} "
    if dependency_job_id:
        slurm_cmd += f"--dependency=afterok:{dependency_job_id} "

    slurm_cmd += (f"--wrap '"
        f"source ~/miniconda3/etc/profile.d/conda.sh; conda activate {args.slurm_conda}; "
        f"export PATH=~/.elan/bin:$PATH; "
        f"{command} && ./notify.sh \"{command}\"'")

    print(f"Submitting SLURM job: {slurm_cmd}")

    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id}")
        return job_id
    else:
        print(f"Error submitting job: {result.stderr}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("stage", type=str,
                        choices=[
                                # generate informal proofs
                                "generate-informal-proof",
                                # Training the sketcher
                                "generate", "build", "train", "auto", "auto-trained",
                                # Preparing test data
                                "generate-formal-statement",
                                # Replacement model
                                "replacement_build", "replacement_train",
                        ])
    parser.add_argument("--models", type=str, nargs="+", default=["gpt-4o"], help="Chain of models.")
    parser.add_argument("--num_examples_cs", type=int, default=5000, help="Number of examples for cold start.")
    parser.add_argument("--test", action="store_true", help="Do the test set only.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model for fine-tuning.")
    parser.add_argument("--hub_model_id", type=str, default="model", help="Hub model ID for training output.")
    parser.add_argument("--base_dataset", type=str)
    parser.add_argument("--informal_proof_data", type=str, default="Lean-workbook-proofs_GPT_from-problem", help="Initial data with informal proofs.")
    parser.add_argument("--informal_proof_model", type=str, default="gpt-4o", help="Model used to build the initial data with informal proofs.")
    parser.add_argument("--eval_data", type=str, help="Evaluate a trained sketcher on a different dataset.")
    parser.add_argument("--eval_num_samples", type=int, default=1)
    parser.add_argument("--add_extra_args", action="store_true", help="Unless 'train' is being used, we need this flag to explicitly pass unknown args.")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--suffix", type=str, default="", help="Suffix for the dataset name.")
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--username", type=str)
    parser.add_argument("--njobs", type=int, default=32, help="Number of parallel jobs for API calls.")
    parser.add_argument("--built_suffix", type=str, default="")

    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--slurm_num_gpus", type=int, default=4)
    parser.add_argument("--slurm_num_cpus", type=int, default=32)
    parser.add_argument("--slurm_mem", type=str, default="128g")
    parser.add_argument("--slurm_hours", type=int, default=48)
    parser.add_argument("--slurm_partition", type=str, default="gpu")
    parser.add_argument("--slurm_conda", type=str, default="prover")
    parser.add_argument("--slurm_always_gpu", action="store_true")
    parser.add_argument("--slurm_account", type=str)

    args, unknown_args = parser.parse_known_args()
    unknown_args_str = " ".join(unknown_args)

    if not args.username:
        api = HfApi()
        args.username = api.whoami()['name']

    for i in range(len(args.models)):
        model = args.models[i]
        if not model.startswith("gpt-") and '/' not in model:
            args.models[i] = f"{args.username}/{model}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.hub_model_id = f"{args.hub_model_id}_{timestamp}"

    commands = []
    if args.stage == "generate-informal-proof":
        commands.append(gen_generate_informal_proof_commands(args, unknown_args_str))
    elif args.stage.startswith("replacement_"):
        commands.append(get_replacement_commands(args.stage, args, unknown_args_str))
    elif args.stage.startswith("auto"):
        if args.stage == "auto":
            commands.append(get_sketcher_commands("train", args, unknown_args_str))
            args.models.append(f"{args.username}/{args.hub_model_id}")
        commands.append(get_sketcher_commands("generate", args, unknown_args_str, test=True))
        commands.append(get_sketcher_commands("generate", args, unknown_args_str, test=False))
        commands.append(get_sketcher_commands("build", args, unknown_args_str, test=False))
    elif args.stage in ["generate", "build", "train"]:
        commands.append(get_sketcher_commands(
            args.stage, args, unknown_args_str, test=args.test))
    elif args.stage == "generate-formal-statement":
        commands.extend(get_formal_statement_commands(args))
    else:
        raise NotImplementedError

    if args.slurm:
        os.makedirs("slurm", exist_ok=True)
        job_ids = []
        for i, command in enumerate(commands):
            job_id = submit_slurm_job(command, args, job_ids[-1] if i > 0 else None)
            if job_id:
                job_ids.append(job_id)
            else:
                print(f"Failed to submit job {i+1}, stopping pipeline")
                break
    else:
        for command in commands:
            print(f"Running command: {command}")
            os.system(f"{command} && ./notify.sh \"{command}\"")
