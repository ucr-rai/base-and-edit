import logging
import sys

import torch
import datasets
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    get_checkpoint,
    get_datasets,
    get_quantization_config,
    get_tokenizer,
)

from utils import setup_chat_format


def prepare_training(logger, model_args, data_args, training_args, args, num_devices, device_map):
    set_seed(training_args.seed)

    assert training_args.output_dir, "Output directory must be specified."
    training_args.logging_dir = training_args.output_dir

    if args.batch_size != -1:
        training_args.gradient_accumulation_steps = (
            args.batch_size
            // training_args.per_device_train_batch_size
            // num_devices
        )
        assert args.batch_size == (
            training_args.gradient_accumulation_steps
            * num_devices
            * training_args.per_device_train_batch_size
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank},"
        f" device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f" distributed training: {bool(training_args.local_rank != -1)},"
        f" 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        branch=data_args.dataset_branch,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )
    logger.info("Training on the following datasets and their proportions: "
                f"{[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}")

    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    tokenizer = get_tokenizer(model_args, data_args)

    model = model_args.model_name_or_path

    if args.setup_template:
        # For ChatML we need to add special tokens and resize the embedding layer
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            quantization_config=quantization_config,
        )
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model, tokenizer = setup_chat_format(model, tokenizer)

    return last_checkpoint, raw_datasets, model, tokenizer
