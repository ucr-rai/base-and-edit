#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging

from accelerate import Accelerator
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    get_kbit_device_map,
    get_peft_config,
)
from trl import SFTTrainer
from config import Config
from train_utils import prepare_training

logger = logging.getLogger(__name__)


def main():
    # This must be done before H4ArgumentParser with the latest `accelerate`
    accelerator = Accelerator()
    device_map = get_kbit_device_map()
    num_devices = accelerator.num_processes

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig, Config))
    model_args, data_args, training_args, args = parser.parse()

    last_checkpoint, raw_datasets, model, tokenizer = prepare_training(
        logger, model_args, data_args, training_args, args, num_devices, device_map)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()
    else:
        logger.info("Skipping push_to_hub because push_to_hub=false")
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
