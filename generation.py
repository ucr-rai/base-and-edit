"""Data generation thread for parallel model inference."""

import re
import yaml
from threading import Thread
from datasets import Dataset, concatenate_datasets
from prompts import *
from utils import remove_lean_comments, api_price
from models import get_model
from base_and_edit.generate_answers import postprocess as ranked_answers_postprocess


def resolve_model_config(model: str) -> dict:
    if ".yaml:" in model:
        config_filename, model_type, model_name = model.split(":")
        if "@" in model_name:
            model_name, checkpoint_name = model_name.split("@")
        else:
            checkpoint_name = None
        with open(config_filename, 'r') as file:
            config = yaml.safe_load(file)
        config = dict(config[model_type][model_name])
        if checkpoint_name:
            config["model"] = checkpoint_name
        return config
    return {"model": model}


class DataThread(Thread):

    def __init__(self, input, output, input_key, output_key,
                 model, idx, progress_bar, batch_size=1):
        super().__init__()
        self.input = input
        self.output = output
        self.input_key = input_key
        self.output_key = output_key
        self.idx = idx
        self.progress_bar = progress_bar
        self.batch_size = batch_size

        self.ds = input
        self.ds_output = None

        config = resolve_model_config(model)
        self.model_name = config["model"]
        self.chat = config.get("chat", True)
        lora_path = config.get("lora_path")
        self.model = get_model(self.model_name, chat=self.chat, lora_path=lora_path)
        if config.get("prompt_file"):
            with open(config["prompt"], 'r') as f:
                self.prompt = f.read()
        elif "prompt" in config:
            self.prompt = config["prompt"]
        else:
            self.prompt = None
        self.post_processing = config.get("post_processing")
        self.system_message = config.get("system_message")
        self.sampling_params = config.get("sampling_params", {})
        self.chat_template_kwargs = config.get("chat_template_kwargs", {})
        self.num_answers = config.get("num_answers", 5)

        if "max_tokens" not in self.sampling_params:
            self.sampling_params["max_tokens"] = 10000

    def _process_input(self, example):
        if self.output_key == "informal_proof":
            if self.input_key == "formal_proof":
                prompt = prompt_lean2nl.format(proof=remove_lean_comments(example["full_proof"]))
            elif self.input_key == "formal_proof_without_comments":
                example["formal_proof"] = example["full_proof"] = remove_lean_comments(example["full_proof"])
                prompt = prompt_lean2nl_without_comments.format(proof=example["full_proof"])
            else:
                if self.model_name == "gpt-4.1-mini":
                    prompt = prompt_solve_informal_gpt41
                else:
                    prompt = prompt_solve_informal
                prompt = prompt.format(informal_problem=example[self.input_key])

        elif self.output_key == "formal_proof":
            if self.input_key == "informal_proof":
                prompt = prompt_translate_sketch_only.format(
                    informal_problem=example["informal_problem"],
                    formal_problem=example["formal_problem"],
                    informal_proof=example["informal_proof"],
                )

                def remove_header_placeholder(header):
                    if ":= sorry" in header:
                        header = header[:header.find(":= sorry") + 2]
                    elif ":= by" in header:
                        header = header[:header.find(":= by")+5]
                    return header
                header = remove_header_placeholder(example["formal_proof"])
                prompt += "\n" + prompt_header.format(header=header)

            elif self.input_key in ["formal_statement", "formal_problem", "formal_with_answer2"]:
                informal = example.get("informal_problem") or example.get("problem") or ""
                fmt_kwargs = {"formal_statement": self._remove_sorry(example[self.input_key])}
                if "{informal_problem}" in self.prompt:
                    fmt_kwargs["informal_problem"] = informal
                prompt = self.prompt.format(**fmt_kwargs)
            else:
                raise NotImplementedError


        elif self.output_key == "formal_problem":
            answer_to_use = example.get("answer") or ""
            if example.get("if_change", False) and "answer_new" in example:
                answer_to_use = example["answer_new"] or ""
            prompt = self.prompt.format(problem=example[self.input_key], answer=answer_to_use)

        elif self.output_key == "ranked_answers":
            prompt = self.prompt.format(problem=example[self.input_key], num_answers=self.num_answers)

        elif self.output_key == "r2_fill_answer":
            task_family = example.get("task_family", "unknown")
            prompt = f"<task_family>{task_family}</task_family>\n\n" + self.prompt.format(
                informal_problem=example.get("problem", example.get("informal_problem", "")),
                formal_problem=example.get("formal_problem", ""),
                original_answer=example.get("original_answer", ""),
            )
        else:
            raise NotImplementedError

        if self.system_message is not None:
            prompt = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]

        example["_prompt"] = prompt
        return prompt

    def _process_output(self, example, output):
        example["_output_raw"] = output

        if self.output_key == "ranked_answers":
            return ranked_answers_postprocess(example, output, self.num_answers)

        if self.post_processing == "none":
            pass
        elif self.post_processing == "extract_code_and_concat":
            pattern = r'```lean4\s*\n.*?:=\s*by\s*\n(.*?)\n```'
            matches = re.findall(pattern, output, re.DOTALL)
            completion = matches[-1].rstrip() if matches else ""
            output = self._remove_sorry(example[self.input_key]) + "\n" + completion
        elif self.post_processing == "extract_code":
            try:
                matches = re.findall(r'```lean4\n(.*?)\n```', output, re.DOTALL)
                output = matches[-1].strip() if matches else "[ERROR] No Lean 4 code block found."
            except:
                output = "[ERROR]"
        elif self.post_processing == "concat":
            if "```" in output:
                output = output[:output.find("```")].rstrip()
            output = self._remove_sorry(example[self.input_key]) + "\n" + output
        elif self.post_processing == "remove_comments":
            pattern = r"/-- [\s\S]*? -/\n"
            output = re.sub(pattern, "", output, flags=re.DOTALL)
        else:
            raise NameError(self.post_processing)

        if self.output_key == "formal_problem":
            example["informal_problem"] = example[self.input_key]
            example["formal_proof"] = (header_default + "\n" + output).strip()
        elif self.output_key == "formal_proof" and "passed" in example:
            example["passed_theorem"] = example["passed"]

        example[self.output_key] = output
        return example

    def _remove_sorry(self, statement):
        if statement.strip().endswith("sorry"):
            statement = statement.strip()[:-len("sorry")].rstrip()
            if not statement.endswith("by"):
                statement += " by"
        return statement

    def run(self):
        count_prompt_tokens = 0
        count_completion_tokens = 0

        if "/" in self.model_name:
            examples_list = list(self.ds)
            for i in range(0, len(examples_list), self.batch_size):
                batch_examples = examples_list[i:i + self.batch_size]
                batch_prompts = [self._process_input(example) for example in batch_examples]
                outputs = self.model.generate(
                    batch_prompts, sampling_params=self.sampling_params,
                    chat_template_kwargs=self.chat_template_kwargs)
                new_examples = [
                    self._process_output(example, output)
                    for (example, output) in zip(batch_examples, outputs)
                ]
                ds_tmp = Dataset.from_list(new_examples)
                if self.ds_output is None:
                    self.ds_output = ds_tmp
                else:
                    self.ds_output = concatenate_datasets([self.ds_output, ds_tmp])
                self.progress_bar.update(len(ds_tmp))
        else:
            for i, example in enumerate(self.ds):
                prompt = self._process_input(example)
                output, usage = self.model.generate(
                    prompt, return_usage=True, sampling_params=self.sampling_params)
                self._process_output(example, output)

                count_prompt_tokens += usage.prompt_tokens
                count_completion_tokens += usage.completion_tokens
                example_batch = {k: [v] for k, v in example.items()}
                ds_tmp = Dataset.from_dict(example_batch)

                if self.ds_output is None:
                    self.ds_output = ds_tmp
                else:
                    self.ds_output = concatenate_datasets([self.ds_output, ds_tmp])

                self.progress_bar.update(1)

        cost = "unknown"
        if "gpt-" in self.model.model_name:
            cost = (api_price[self.model.model_name][0] * count_prompt_tokens
                    + api_price[self.model.model_name][1] * count_completion_tokens) / 1e6
        print(f"Job {self.idx} done.", f"Cost: {cost}" if cost != "unknown" else "")
