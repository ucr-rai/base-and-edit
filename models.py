import torch
import os
from types import SimpleNamespace
from typing import Union, List
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from tenacity import retry, wait_random_exponential
from vllm import LLM
from vllm.lora.request import LoRARequest
from dotenv import load_dotenv


load_dotenv()


class Model:
    def generate(self, prompts: Union[str, List[str]]):
        pass


class OpenAIModel(Model):
    def __init__(self, api_key, model_name, base_url=None, chat=True):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.chat = chat
        self.additional_kwargs = {}
        if model_name.startswith("openrouter-"):
            if api_key is None:
                api_key = os.getenv("OPENROUTER_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            # e.g., openrouter-qwen:qwen3-235b-a22b-2507 -> qwen/qwen3-235b-a22b-2507
            # (to distinguish from vLLM models)
            self.model_name = self.model_name[11:].replace(":", "/")
        elif model_name.startswith("wandb-"):
            if api_key is None:
                api_key = os.getenv("WANDB_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url="https://api.inference.wandb.ai/v1")
            self.model_name = self.model_name[6:].replace(":", "/")
        elif base_url is None:
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            if "gpt-5" in model_name:
                if "[reasoning_effort=low]" in model_name:
                    self.additional_kwargs["reasoning_effort"] = "low"
                elif "[reasoning_effort=high]" in model_name:
                    self.additional_kwargs["reasoning_effort"] = "high"
                elif "[reasoning_effort=minimal]" in model_name:
                    self.additional_kwargs["reasoning_effort"] = "minimal"
                if "reasoning_effort" in self.additional_kwargs:
                    print("Using reasoning effort:", self.additional_kwargs["reasoning_effort"])
                    self.model_name = model_name[:model_name.index("[")]
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    @retry(wait=wait_random_exponential(multiplier=1, max=1000),
           before_sleep=lambda retry_state:
           print(f"Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}"))
    def generate(self, prompt: str, max_new_tokens=None, return_usage=False, sampling_params=None):
        if sampling_params is None:
            sampling_params = {}

        api_params = {**self.additional_kwargs, **sampling_params}
        if max_new_tokens is not None and "max_tokens" not in sampling_params:
            api_params["max_tokens"] = max_new_tokens

        if self.chat:
            if isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]
            if "gpt-5" in self.model_name:
                if "max_tokens" in api_params:
                    api_params["max_completion_tokens"] = api_params.pop("max_tokens")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **api_params,
            )
            content = response.choices[0].message.content
        else:
            response = self.client.completions.create(
                model=self.model_name, prompt=prompt, **api_params,
            )
            content = response.choices[0].text.strip()
        if return_usage:
            return content, response.usage
        else:
            return content


class vLLMModel(Model):
    def __init__(self, model_name, tensor_parallel_size, dtype=torch.bfloat16, chat=True, lora_path=None):
        self.model_name = model_name
        self.chat = chat
        self.lora_path = lora_path
        if "@" in model_name:
            model_name, revision = model_name.split("@")
        else:
            revision = None
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "0")) or None
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        seed_env = os.environ.get("VLLM_SEED")
        seed = int(seed_env) if seed_env else None
        llm_kwargs = dict(
            model=model_name, revision=revision,
            tensor_parallel_size=tensor_parallel_size, dtype=dtype, trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if seed is not None:
            llm_kwargs["seed"] = seed
        if lora_path is not None:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
        self.llm = LLM(**llm_kwargs)

    def _get_tokenizer(self):
        # vLLM tokenizer access changed across releases.
        if hasattr(self.llm, "get_tokenizer"):
            tokenizer = self.llm.get_tokenizer()
            if tokenizer is not None:
                return tokenizer

        tokenizer_holder = getattr(self.llm.llm_engine, "tokenizer", None)
        if tokenizer_holder is None:
            raise AttributeError("Unable to locate tokenizer on the vLLM engine")

        tokenizer = getattr(tokenizer_holder, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        return tokenizer_holder

    def generate(self, prompts, max_new_tokens=None, return_usage=False, sampling_params=None, chat_template_kwargs=None):
        if sampling_params is None:
            sampling_params = {}
        if chat_template_kwargs is None:
            chat_template_kwargs = {}

        valid_inputs = []
        valid_indices = []
        contents = []

        tokenizer = self._get_tokenizer()

        lengths = []
        for i, prompt in enumerate(prompts):
            if self.chat:
                if isinstance(prompt, str):
                    input_msgs = [{"role": "user", "content": prompt}]
                else:
                    assert isinstance(prompt, list) and isinstance(prompt[0], dict)
                    input_msgs = prompt
                token_ids = tokenizer.apply_chat_template(input_msgs, tokenize=True, **chat_template_kwargs)
            else:
                input_msgs = prompt
                token_ids = tokenizer.encode(prompt)

            if len(token_ids) + 5 > self.llm.llm_engine.model_config.max_model_len:
                contents.append("")
            else:
                valid_inputs.append(input_msgs)
                valid_indices.append(i)
                contents.append(None)
                lengths.append(self.llm.llm_engine.model_config.max_model_len - len(token_ids) - 5)

        if valid_inputs:
            if self.chat:
                vllm_sampling_params = self.llm.get_default_sampling_params()
                for key, value in sampling_params.items():
                    setattr(vllm_sampling_params, key, value)
                if max_new_tokens is not None and "max_tokens" not in sampling_params:
                    vllm_sampling_params.max_tokens = max_new_tokens
                chat_kwargs = dict(chat_template_kwargs=chat_template_kwargs)
                if self.lora_path:
                    chat_kwargs["lora_request"] = LoRARequest("adapter", 1, self.lora_path)
                outputs = self.llm.chat(valid_inputs, vllm_sampling_params, **chat_kwargs)
            else:
                vllm_sampling_params_list = []
                for l in lengths:
                    vllm_sampling_params = self.llm.get_default_sampling_params()
                    for key, value in sampling_params.items():
                        setattr(vllm_sampling_params, key, value)
                    if "max_tokens" in sampling_params:
                        vllm_sampling_params.max_tokens = min(sampling_params["max_tokens"], l)
                    elif max_new_tokens:
                        vllm_sampling_params.max_tokens = min(max_new_tokens, l)
                    else:
                        vllm_sampling_params.max_tokens = l
                    vllm_sampling_params_list.append(vllm_sampling_params)
                gen_kwargs = {}
                if self.lora_path:
                    gen_kwargs["lora_request"] = LoRARequest("adapter", 1, self.lora_path)
                outputs = self.llm.generate(valid_inputs, vllm_sampling_params_list, **gen_kwargs)

            for j, output in enumerate(outputs):
                original_index = valid_indices[j]
                content = output.outputs[0].text
                contents[original_index] = content

        if return_usage:
            return contents, None
        else:
            return contents


def get_model(model_name, dtype="auto", api_key=None, base_url=None, chat=True, lora_path=None):
    if model_name.startswith("gemini"):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return OpenAIModel(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model_name=model_name,
            chat=chat,
        )
    if base_url is None and "/" in model_name:
        return vLLMModel(
            model_name=model_name,
            tensor_parallel_size=torch.cuda.device_count(), dtype=dtype, chat=chat,
            lora_path=lora_path,
        )
    else:
        return OpenAIModel(api_key=api_key, base_url=base_url, model_name=model_name, chat=chat)
