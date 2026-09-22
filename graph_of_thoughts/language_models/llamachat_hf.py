# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Ales Kubicek

import os
import torch
from typing import List, Dict, Union
from .abstract_language_model import AbstractLanguageModel


class Llama2HF(AbstractLanguageModel):
    """
    An interface to use LLaMA 2 models through the HuggingFace library.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "llama7b-hf", cache: bool = False
    ) -> None:
        """
        Initialize an instance of the Llama2HF class with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to an empty string.
        :type config_path: str
        :param model_name: Specifies the name of the LLaMA model variant. Defaults to "llama7b-hf".
                           Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config.get("prompt_token_cost", 0.0)
        self.response_token_cost: float = self.config.get("response_token_cost", 0.0)
        self.temperature: float = self.config.get("temperature", 0.6)
        self.top_k: int = self.config.get("top_k", 10)
        self.max_tokens: int = self.config.get("max_tokens", 4096)

        self._api_mode = "base_url" in self.config and "api_key" in self.config
        if self._api_mode:
            from openai import OpenAI

            self.base_url = self.config["base_url"]
            self.api_key = os.getenv("OPENAI_API_KEY", self.config["api_key"])
            if not self.api_key:
                raise ValueError("api_key required for Llama API (config or OPENAI_API_KEY)")
            self.organization = self.config.get("organization", "")
            self.stop = self.config.get("stop")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization or None,
            )
            return

        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]
        import transformers
        from huggingface_hub import get_token

        hf_token = self.config.get("token") or os.environ.get("HF_TOKEN") or get_token()
        hf_model_id = f"meta-llama/{self.model_id}"
        model_config = transformers.AutoConfig.from_pretrained(
            hf_model_id, token=hf_token
        )
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_model_id, token=hf_token
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
        )
        self.model.eval()
        torch.no_grad()
        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-generation"
        )

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Query the LLaMA 2 model for responses (API or local).

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the LLaMA 2 model (list of {"generated_text": str}).
        :rtype: List[Dict]
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if self._api_mode:
            # Many OpenAI-compatible APIs (e.g. Aval AI) allow at most n=1 per request.
            n_per_request = 1
            response = []
            for _ in range(num_responses):
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": query}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=n_per_request,
                    stop=self.stop,
                )
                self.prompt_tokens += resp.usage.prompt_tokens
                self.completion_tokens += resp.usage.completion_tokens
                response.extend(
                    [{"generated_text": (c.message.content or "").strip()} for c in resp.choices]
                )
            prompt_k = float(self.prompt_tokens) / 1000.0
            completion_k = float(self.completion_tokens) / 1000.0
            self.cost = self.prompt_token_cost * prompt_k + self.response_token_cost * completion_k
            self.logger.debug("Llama API cost: %s", self.cost)
        else:
            sequences = []
            inst = f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {query} [/INST]"
            for _ in range(num_responses):
                sequences.extend(
                    self.generate_text(
                        inst,
                        do_sample=True,
                        top_k=self.top_k,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_length=self.max_tokens,
                    )
                )
            response = [
                {"generated_text": sequence["generated_text"][len(inst) :].strip()}
                for sequence in sequences
            ]

        if self.cache:
            self.response_cache[query] = response
        return response

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_responses: The response list of dictionaries generated from the `query` method.
        :type query_responses: List[Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        return [query_response["generated_text"] for query_response in query_responses]
