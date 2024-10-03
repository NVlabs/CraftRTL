# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import json
import tqdm
import os
import logging
import random
import requests
from datetime import datetime
import argparse
import time

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from utils import compute_fingerprint, retry_with_exponential_backoff

from pathlib import Path


def make_requests(*args, **kwargs):
    if "nvcf" in kwargs["engine"]:
        return make_nvcf_requests(*args, **kwargs)
    else:
        return make_local_requests(*args, **kwargs)


def make_nvcf_requests(engine, prompts, max_tokens, temperature, n, seed=87):

    if engine == 'nvcf-llama3-8b-instruct':
        model = 'meta/llama3-8b-instruct'
    elif engine == 'nvcf-llama3-70b-instruct':
        model = 'meta/llama3-70b-instruct'
    elif engine == 'nvcf-mixtral-8x22b-instruct':
        model = "mistralai/mixtral-8x22b-instruct-v0.1"
    elif engine == 'nvcf-mistral-large':
        model = "mistralai/mistral-large"
    elif engine == 'nvcf-nemotron-4-340b-instruct':
        model = "nvidia/nemotron-4-340b-instruct"

    api_key = os.environ["API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"]
        
    from openai import OpenAI
    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = api_key
    )
    completion = client.chat.completions.create(
      model=model,
      messages=prompts,
      temperature=temperature,
      top_p=1,
      max_tokens=max_tokens,
      stream=False
    )
    result = completion.dict()
    result["system_fingerprint"] = compute_fingerprint(
        result["choices"][0]["message"]["content"]
    )
    result["choices"][0]["finish_reason"] = "stop"
    return result


def make_local_requests(engine, prompts, max_tokens, temperature, n, seed=87):

    if make_local_requests.pipeline is None:
        import transformers
        make_local_requests.pipeline = transformers.pipeline(
            "text-generation",
            model="codellama/CodeLlama-7b-Instruct-hf",
            #torch_dtype=torch.bfloat16,
            device_map="auto",
            model_kwargs={"cache_dir": "./code_repair_examples/cache"}
        )
    
    hyper_params = dict(
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=max_tokens,
    )
    out = make_local_requests.pipeline(
        prompts[-1]['content'],
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        return_full_text=False,
        **hyper_params
    )
    output = out[0]['generated_text']
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": output}}],
        "system_fingerprint": compute_fingerprint(output),
    }


    response = make_local_requests.session.post(
        "http://0.0.0.0:8080/completion",
        json={
            "engine": engine,
            "prompt": prompts,
            "max_tokens": target_length,
        },
    )
    if response.status_code == 200:
        output = response.json()["item"][
            "result"
        ]  # depends on the local endpoint response schema
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": output}}],
            "system_fingerprint": compute_fingerprint(output),
        }


make_local_requests.session = requests.Session()
make_local_requests.pipeline = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, help="The llm to use.", default="mixtral")
    parser.add_argument(
        "--content", type=str, help="The llm to use.", default="Write me a function in python calculating sum of two numbers.?"
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs", default=5, type=int, help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate.",
        default=1,
    )
    parser.add_argument(
        "--best_of",
        type=int,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server.",
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    prompt = [{"content": args.content, "role": "user"}]
    results = make_requests(
        engine=args.engine,
        prompts=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n,
    )
    print(json.dumps(results))
