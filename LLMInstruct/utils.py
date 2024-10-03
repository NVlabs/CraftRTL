# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import functools
import hashlib
import json
import os
import re
import random
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar, Dict

import openai
import tiktoken
import gzip
import pandas as pd
from datasets import Dataset, load_dataset


N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2
logger = logging.getLogger(__name__)


def read_jsonl(filename: str, ignore_error: bool = True) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with Path(filename).open("rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        try:
                            yield json.loads(line)
                        except Exception:
                            if ignore_error:
                                continue
                            raise
    else:
        with Path(filename).open("r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    try:
                        yield json.loads(line)
                    except Exception:
                        if ignore_error:
                            continue
                        raise


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    mode = "ab" if append else "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with Path(filename).open(mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with Path(filename).open(mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def read_data(data_dir, dataset_name) -> Dataset:
    if "csv" in dataset_name:
        df = pd.read_csv(os.path.join(data_dir, dataset_name))
        dataset = Dataset.from_pandas(df)
    elif "jsonl" in dataset_name:
        df = pd.DataFrame(read_jsonl(os.path.join(data_dir, dataset_name)))
        dataset = Dataset.from_pandas(df)
    elif "txt" in dataset_name:
        text = open(os.path.join(data_dir, dataset_name), "rb").readlines()
        df = pd.DataFrame({'text': text})
        dataset = Dataset.from_pandas(df)
    else:
        dataset: Dataset = load_dataset(
            dataset_name,
            data_dir=data_dir,
#             split=split,
            num_proc=N_CORES,
        )
    return dataset


def parse_markdown_code_block(text: str, ext: str = "verilog"):
    try:
        pattern = r"```.*?\n(.*?)```"
        # Use re.DOTALL to make '.' match also newlines
        match = re.search(pattern, text, re.DOTALL)
        if match is None:
            return text
        else:
            return match.group(1)
    except Exception:
        try:
            cleaned_output = text.strip()
            if f"```{ext}" in cleaned_output:
                _, cleaned_output = cleaned_output.split(f"```{ext}")
            if "```" in cleaned_output:
                cleaned_output, _ = cleaned_output.split("```")
            if cleaned_output.startswith(f"```{ext}"):
                cleaned_output = cleaned_output[len(f"```{ext}") :]
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```") :]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[: -len("```")]
            return cleaned_output.strip()
        except:
            logger.warning("Parse markdown code block failed")
            return ""


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def retry_with_exponential_backoff(
    errors: tuple,
    initial_delay: float = 30,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
):
    """Retry a function with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )
                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    # Sleep for the delay
                    time.sleep(delay)
                    # time.sleep(60)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


ERRORS = (
    Exception
)

try:
    OPENAI_CLIENT: openai.OpenAI | None = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL")
    )
except:
    OPENAI_CLIENT = None


@retry_with_exponential_backoff(ERRORS)
def chat_completions_with_backoff(*args, **kwargs):

    from llm_api import make_requests

    kwargs["engine"] = kwargs.pop("model")
    kwargs["prompts"] = kwargs.pop("messages")
    response = make_requests(*args, **kwargs)
    return response


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str='gpt-4') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    # encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def limit_string_to_tokens(string:str, token_count: int=2048, model: str='gpt-4') -> str:
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    num_tokens = len(encoded_string)
    if num_tokens > token_count:
       return encoding.decode(encoded_string[:token_count])
    else:
        return string

def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_fingerprint(*args: Any, hash_length: int = None) -> str:
    combined = "".join(map(str, args))
    content = hashlib.sha256(combined.encode()).hexdigest()
    if hash_length is not None:
        content = content[:hash_length]
    return content


"""
Adapted util functions from running verilog evaluations
"""
def valid_module(code: str):
    import importlib
    spec = importlib.util.find_spec('pyverilog')
    if spec is None:
        return False

    from pyverilog.vparser.parser import parse
    try:
        ast, directives = parse()
        return True
    except Exception:
        return False
    

def remove_module_header(code: str):
    st = code.find('module ')
    ed = code.find(');')
    if st != -1 and ed != -1:
        return code[ed+2:]
    else:
        return code
    
def post_process_completion(completion: str, remove_header: bool = False, rtllm: bool = False):
    raw_completion = completion

    if 'endmodule' not in completion:
        completion = completion + '\nendmodule'
    else:
        idx = completion.rindex('endmodule')
        completion = completion[:idx] + 'endmodule'

    # markdown code block
    if '```verilog' in raw_completion:
        completion = parse_markdown_code_block(completion)
    
    # do not remove module header if rtllm
    # we will not be adding module header for rtllm in check_correctness
    # this is for handling cases where multiple modules are required
    if (not rtllm) or completion.count("endmodule") == 1:
        pattern = r"\bmodule\b[\s\S]*?\bendmodule\b"
        matches = re.findall(pattern, completion)
        if matches and (valid_module(matches[0]) or remove_header):
            completion = remove_module_header(matches[0])

    return completion

def read_problems(evalset_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)