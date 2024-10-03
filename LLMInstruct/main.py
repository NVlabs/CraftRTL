# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import gzip
import json
import random
import pandas as pd
import os
import bdb
import logging
from datasets import Dataset
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Iterable, Dict, List

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from LLMInstruct.config import configure_logging
from LLMInstruct.utils import read_data, compute_fingerprint, read_jsonl
from LLMInstruct.task.factory import task_factory
from LLMInstruct.executor.verilog_executor import check_correctness
import warnings


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Args:
    seed_code_start_index: int
    # `seed_code_start_index` + `max_new_data` is the last-to-end seed code index
    max_new_data: int
    seed: int = field(default=976)

    temperature: float = field(default=0.0)
    engine: str = field(default="gpt35")
    model: str = field(default="gpt-3.5-turbo-1106") # this argument is deprecated. When calling nvcf engine will embed model name. Using this for tiktoken token counting.
    model_max_tokens: int = field(default=16384)
    max_new_tokens: int = field(default=1024)

    min_lines: int = field(default=32)
    max_lines: int = field(default=64)
    chunk_size: int = field(default=1000)

    dataset_name: str = field(default="bigcode/starcoderdata")
    data_dir: str = field(default="verilog")
    max_considered_data: int = field(default=150000)
    output_path: str = "./output"
    suffix: str = ""
    fewshot: str = ""
    error_report: str=""

    prompt_template: str = field(default="./prompt/solution.txt")
    task: str = field(default="code_generate")
    parallel: int = 1
    error_margin = 10
    persistent: int = 1
    shuffle: bool = True
    resume: str = ""
    input_key: str = "input"
    output_key: str = ""
    llm_filter: bool = True
    llm_reward: bool = False

    def fingerprint(self, prompt_template: str) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = (
            self.seed,
            self.temperature,
            self.engine,
            self.model_max_tokens,
            self.min_lines,
            self.max_lines,
            self.chunk_size,
            self.dataset_name,
            self.data_dir,
            self.max_considered_data,
            self.prompt_template,
            self.error_margin,
        )
        return compute_fingerprint(*args, hash_length=5)


def map_dataset(examples: dict, indices: List[int], args: Args) -> dict:
    random.seed(args.seed + indices[0])
    return {
        "seed": [content for content in examples[args.data_field]],
        "raw_index": indices,
    }


def read_dataset(args):
    split = (
        f"train[:{args.max_considered_data}]"
        if args.max_considered_data is not None
        else "train"
    )
    dataset = read_data(args.data_dir, args.dataset_name)

    # map and shuffle
    random.seed(args.seed)

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.map(lambda _, index: {"index": index}, with_indices=True)
    return dataset


def run(args, dataset, path: str, iterations=None):
    # assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a") as f_out:
        print("Saving to", path)
        cnt = 0
        task = task_factory(args)
        exists = set()

        print(f"Working with {len(dataset)} amount of data.")
        if args.resume:
            exists = set([i[args.input_key] for i in read_jsonl(str(path))])
            print(f"Loaded {len(exists)} data from {str(path)}!")

        if iterations is None:
            iterations = args.max_new_data
        with tqdm(total=iterations) as pbar:
            removed_index = set()
            index = 0
            while cnt < len(exists) and index < len(dataset):
                example = dataset[index]

                search_key = args.output_key

                if args.resume and example[search_key] in exists:
                    cnt += 1
                    pbar.update(1)
                removed_index.add(index)
                index += 1
            print(f"All data loaded for {path}, starting generation!")
            while cnt < iterations:
                for new_index in range(0, len(dataset)):
                    example = dataset[new_index]
                    if cnt < args.seed_code_start_index:
                        cnt += 1
                        pbar.update(1)
                        continue
                    
                    if new_index in removed_index:
                        continue
                    
                    for _ in range(args.persistent):
                        try:
                            data = task(example)
                            if data is not None:
                                f_out.write(json.dumps(data) + "\n")
                                cnt += 1
                                pbar.update(1)
                                if cnt % 16 == 0:
                                    f_out.flush()
                                break
                        except (KeyboardInterrupt, bdb.BdbQuit):
                            raise
                        except Exception:
                            logger.exception("")
                            continue
                break


def run_parallel(args, dataset):

    start_index = args.seed_code_start_index

    if args.parallel > 1:

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            batch_size = min(start_index + args.max_new_data, len(dataset)) // args.parallel
            for i in range(args.parallel):
                local_start_index = start_index + batch_size * i
                local_end_index = local_start_index + batch_size
                sub_dataset = dataset.select(range(local_start_index, local_end_index))
                path = Path(f"{experiment_naming_prefix(args)}.{i}.jsonl")
                future = executor.submit(run, args, sub_dataset, path, batch_size)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

    else:
        # Every run should produce the same data as long as the default params are not changed
        end_index = min(start_index + args.max_new_data, len(dataset))
        dataset = dataset.select(range(start_index, end_index))

        path = Path(f"{experiment_naming_prefix(args)}.jsonl")
        run(args, dataset, path)


def experiment_naming_prefix(args):
    prefix = f"{args.output_path}/{args.engine}_{args.dataset_name.rsplit('.', 1)[0]}_{args.max_new_data}"
    if args.suffix:
        prefix = f"{prefix}_{args.suffix}"
    return prefix


def main():
    args, *_ = cast(
        tuple[Args, ...], HfArgumentParser(Args).parse_args_into_dataclasses()
    )
    configure_logging()
    dataset = read_dataset(args)
    run_parallel(args, dataset)


if __name__ == "__main__":
    main()
