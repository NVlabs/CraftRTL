# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import logging
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Optional
from datasets import Dataset
from LLMInstruct.executor.verilog_executor import check_correctness
from LLMInstruct.utils import parse_markdown_code_block, read_jsonl
from LLMInstruct.task.base import BaseTask
from LLMInstruct.sampler.fewshot import FewShotSampler
from LLMInstruct.decontamination.similarity_filter import JaccardFilter


logger = logging.getLogger(__name__)


class MyFewShotSampler(FewShotSampler):
    def _filter(self, dataset: Dataset):
        after_filter_dataset = self.dataset.filter(lambda example: (len(example["detail_description"]) < 128 and len(example['canonical_solution']) < 256))
        print(f"FewShotSampler filtering {len(after_filter_dataset)}/{len(dataset)} after before.")
        return after_filter_dataset


class InstructGenLargeTask(BaseTask):

    system_prompt = "You are exceptionally skilled at generating high-quality Verilog problem and spec description from the given code."

    def __init__(self, config):
        self.args = config
        self.fewshot_sampler = FewShotSampler(self.args.fewshot)
        self.prompt_template = Path(self.args.prompt_template).read_text()

        benchmark_dataset = pd.concat([
            pd.DataFrame(read_jsonl('./dataset/benchmark/verilogeval-manual.jsonl.gz')),
            pd.DataFrame(read_jsonl('./dataset/benchmark/verilogeval-machine.jsonl.gz')),
            pd.DataFrame(read_jsonl('./dataset/benchmark/rtllm.jsonl')),
        ])
        self.filter = JaccardFilter(instructions=benchmark_dataset.detail_description.values.tolist())

    def construct_prompt(self, example: dict):
        try:
            icl_sample = self.fewshot_sampler.sample()
            icl_sample['text'] = str(icl_sample['text'])
            prompt = self.prompt_template.format(
                icl_problem=icl_sample["text"]
            )
            return prompt, icl_sample['text']
        except KeyError as e:
            logger.error(example.keys())
            raise

    def parse(self, response_text: str) -> Optional[str]:
        if "<PROBLEM>"  in response_text and "</PROBLEM>" in response_text:
            st = response_text.find("<PROBLEM>") + len("<PROBLEM>")
            ed = response_text.find("</PROBLEM>")
            if st is None or ed is None:
                return None
            return response_text[st:ed].strip()
        elif "<PROBLEM>" and "</PROBL" in response_text:  # sometimes happens
            st = response_text.find("<PROBLEM>") + len("<PROBLEM>")
            ed = response_text.find("</PROBL")
            if st is None or ed is None:
                return None
            return response_text[st:ed].strip()
        else:
            return parse_markdown_code_block(response_text)
        
    def decontaminate(self, result: str):
        if self.filter.validate(result):
            self.filter.add(result)
            return result

    def __call__(self, example: dict):
        prompt, icl_sample = self.construct_prompt(example)

        raw_result = self.generate(prompt)
        if raw_result is None:
            return
        result = self.parse(raw_result)
        if result is None:
            return
        result = self.decontaminate(result)
        if result is None:
            return

        data = dict(
            index=example["index"],
            seed=icl_sample,
            problem=result,
            raw=raw_result,
        )
        return data
