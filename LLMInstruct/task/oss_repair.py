# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import logging
from abc import ABC
from pathlib import Path
from typing import Optional
from datasets import Dataset
from LLMInstruct.executor.verilog_executor import check_correctness
from LLMInstruct.utils import parse_markdown_code_block, read_jsonl
from LLMInstruct.task.base import BaseTask
from LLMInstruct.sampler.fewshot import FewShotSampler
from LLMInstruct.decontamination.similarity_filter import JaccardFilter
import pandas as pd

logger = logging.getLogger(__name__)

class OSSRepairTask(BaseTask):

    system_prompt = "You are exceptionally skilled at generating high-quality Verilog problem and specification description from the given code."

    def __init__(self, config):
        self.args = config
        self.prompt_template = Path(self.args.prompt_template).read_text()

        benchmark_dataset = pd.concat([
             pd.DataFrame(read_jsonl('./dataset/benchmark/verilogeval-manual.jsonl.gz')),
             pd.DataFrame(read_jsonl('./dataset/benchmark/verilogeval-machine.jsonl.gz')),
             pd.DataFrame(read_jsonl('./dataset/benchmark/rtllm.jsonl')),
        ]).dropna()

        self.filter = JaccardFilter(
            instructions=benchmark_dataset.detail_description.values.tolist()+ benchmark_dataset.canonical_solution.values.tolist()
        )
        self.fewshot_sampler = FewShotSampler(self.args.error_report)

    def construct_prompt(self, example: dict, icl_sample: dict):
        try:
            prompt = self.prompt_template.format(
                code=example['input'],
                icl_correct=icl_sample["correct"],
                icl_reason=icl_sample['reason'],
                icl_error=icl_sample['error'],
            )
            return prompt
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


        icl_sample = self.fewshot_sampler.sample()
        prompt = self.construct_prompt(example, icl_sample)

        raw_result = self.generate(prompt)
        if raw_result is None:
            return
        result = self.parse(raw_result)
        if result is None:
            return
        result = self.decontaminate(result)
        if result is None:
            return

        # logging.info(f"[Problem]\n{result}\n\n")
        data = dict(
            index=example["index"],
            input=result,
            output=example['input'],
            error=icl_sample['error'],
            correct=icl_sample['correct'],
            reason=icl_sample['reason'],
            task_id=icl_sample['task_id'],

        )
        return data
