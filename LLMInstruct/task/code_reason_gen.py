# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import logging
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Optional
from LLMInstruct.executor.verilog_executor import check_correctness
from LLMInstruct.utils import parse_markdown_code_block, read_jsonl
from LLMInstruct.task.base import BaseTask
from LLMInstruct.decontamination.llm_filter.llm_filter import LLMFilter
from LLMInstruct.decontamination.llm_filter.nemotron_340b_reward import NemotronRewardFilter
from LLMInstruct.decontamination.similarity_filter import JaccardFilter


logger = logging.getLogger(__name__)

class CodeReasonGenTask(BaseTask):

    system_prompt = "You are exceptionally skilled at generating high-quality Verilog code and offering precise solutions to the given problem."

    def __init__(self, config):
        self.args = config
        self.prompt_template = Path(self.args.prompt_template).read_text()
        
        benchmark_dataset = pd.concat([
             pd.DataFrame(read_jsonl('./dataset/benchmark/verilogeval-manual.jsonl.gz')),
             pd.DataFrame(read_jsonl('./dataset/benchmark/verilogeval-machine.jsonl.gz')),
             pd.DataFrame(read_jsonl('./dataset/benchmark/rtllm.jsonl')),
        ]).dropna()

        self.similarity_filter = JaccardFilter(
            instructions=benchmark_dataset.detail_description.values.tolist()+ benchmark_dataset.canonical_solution.values.tolist()
        )
        self.llm_filter = None
        self.nemotron_reward = None

        if self.args.llm_filter:
            self.llm_filter = LLMFilter(config)
        if self.args.llm_reward:
            self.nemotron_reward = NemotronRewardFilter(config)
        else:
            self.nemotron_reward = None

    def construct_prompt(self, example: dict):
        prompt = self.prompt_template.format(problem=example["problem"])
        return prompt

    def parse(self, response_text: str) -> Optional[str]:
        
        solution = None
        reasoning = None
        
        if "<SOLUTION>" and "</SOLUTION>" in response_text:
            st = response_text.find("<SOLUTION>") + len("<SOLUTION>")
            ed = response_text.find("</SOLUTION>")
            if st is not None and ed is not None:
                solution = response_text[st:ed].strip()
        elif "<SOLUTION>" and "</SOLUT" in response_text:  # sometimes happens
            st = response_text.find("<SOLUTION>") + len("<SOLUTION>")
            ed = response_text.find("</SOLUT")
            if st is not None and ed is not None:
                solution = response_text[st:ed].strip()
                
        if "<REASON>" and "</REASON>" in response_text:
            st = response_text.find("<REASON>") + len("<REASON>")
            ed = response_text.find("</REASON>")
            if st is not None and ed is not None:
                reasoning = response_text[st:ed].strip()
        
        return solution, reasoning

    def evaluate(self, solution: str, problem: str) -> dict:
        iverilog_result = check_correctness(solution, 30, compile_only="iverilog")
        scores = dict(iverilog_compiler_passed=iverilog_result["passed"], 
                        iverilog_compiler_log=iverilog_result["feedback"]["compiler_log"])
        
        # Results did not pass syntax check, no need to use llm to filter
        if not iverilog_result["passed"]:
            return scores

        if self.llm_filter:
            llm_verification, llm_reason = self.llm_filter.validate(
                {'problem': problem, 'solution': solution},
                tag="VALID",
                target="True"
            )
            scores.update(dict(
                llm_verification=llm_verification,
                llm_reason=llm_reason
            ))

        if self.nemotron_reward:
            nemotron_reward = self.nemotron_reward.validate({'problem': problem, 'solution': solution})
            scores.update(nemotron_reward)

        return scores
    
    def decontaminate(self, result: str):
        if self.similarity_filter.validate(result):
            self.similarity_filter.add(result)
            return result
        
    def llm_verify(self, problem: str, solution: str):
        return self.llm_filter.validate(
            {
                'problem': problem,
                'solution': solution
            },
            tag="VALID",
            target="True"
        )

    def __call__(self, example: dict):
        prompt = self.construct_prompt(example)
        raw_result = self.generate(prompt)
        
        if raw_result is None:
            return
        solution, reasoning = self.parse(raw_result)
        if solution is None:
            return
        result = self.decontaminate(solution)
        if result is None:
            return

        eval_result = self.evaluate(result, example['problem'])

        data = dict(
            index=example["index"],
            input=example['problem'],
            output=result,
            reasoning=reasoning,
            raw=raw_result,
        )
        data.update(eval_result)
        return data
