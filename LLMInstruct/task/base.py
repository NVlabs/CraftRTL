# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

from LLMInstruct.utils import num_tokens_from_string, chat_completions_with_backoff


class Task(ABC):

    @abstractmethod
    def construct_prompt(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def decontaminate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class BaseTask(Task):

    system_prompt = "You are a helpful assistant."

    def __init__(self, config, *args, **kwargs):
        self.args = config

    def construct_prompt(self, example: dict) -> str:
        prompt_template = Path(self.args.prompt_template).read_text()
        prompt = prompt_template.format()
        return prompt

    def generate(self, prompt: str, check_stop: bool = True) -> Optional[str]:

        # Make sure the generation is within the context size of the model
        max_new_tokens = min(
            self.args.max_new_tokens,
            self.args.model_max_tokens - num_tokens_from_string(prompt, self.args.model)
            # error margin (e.g., due to conversation tokens)
            - self.args.error_margin,
        )

        # construct chat message and call
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = chat_completions_with_backoff(
            model=self.args.engine,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,
            temperature=self.args.temperature,
            seed=self.args.seed,
        )

        # postprocess
        if response is None:
            return
        choice = response["choices"][0]
        if check_stop and choice["finish_reason"] != "stop":
            print(f'finish reason is not stop: {choice["finish_reason"]}')
            return
        output = choice["message"]["content"]
        if "mixtral" in self.args.engine.lower():
            # weird decoding problem
            output = output.replace("\_", "_")
        return output

    def parse(self, result: str) -> Optional[str]:
        return result

    def decontaminate(self, result: str) -> Optional[str]:
        if result is None or len(result) == 0:
            return
        return result

    def evaluate(self, response_text: str) -> dict:
        return {}

    def __call__(self, example: dict) -> Optional[str]:
        prompt = self.construct_prompt(example)
        raw_result: str = self.generate(prompt)
        result: str = self.parse(raw_result)
        result: str = self.decontaminate(result)
        if result is None:
            return

        eval_result: dict = self.evaluate(result)
        data = dict(
            raw_index=example["raw_index"],
            index=example["index"],
            seed=example["seed"],
            fingerprint=response["system_fingerprint"],
            raw=choice["message"]["content"],
        )
        data.update(eval_result)
        return data
