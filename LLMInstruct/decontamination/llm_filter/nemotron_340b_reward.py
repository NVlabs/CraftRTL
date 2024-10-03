# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import os
from typing import Optional
from openai import OpenAI

from ..base import BaseFilter



class NemotronRewardFilter(BaseFilter):

    def __init__(
        self,
        args,
        prompt_template: str = "",
    ):
        self.prompt_template = prompt_template
        self.args = args
        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = os.environ['API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC'],
        )
        
    def parse(self, text: str):
        scores = {}
        for i in text.strip().split(','):
            key, value = i.split(':')
            scores[key] = float(value)
        return scores

    def validate(self, input_dict: dict) -> dict:

        # construct chat message and call
        messages = [
            {"role": "user", "content": input_dict['problem']},
            {"role": "assistant", "content": input_dict['solution']},
        ]
        completion = self.client.chat.completions.create(
            model="nvidia/nemotron-4-340b-reward",
            messages=messages,
        )
        output = completion.choices[0].message[0].content
        return self.parse(output)
