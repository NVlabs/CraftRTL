# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from typing import Optional
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List
from langchain.pydantic_v1 import BaseModel, Field

from ..base import BaseFilter
from LLMInstruct.utils import num_tokens_from_string, chat_completions_with_backoff

LLM_FILTER_PROMPT = """Check if the given Verilog module is a valid solution to the problem. The output should be in
“True” or “False” and be enclosed within <VALID> </VALID> tags and the explanation in <REASON></REASON> tags.
Now check the following:
<PROBLEM>
{problem}
<PROBLEM>
<SOLUTION>
{solution}
</SOLUTION>
"""


class LLMFilter(BaseFilter):

    def __init__(
        self,
        args,
    ):
        self.prompt_template = LLM_FILTER_PROMPT
        self.args = args
        
    def parse(self, text: str, tag: str):
        head = f"<{tag}>"
        tail = f"</{tag}>"
        if head in text and tail in text:
            st = text.find(head)
            ed = text.find(tail)
            return text[st+len(head): ed].strip()

    def validate(self, input_dict: dict, tag: str, target: str) -> bool:

        # construct chat message and call
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.prompt_template.format(**input_dict)},
        ]
        response = chat_completions_with_backoff(
            model=self.args.engine,
            messages=messages,
            max_tokens=4096,
            n=1,
            temperature=self.args.temperature,
            seed=self.args.seed,
        )
        # postprocess
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            return False
        output = choice["message"]["content"]
        if "mixtral" in self.args.engine.lower():
            # weird decoding problem
            output = output.replace("\_", "_")
            
        valid = self.parse(output, tag)
        reason = self.parse(output, "REASON")
        return (valid == target), reason
