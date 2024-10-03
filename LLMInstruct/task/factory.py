# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from .base import BaseTask
from .instruct_gen_large import InstructGenLargeTask
from .wiki_instruct_gen import WikiInstructGenTask
from .oss_gen_large import OSSGenLargeTask
from .oss_repair import OSSRepairTask
from .code_reason_gen import CodeReasonGenTask
from .code_reason_oss_gen import CodeReasonOSSTask


def task_factory(config, *args, **kwargs) -> BaseTask:
    if config.task == "code_reason_generate":
        return CodeReasonGenTask(config, *args, **kwargs)
    elif config.task == "code_reason_oss":
        return CodeReasonOSSTask(config, *args, **kwargs)
    elif config.task == "instruct_generate_wiki":
        return WikiInstructGenTask(config, *args, **kwargs)
    elif config.task == "instruct_generate_large":
        return InstructGenLargeTask(config, *args, **kwargs)
    elif config.task == "oss_generate_large":
        return OSSGenLargeTask(config, *args, **kwargs)
    elif config.task == "oss_repair":
        return OSSRepairTask(config, *args, **kwargs)
    else:
        return BaseTask(config, *args, **kwargs)
