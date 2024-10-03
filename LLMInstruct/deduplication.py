# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from LLMInstruct.decontamination.similarity_filter import JaccardFilter
from LLMInstruct.utils import read_data, write_jsonl
from tqdm import tqdm

def validate_data(filter, data, keywords):
    for key in keywords:
        if not filter.validate(data[key]):
            return False

    for key in keywords:
        filter.add(data[key])
    return True


def dedupe_generated(data_dir, data_name, benchmark_dir="./dataset/benchmark", benchmark_name="verilogeval-manual.jsonl.gz", keywords=['text']):
    benchmark = read_data(benchmark_dir, benchmark_name)
    dataset = read_data(data_dir, data_name)
    jaccard_filter = JaccardFilter(instructions=benchmark["prompt"]+benchmark["canonical_solution"])
    dedupe_results = []
    for data in tqdm(dataset):
        if validate_data(jaccard_filter, data, keywords):
            dedupe_results.append(data)
    return dedupe_results


# modify the following
data_dir = ""
data_name = ""
output_file = ""
keywords = ["input", "output"]

data = dedupe_generated(data_dir, data_name, keywords=keywords)
write_jsonl(output_file, data)
