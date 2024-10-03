#!/bin/bash

num_instructions_to_generate=2
data_dir="./output/instructions/oss"
engine="nvcf-nemotron-4-340b-instruct"
# modify the following if name is different ./output/instructions/oss/nvcf-nemotron-4-340b-instruct_the_stack_v2_cleaned.sample_2.jsonl
dataset_name="nvcf-nemotron-4-340b-instruct_the_stack_v2_cleaned.sample_2.jsonl"
prompt="./prompt/solution_reasoning_oss.txt"
task="code_reason_oss"
output_path="./output/sdg_data/oss"
batch_size=1


python LLMInstruct/main.py \
  --seed_code_start_index 0 \
  --max_new_data ${num_instructions_to_generate} \
  --parallel ${batch_size} \
  --data_dir ${data_dir} \
  --dataset_name ${dataset_name} \
  --engine ${engine} \
  --prompt ${prompt} \
  --task ${task} \
  --max_new_tokens 2048 \
  --temperature 0.01 \
  --output_path ${output_path} \
  --llm_filter True
    