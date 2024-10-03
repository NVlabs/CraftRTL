#!/bin/bash


num_instructions_to_generate=1
data_dir="./output/instructions/self"
engine="nvcf-nemotron-4-340b-instruct"
# modify the following if name is different
dataset_name="nvcf-nemotron-4-340b-instruct_the_stack_v2_cleaned.sample_1.jsonl"
prompt="./prompt/solution_reasoning.txt"
task="code_reason_generate"
output_path="./output/sdg_data/self"
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




# num_instructions_to_generate=100000
# data_dir="./output/seed_mixtral_8x22b"
# engine="mixtral-nvcf2"
# dataset_name="mixtral-nvcf-22b_github_sv3_100000_seed_all.deduped.jsonl"
# prompt="./prompt/solution_reasoning.txt"


# python LLMInstruct/main.py \
#   --seed_code_start_index 0 \
#   --max_new_data ${num_instructions_to_generate} \
#   --data_dir ${data_dir} \
#   --dataset_name ${dataset_name} \
#   --engine ${engine} \
#   --prompt ${prompt} \
#   --parallel 4 \
#   --output_path ./output/seed_mixtral_8x22b \
#   --task code_reason_generate
