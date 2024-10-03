#!/bin/bash

num_instructions_to_generate=1
data_dir="dataset"
engine="nvcf-nemotron-4-340b-instruct"
dataset_name="the_stack_v2_cleaned.sample.jsonl"
prompt="./prompt/seed-icl.txt"
task="instruct_generate_large"
output_path="./output/instructions/self"
batch_size=1
fewshot="./dataset/fewshot_seed.txt"


python LLMInstruct/main.py \
    --seed_code_start_index 0 \
    --parallel ${batch_size} \
    --max_new_data ${num_instructions_to_generate} \
    --data_dir ${data_dir} \
    --dataset_name ${dataset_name} \
    --engine ${engine} \
    --prompt ${prompt} \
    --task ${task} \
    --max_new_tokens 2048 \
    --output_path ${output_path} \
    --temperature 0.1 \
    --fewshot ${fewshot}
    
    