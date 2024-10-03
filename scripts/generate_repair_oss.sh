#!/bin/bash


num_instructions_to_generate=3
data_dir="./dataset"
engine="nvcf-nemotron-4-340b-instruct"
dataset_name="the_stack_v2_cleaned.sample.jsonl"
prompt="./prompt/oss_repair4.txt"
task="oss_repair"
output_path="./output/repair"
fewshot="./code_repair_examples/error_report.sample.jsonl"


python LLMInstruct/main.py \
    --seed_code_start_index 0 \
    --parallel 1 \
    --max_new_data ${num_instructions_to_generate} \
    --data_dir ${data_dir} \
    --dataset_name ${dataset_name} \
    --engine ${engine} \
    --prompt ${prompt} \
    --task ${task} \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --output_path ${output_path} \
    --error_report ${fewshot} 
    
    