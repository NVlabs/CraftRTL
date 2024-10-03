# run the following without self-consistency checks
python LLMInstruct/error_report.py \
    --output ./examples/output \
    --temp ./examples/tmp \
    --cache_path ./examples/cache \
    --exp_path ./code_repair_examples/exp \
    --num_samples 5 \
    --workers 32 

# run the following with self-consistency checks
# will likely need to increase --iters 

#python LLMInstruct/error_report.py \
#    --output ./examples/output \
#    --temp ./examples/tmp \
#    --cache_path ./examples/cache \
#    --exp_path ./code_repair_examples/exp \
#    --num_samples 5 \
#    --workers 32 \
#    --iters 1 \
#    --self_consist