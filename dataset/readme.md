# dataset
* fewshot_seed.txt: seed examples for self-instruct. We use problem descriptions from VerilogEval-Human as example. Should be replaced with LLM generated problem description.
* fewshot_entities.txt: seed examples for docu-instruct
* the_stack_v2_cleaned.sample.jsonl: examples of cleaned up verilog module from the_stack_v2. We use canonical solution from VerilogEval-Human as example. Replace with processed contents from the_stack_v2.

### Benchmark
* verilogeval-mahcine.jsonl.gz: VerilogEval-Machine
* verilogeval-manual.jsonl.gz: VerilogEval-Human
* rtllm.jsonl: RTLLM-v1.1 fixed version
