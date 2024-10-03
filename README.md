# CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair

This is an implementation for the data generation pipeline
described in the paper "[CraftRTL: High-quality Synthetic 
Data Generation for Verilog Code Models with Correct-by-Construction 
Non-Textual Representations and Targeted Code Repair](https://arxiv.org/abs/2409.12993)".


## Installation

Make sure to use python 3.10 or later:
```
$ conda create -n craftrtl python=3.10
$ conda activate craftrtl
```

Check out and install this repository:
```
$ git clone https://github.com/NVlabs/CraftRTL
$ cd CraftRTL
$ pip install -e .
```

Install [ICARUS Verilog](https://github.com/steveicarus/iverilog). This is required
for simulations when checking for self-consistency during error report generation:
```
$ git clone https://github.com/steveicarus/iverilog.git && cd iverilog \
        && git checkout v12-branch \
        && sh ./autoconf.sh && ./configure && make -j4\
        && make install
```

## Setting Up API Key for NVIDIA NIM
Before data generation, you need to set up access to models hosted through [NVIDIA NIM](https://build.nvidia.com/explore/discover).
Instructions for generating API key is [here](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#generate-an-api-key).
```
$ export API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC=$API_KEY
```

## Basic Synthetic Data Generation

We use a variety of prompt techniques to generate diverse data. You can also add your own method [here](LLMInstruct/readme.md).

### Self-Instruct
We provide an example of the [seed instructions](dataset/fewshot_seed.txt). Prompts for generating [instructions](prompt/concept-icl.txt) and [solutions](prompt/solution_reasoning.txt).
1. Generate instructions based on given seed instructions
```
script/generate_instruction_self.sh
```
2. Generate solutions for the previously generated instructions
```
script/generate_solution_self.sh
```

### Oss-Instruct
We provide an example of the extracted [open-source code snippets](dataset/the_stack_v2_cleaned.sample.jsonl). Prompts for generating [instructions](prompt/oss.txt) and [solutions](prompt/solution_reasoning_oss.txt).
1. Generate instructions based on given code snippets
```
script/generate_instruction_oss.sh
```
2. Generate solutions for the previously generated instructions
```
script/generate_solution_self.sh
```

### Docu-Instruct
We provide an example of the [wikipedia entities](dataset/wiki_entities.jsonl). Prompts for generating [instructions](prompt/concept-icl.txt) and [solutions](prompt/solution_reasoning.txt).
1. Generate instructions based on given wiki entities
```
script/generate_instruction_wiki.sh
```
2. Generate solutions for the previously generated instructions
```
script/generate_solution_wiki.sh
```

### Non-Textual Representations
Prompts for generating [instructions](prompt/kmap.txt) and [solutions](prompt/solution_reasoning.txt).
1. Generate instructions to include non-textual representations
```
script/generate_instruction_kmap.sh
```
2. Generate solutions for the previously generated instructions
```
script/generate_solution_kmap.sh
```

## Correct-by-Construction Data
We target non-textual representation problems by ensuring correctness through correct-by-construction. 
Please refer to specific scripts under [correct-by-construction](correct-by-construction) for details.

## Code Repair Data

1. Generate error report. You can refer to examples on input [error code](code_repair_examples/exp/testrun/checkpoint-step=1).
```
script/generate_error_report.sh
```

2. Generate code repair data. We provide an example on the [error report](code_repair_examples/error_report.sample.jsonl).
```
script/generate_repair_oss.sh
```

## License
Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License (1-Way Commercial) License. Click [here](LICENSE) to view a copy of this license.

## Citation

Please cite using the following bibtex entry:

```
@misc{liu2024craftrtlhighqualitysyntheticdata,
      title={CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair}, 
      author={Mingjie Liu and Yun-Da Tsai and Wenfei Zhou and Haoxing Ren},
      year={2024},
      eprint={2409.12993},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2409.12993}, 
}
```
