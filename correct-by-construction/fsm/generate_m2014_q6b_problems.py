"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

From problem m2014_q6b:
```
Consider the state machine shown below:

// A (0) --0--> B
// A (0) --1--> A
// B (0) --0--> C
// B (0) --1--> D
// C (0) --0--> E
// C (0) --1--> D
// D (0) --0--> F
// D (0) --1--> A
// E (1) --0--> E
// E (1) --1--> D
// F (1) --0--> C
// F (1) --1--> D

// Assume that you want to Implement the FSM using three flip-flops and state codes y[3:1] = 000, 001, ..., 101 for states A, B, ..., F, respectively. Implement just the next-state logic for y[2] in Verilog. The output Y2 is y[2].
```
"""
from templates.m2014_q6b import *
import random
import json

num_states = 6
input_signal_name = 'w'
output_signal_name = 'y'
state_names = None

random.seed("1234")
n = 2
out_file = f'm2014_q6b_{num_states}.jsonl'

import_global(num_states, input_signal_name, output_signal_name)

deduplication = set([decontaminate])
print(f"Decontaminate on the following:\n```{decontaminate}```")
problems = []


for _ in range(n):

    try:
        index = random.choice([1,2,3,2,2])
        g, problem_statement, state_graph = generate_question(inverse_state_names=False, index=index)
        reset_state = g.nodes[0]['name']
        state_graph = sort_lines(state_graph)
        if state_graph not in deduplication:
            #print(problem_statement)
            reasoning = generate_reasoning_solution(g, reset_state, index=index)
            #print(reasoning)
            deduplication.add(state_graph)
            problems.append( {'input': problem_statement, 'output': reasoning} )
        else:
            print('deduplicated')
    except:
        continue

with open(out_file, 'w') as f:
    for x in problems:
        f.write(json.dumps(x))
        f.write('\n')