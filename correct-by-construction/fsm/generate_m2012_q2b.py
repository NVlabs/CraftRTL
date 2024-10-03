"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem 2012_q2b.

```
// Consider the state machine shown below:

// A (0) --1--> B
// A (0) --0--> A
// B (0) --1--> C
// B (0) --0--> D
// C (0) --1--> E
// C (0) --0--> D
// D (0) --1--> F
// D (0) --0--> A
// E (1) --1--> E
// E (1) --0--> D
// F (1) --1--> C
// F (1) --0--> D

// Assume that a one-hot code is used with the state assignment y[5:0] = 000001(A), 000010(B), 000100(C), 001000(D), 010000(E), 100000(F)

// Write a Verilog for the signal Y1, which is the input of state flip-flop y[1], for the signal Y3, which is the input of state flip-flop y[3]. Derive the Verilog by inspection assuming a one-hot encoding.
```
"""
from templates.m_2012_q2b import *
import random
import json

num_states = 6
input_signal_name = 'w'
output_signal_name = 'y'
state_names = None
out_file = f'm_2012_q2b_{num_states}.jsonl'

random.seed("1234")
n = 3

import_global(num_states, input_signal_name, output_signal_name)

deduplication = set([decontaminate])
print(f"Decontaminate on the following:\n```{decontaminate}```")

problems = []

for _ in range(n):
    try:
        index0 = random.choice([1,1,1,0,2])
        index1 = random.choice([3,3,3,4,5])
        index = [index0, index1]
        g, problem_statement, state_graph = generate_question(inverse_state_names=False, index=index)
        reset_state = g.nodes[0]['name'] 
        #print(state_graph)
        state_graph = sort_lines(state_graph)
        if state_graph not in deduplication:
            #print(state_graph)
            #print(problem_statement)
            reasoning = generate_reasoning_solution(g, reset_state, index=index)
            assert reasoning # filtering case where state A is unreachable, get_code_and_logic() will return None
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