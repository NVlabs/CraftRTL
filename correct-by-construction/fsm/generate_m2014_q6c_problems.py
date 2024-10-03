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

// Resets into state A. For this part, assume that a one-hot code is used with the state assignment y[6:1] = 000001, 000010, 000100, 001000, 010000, 100000 for states A, B,..., F, respectively.

// Write Verilog for the next-state signals Y2 and Y4 corresponding to signal y[2] and y[4]. Derive the logic equations by inspection assuming a one-hot encoding.
```
"""

from templates.m2014_q6c import *
import random
import json

num_states = 6
input_signal_name = 'w'
output_signal_name = 'y'
state_names = None

n = 3
out_file = f'm_2014_q6c_{num_states}.jsonl'

import_global(num_states, input_signal_name, output_signal_name)

random.seed("1234")
deduplication = set([decontaminate])
print(f"Decontaminate on the following:\n```{decontaminate}```")
problems = []

for _ in range(n):

    try:
        index0 = random.choice([1,2,2,2,3])
        index1 = random.choice([4,4,4,5,6])
        index = [index0, index1]
        g, problem_statement, state_graph = generate_question(inverse_state_names=False, index=index)
        reset_state = g.nodes[0]['name']
        state_graph = sort_lines(state_graph)
        if state_graph not in deduplication:
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