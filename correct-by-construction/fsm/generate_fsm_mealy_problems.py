"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem ece241_2014_q5b.
The original problem only has 2 states, we sample for 4 states. No need to decontaminate.

The following diagram is a Mealy machine implementation of the 2's complementer. Implement in Verilog using one-hot encoding. Resets into state A and reset is asynchronous active-high.

// A --x=0 (z=0)--> A
// A --x=1 (z=1)--> B
// B --x=0 (z=1)--> B
// B --x=1 (z=0)--> B

Note: we also generalize the problem statement to `The following diagram is a Mealy machine.` 
We also adjust the problem statement to the following during evaluation:
```
The following diagram is a Mealy machine. Implement in Verilog using one-hot encoding. Resets into state A and reset is asynchronous active-high.
```
"""

from templates.fsm_mealy import *
import random
import json

num_states = 4
input_signal_name = 'in'#['in0','in1','in2', 'in3']
output_signal_name = 'z'
state_names = None

n = 2
out_file = f'fsm_mealy.jsonl'

import_global(num_states, input_signal_name, output_signal_name, state_names)

deduplication = set()
problems = []

random.seed("1234")

for _ in range(n):

    try:
        inverse_state_names = False
        g, problem_statement, state_graph = generate_question(inverse_state_names=inverse_state_names)
        reset_state = g.nodes[0]['name']
        if state_graph not in deduplication:
            #print(problem_statement)
            reasoning = generate_reasoning_solution(g, reset_state)
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