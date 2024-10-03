"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem fsm_onehot.

```
Given the follow state machine with 1 input and 2 outputs (the outputs are given as "(out1, out2)"):

// S0 (0, 0) --0--> S0
// S0 (0, 0) --1--> S1
// S1 (0, 0) --0--> S0
// S1 (0, 0) --1--> S2
// S2 (0, 0) --0--> S0
// S2 (0, 0) --1--> S3
// S3 (0, 0) --0--> S0
// S3 (0, 0) --1--> S4
// S4 (0, 0) --0--> S0
// S4 (0, 0) --1--> S5
// S5 (0, 0) --0--> S8
// S5 (0, 0) --1--> S6
// S6 (0, 0) --0--> S9
// S6 (0, 0) --1--> S7
// S7 (0, 1) --0--> S0
// S7 (0, 1) --1--> S7
// S8 (1, 0) --0--> S0
// S8 (1, 0) --1--> S1
// S9 (1, 1) --0--> S0
// S9 (1, 1) --1--> S1

// Suppose this state machine uses one-hot encoding, where state[0] through state[9] correspond to the states S0 though S9, respectively. The outputs are zero unless otherwise specified.

// Write Verilog implementing the state transition logic and output logic portions of the state machine (but not the state flip-flops). You are given the current state in state[9:0] and must produce next_state[9:0] and the two outputs. Derive the logic equations by inspection assuming a one-hot encoding.
```
"""

from templates.fsm_onehot import *
import random
import json

num_states = 10
input_signal_name = 'in'
output_signal_name = 'out1'
state_names = [f"S{i}" for i in range(10)]

n = 2
out_file = f'fsm_onehot_{num_states}.jsonl'

import_global(num_states, input_signal_name, output_signal_name)

random.seed("1234")

deduplication = set([decontaminate])
print(f"Decontaminate on the following:\n```{decontaminate}```")
problems = []

for _ in range(n):

    try:
        g, problem_statement, state_graph = generate_question(inverse_state_names=False, state_names=state_names)
        reset_state = g.nodes[0]['name']
        if state_graph not in deduplication:
            #print(problem_statement)
            reasoning = generate_reasoning_solution(g, reset_state) 
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