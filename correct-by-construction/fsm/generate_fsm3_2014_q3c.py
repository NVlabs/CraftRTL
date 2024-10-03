"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem fsm3onehot.

```
The following is the state transition table for a Moore state machine with one input, one output, and four states. Use the following one-hot state encoding: A=4'b0001, B=4'b0010, C=4'b0100, D=4'b1000. Derive state transition and output logic equations by inspection assuming a one-hot encoding. Implement only the state transition logic and output logic (the combinational logic portion) for this state machine. 
// State | Next state in=0, Next state in=1 | Output
// A | A, B | 0
// B | C, B | 0
// C | A, D | 0
// D | C, B | 1
```
"""
from templates.fsm3_2014_q3c import *
import random
import json

num_states = 5
input_signal_name = 'x'
output_signal_name = 'z'
state_names = None

random.seed("1234")
n = 2
out_file = f'fsm3_2014_q3c_{num_states}.jsonl'

import_global(num_states, input_signal_name, output_signal_name)

deduplication = set([decontaminate])
print(f"Decontaminate on the following:\n```{decontaminate}```")

problems = []

for _ in range(n):

    try:
        index = random.choice([0,1,2,0,0])
        g, problem_statement, transition_table = generate_question(inverse_state_names=False, index=index)
        reset_state = g.nodes[0]['name']
        if transition_table not in deduplication:
            #print(problem_statement)
            reasoning = generate_reasoning_solution(g, reset_state, index=index)
            #print(reasoning)
            deduplication.add(transition_table)
            problems.append( {'input': problem_statement, 'output': reasoning} )
        else:
            print('deduplicated')
    except:
        continue

with open(out_file, 'w') as f:
    for x in problems:
        f.write(json.dumps(x))
        f.write('\n')