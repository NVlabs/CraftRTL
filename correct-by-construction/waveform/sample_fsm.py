"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

We generate the verilog code re-using code from ./../fsm
Generation logic utils_fsm.py similar to ./../fsm/utils.py
We generate code similar to problem solution of fsm1s. See fsm1s.py for details.

Simulation and post-process in simulation_fsm.py
Testbench: ./test/test.v

We do not perform deconatmination since non of the benchmark problems are related.

"""

from fsm1s import *
import json
random.seed('1234')
from tqdm import tqdm

from simulation import obtain_waveform

num_states = 6
input_signal_name = 'in'
output_signal_name = 'out'

num_samples = 20 # put this large as test input might not reach all states or generate meaningful out
out_file = f'fsm1s_{num_states}.jsonl'

import_global(num_states, input_signal_name, output_signal_name)

with open('./test/test.v', 'r') as f:
    test = f.read()

deduplication = set()
problems = []

for _ in  tqdm(range(num_samples)):
    try:
        g, problem_statement, state_graph = generate_question(inverse_state_names=False)
        reset_state = g.nodes[0]['name']
        if state_graph not in deduplication:
            #print(problem_statement)
            reasoning, problem_statement = generate_reasoning_solution(g, reset_state)
            assert reasoning # This would filter non-meaningful output. See fsm1s.py contain_meaningful_output().
            #print(reasoning)
            deduplication.add(state_graph)
            problems.append( {'input': problem_statement, 'output': reasoning} )
    except:
        continue


with open('waveform_sequential.jsonl', 'w') as f:
    for x in problems:
        f.write(json.dumps(x))
        f.write('\n')
