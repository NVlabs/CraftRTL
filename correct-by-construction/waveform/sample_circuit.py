"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

We generate the verilog code re-using code from ./../boolean_logic
Generation logic utils.py similar to ./../boolean_logic/utils.py

Simulation and post-process in simulation.py
Testbench: ./test/test.v

Since we sample for 4 variables, we decontaminate with the following benchmark problems.

circuit2:
    min_terms = 0000, 0011, 0101, 0110, 1001, 1010, 1100, 1111
    min_terms = [0, 3, 5, 6, 9, 10, 12, 15]

circuit3: 
    min_terms = 0101, 0110, 0111, 1001, 1010, 1011, 1101, 1110, 1111
    min_terms = [5, 6, 7, 9, 10, 11, 13, 14, 15]

circuit4:
    min_terms = 0010, 0011, 0100, 0101, 0110, 0111, 1010, 1011, 1100, 1101, 1110, 1111
    min_terms = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]

"""

from utils import *
import json
random.seed('1234')
num_samples=2
all_data_samples = []
sample_no_care = False # we don't sample don't cares for this
from sympy import symbols
from tqdm import tqdm

from simulation import obtain_waveform

problem = """This is a combinational circuit. Read the simulation waveforms to determine what the circuit does, then implement it.

{waveform}

module top_module(
\tinput a, 
\tinput b,
\tinput c,
\tinput d,
\toutput q
);"""

code = """
module top_module(
\tinput a, 
\tinput b,
\tinput c,
\tinput d,
\toutput q
);

\tassign q = {logic_equation};
endmodule
"""

solution = """The input variables are: {variables}.
Based on the simulation waveform, I can transform in to the following truth table:
{truthtable}

The minterms (when output is 1) are:
{minterms}
This corresponds to the following minterms logic:
`{logic_equation}`

Finally, based on the above logic equation, I can now write the Verilog code:
```
module top_module(
\tinput a, 
\tinput b,
\tinput c,
\tinput d,
\toutput q
);

\tassign q = {logic_equation};
endmodule
```
"""

circuit2_min_terms = frozenset([0, 3, 5, 6, 9, 10, 12, 15])
circuit3_min_terms = frozenset([5, 6, 7, 9, 10, 11, 13, 14, 15])
circuit4_min_terms = frozenset([2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15])
benchmark_min_terms = set([circuit2_min_terms, circuit3_min_terms, circuit4_min_terms])


with open('./test/test.v', 'r') as f:
    test = f.read()

for _ in  tqdm(range(num_samples)):
    min_terms, no_cares = random_sample_minterm(4, False) 

    if frozenset(min_terms) in benchmark_min_terms:
        print("Min terms in benchmark.")
        continue

    symbol = [symbols('a'), symbols('b'), symbols('c'), symbols('d')]
    #min_terms = list(min_term)
    no_cares = []
    sop_style = 1 #random.choice([0,1])
    permute = 0 #random.choice([0,1])
    no_care_symbol = ['d'] #random.choice(['x', 'X', 'd'])
    #print(symbol)
    #print(min_terms)
    # generate terms
    min_terms_string, min_terms_symbol = convert_min_term_sop(min_terms, symbol, sop_style)
    min_terms_symbol = str(SOPform(symbol, min_terms, no_cares))
    #print(min_terms_symbol)
    no_cares_string, no_cares_symbol = None, None
    #print(min_terms_string)

    # generate truthtable and karnauph maps
    truthtable = print_table_minterms(min_terms, no_cares, symbol, no_care_symbol)
    kmap = print_karnaugh_map(min_terms, no_cares, symbol, no_care_symbol, permute, comment=True)

    try:
        cur_test = test + "\n" + code.format(logic_equation=min_terms_string)
        waveform = obtain_waveform(cur_test)[0]
        assert waveform is not None
    except:
        continue

    input_ = problem.format(waveform=waveform)
    output_ = solution.format(kmap=kmap, truthtable=truthtable, minterms=convert_min_terms(min_terms_string, min_terms, 4), logic_equation=min_terms_string, variables=[str(x) for x in symbol])

    sample = {'variables': [str(x) for x in symbol], 
              'min_terms': min_terms,
              'no_cares': no_cares,
              'min_terms_string': min_terms_string,
              'min_terms_reduced': min_terms_symbol,
              'no_cares_string': None,
              'no_cares_reduced': None,
              'truthtable': truthtable,
              'kmap': kmap
              }
    
    data = {'input': input_, 'output': output_}
    all_data_samples.append(data)

with open('waveform_combinational.jsonl', 'w') as f:
    for x in all_data_samples:
        f.write(json.dumps(x))
        f.write('\n')
