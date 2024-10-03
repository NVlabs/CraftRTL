"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

This script is similar to sample_kmap except the variable naming to x[3:0]
Template based problem and solution generation for Karnaugh Maps. 

Decontaminate with following:
m2014_q3:
//        x[1]x[2]
// x[3]x[4]   00 01 11 10
//  00 | d | 0 | d | d |
//  01 | 0 | d | 1 | 0 |
//  11 | 1 | 1 | d | d |
//  10 | 1 | 1 | 0 | d |

min_terms = 0111, 1100, 1101, 1000, 1001
min_terms = [7, 12, 13, 8, 5]

if sample_no_care is False, also deconaminate on following. Also need to slightly adjust `problem` statement accordingly.
2012_q1g:
//        x[1]x[2]
// x[3]x[4]   00 01 11 10
//  00 | 1 | 0 | 0 | 1 |
//  01 | 0 | 0 | 0 | 0 |
//  11 | 1 | 1 | 1 | 0 |
//  10 | 1 | 1 | 0 | 1 |

min_terms = 0000, 0010, 1100, 1101, 1111, 1000, 1001, 1010
min_terms = [0, 2, 12, 13, 15, 8, 9, 10]

"""

from utils import *
import json
random.seed('abcd')
num_samples=2 # Set this to the desired amount of data samples.
all_data_samples = []
from tqdm import tqdm
from sympy import symbols

problem = """Consider the function f shown in the Karnaugh map below. d is don't-care, which means you may choose to output whatever value is convenient. Implement this function. 
{kmap}

module top_module(
\tinput [4:1] x,
\toutput logic f
);"""

solution = """The input variables are: {variables}.
Based on the Karnaugh map, I can transform in to the following truth table:
{truthtable}

The minterms (when output is 1) are:
{minterms}
This corresponds to the following minterms logic:
`{logic_equation}`

Finally, based on the above logic equation, I can now write the Verilog code that could be described by the Karnaugh map:
```
module top_module(
\tinput [4:1] x,
\toutput logic f
);

\tassign f = {logic_equation};
endmodule
```
"""

p1_min_terms = frozenset([7, 12, 13, 8, 5])
p2_min_terms = frozenset([0, 2, 12, 13, 15, 8, 9, 10])
benchmark_min_terms = set([p1_min_terms, p2_min_terms])

for _ in  tqdm(range(num_samples)):
    min_terms, no_cares = random_sample_minterm(4, True)  # set to False if don't want no-cares
    symbol = [symbols('x[3]'), symbols('x[4]'), symbols('x[1]'), symbols('x[2]')]

    sop_style = 1 # style of sop. set to 1 always.
    permute = 0 #random.choice([0,1])
    no_care_symbol = 'd' #random.choice(['x', 'X', 'd'])
    #print(symbol)
    #print(min_terms)

    # Decontaminate benchmarks
    if frozenset(min_terms) in benchmark_min_terms:
        print("Min terms in benchmark.")
        continue

    # generate terms
    min_terms_string, min_terms_symbol = convert_min_term_sop(min_terms, symbol, sop_style)
    min_terms_symbol = str(SOPform(symbol, min_terms, no_cares))
    #print(min_terms_symbol)
    no_cares_string, no_cares_symbol = None, None
    #print(min_terms_string)

    # generate truthtable and karnauph maps
    truthtable = print_table_minterms(min_terms, no_cares, symbol, no_care_symbol)
    kmap = print_karnaugh_map(min_terms, no_cares, symbol, no_care_symbol, permute, comment=True)

    input_ = problem.format(kmap=kmap)
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

with open('m2014_q3.jsonl', 'w') as f:
    for x in all_data_samples:
        f.write(json.dumps(x))
        f.write('\n')
