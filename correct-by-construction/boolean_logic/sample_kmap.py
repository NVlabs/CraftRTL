"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for Karnaugh Maps.

kmap1 is 3 variables, we sample for 4 variables in this script. You can modify this script to enumerate all var3.

Decontaminate with following:
kmap2:
//        ab
// cd   00 01 11 10
//  00 | 1 | 1 | 0 | 1 |
//  01 | 1 | 0 | 0 | 1 |
//  11 | 0 | 1 | 1 | 1 |
//  10 | 1 | 1 | 0 | 0 |

min_terms = 0000, 0001, 0010, 0100, 0110, 1101, 1111, 1110, 1000, 1001 
min_terms = [0, 1, 2, 4, 6, 13, 15, 14, 8, 9]

if sample_no_care is True, also deconaminate on following. Also need to slightly adjust `problem` statement accordingly to indicate no-cares.
kmap4:
//        ab
// cd   00 01 11 10
//  00 | 0 | 1 | 0 | 1 |
//  01 | 1 | 0 | 1 | 0 |
//  11 | 0 | 1 | 0 | 1 |
//  10 | 1 | 0 | 1 | 0 |

min_terms = 0001, 0010, 0100, 0111, 1101, 1110, 1000, 1011
min_terms = [1, 2, 4, 7, 13, 14, 8, 11]


if permute = 1, this would randomly shift row/columns in representation. Deconatminate agains kmap3:
//        ab
// cd   01 00 10 11
//  00 | d | 0 | 1 | 1 |
//  01 | 0 | 0 | d | d |
//  11 | 0 | 1 | 1 | 1 |
//  10 | 0 | 1 | 1 | 1 |

min_terms = 0010, 0011, 1100, 1110, 1111, 1000, 1010, 1011
min_terms = [2, 3, 12, 14, 15, 8, 10, 11]

"""

from utils import *
import json

random.seed('1234')
num_samples=2  # Set this to the desired amount of data samples.
all_data_samples = []
sample_no_care = False # we don't sample don't cares for this. Set to True for kmap3 like data.

from sympy import symbols
from tqdm import tqdm

problem = """Implement the circuit described by the Karnaugh map below.

{kmap}

module top_module(
\tinput a, 
\tinput b,
\tinput c,
\tinput d,
\toutput out
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
\tinput a, 
\tinput b,
\tinput c,
\tinput d,
\toutput out
);

\tassign out = {logic_equation};
endmodule
```
"""

kmap2_min_terms = frozenset([0, 1, 2, 4, 6, 13, 15, 14, 8, 9])
kmap3_min_terms = frozenset([2, 3, 12, 14, 15, 8, 10, 11])
kmap4_min_terms = frozenset([1, 2, 4, 7, 13, 14, 8, 11])
benchmark_min_terms = set([kmap2_min_terms, kmap3_min_terms, kmap4_min_terms])

for _ in  tqdm(range(num_samples)):
    min_terms, no_cares = random_sample_minterm(4, False) 

    # Decontaminate benchmarks
    if frozenset(min_terms) in benchmark_min_terms:
        print("Min terms in benchmark.")
        continue

    symbol = [symbols('c'), symbols('d'), symbols('a'), symbols('b')]
    no_cares = []
    sop_style = 1 # style of sop. set to 1 always.
    permute = 0 # this would turn on permutation for kmap4 like data.
    no_care_symbol = ['d'] # random.choice(['x', 'X', 'd']). if you want to include such diversity.
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

with open('kmap.jsonl', 'w') as f:
    for x in all_data_samples:
        f.write(json.dumps(x))
        f.write('\n')
