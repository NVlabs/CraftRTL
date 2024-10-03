"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for Truth Tables.

truthtable1 is 3 variables, we sample for 4 variables in this script. You can modify this script to enumerate all var3.

No need to decontaminate with truthtable1 unless enumerate var3. We omit the implelmentation.

"""

from utils import *
import json
random.seed('1234')
all_data_samples = []
sample_no_care = False # we don't sample don't cares for this
from sympy import symbols
from tqdm import tqdm 

num_samples=2

problem = """Create a combinational circuit that implements the truth table.

{truthtable}

module top_module(
\tinput x4,
\tinput x3, 
\tinput x2,
\tinput x1,
\toutput f
);"""

solution = """The input variables are: {variables}.
Based on the truth table:

The minterms (when output is 1) are:
{minterms}
This corresponds to the following minterms logic:
`{logic_equation}`

Finally, based on the above logic equation, I can now write the Verilog code:
```
module top_module(
\tinput x4,
\tinput x3, 
\tinput x2,
\tinput x1,
\toutput f
);

\tassign f = {logic_equation};
endmodule
```
"""

# for min_term in enumerate_min_terms(3):
# Use the above to enumerate min terms and comment out random_sample_minterm
# Refer to utils.py for implementation details.

for _ in tqdm(range(num_samples)):
    min_term, no_cares = random_sample_minterm(4, False)  # set to True if generating don't cares
    symbol = [symbols('x4'), symbols('x3'), symbols('x2'), symbols('x1')]
    min_terms = list(min_term)
    #no_cares = [] 
    sop_style = 1 
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
    truthtable = print_table_minterms(min_terms, no_cares, symbol, no_care_symbol, comment=True)
    kmap = print_karnaugh_map(min_terms, no_cares, symbol, no_care_symbol, permute)#, comment=True)

    input_ = problem.format(truthtable=truthtable)
    output_ = solution.format(kmap=kmap, truthtable=truthtable, minterms=convert_min_terms(min_terms_string, min_terms, 3), logic_equation=min_terms_string, variables=[str(x) for x in symbol])

    sample = {'variables': [str(x) for x in symbol], 
              'min_terms': min_term,
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

with open('truthtable1.jsonl', 'w') as f:
    for x in all_data_samples:
        f.write(json.dumps(x))
        f.write('\n')
