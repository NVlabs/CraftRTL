"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

From problem m2014_q6b:
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

// Assume that you want to Implement the FSM using three flip-flops and state codes y[3:1] = 000, 001, ..., 101 for states A, B, ..., F, respectively. Implement just the next-state logic for y[2] in Verilog. The output Y2 is y[2].
```
"""
from utils import *

def import_global(a, b, c):
    global num_states, input_signal_name, output_signal_name
    num_states = a
    input_signal_name = b
    output_signal_name = c

def get_input_encoding(parameters):
    encoding = parameters[len('parameter '):-1].split(' ')
    return ' '.join(encoding)

def reformat_transition_table(table):
    replacement = {'A': '000', 'B': '001', 'C': '010', 'D': '011', 'E': '100'}
    table = table.replace('state |', 'Present state y[2:0] |')
    table = table.replace('in=0', 'Y[2:0] x=0')
    table = table.replace('in=1', 'Y[2:0] x=1')
    table = table.replace('Output', "Output z")
    for x in replacement:
        table = table.replace(x, replacement[x])
    return table

def output_Y_reason(index):
    if index == 1:
        reason = "Y1 corresponds to 001 (B), 011 (D)"
        code = "\tassign Y1 = ( next == B || next == D );"
        return reason, code
    
    elif index == 2:
        reason = "Y2 corresponds to 010 (C), 011 (D)"
        code = "\tassign Y2 = ( next == C || next == D );"
        return reason, code
    
    reason = "Y3 corresponds to 100 (E)"
    code = "\tassign Y3 = ( next == E );"
    return reason, code

decontaminate = """// A (0) --0--> B
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
// F (1) --1--> D"""

problem_template = """Consider the state machine shown below:

{transition_table}

// Assume that you want to Implement the FSM using three flip-flops and state codes y[3:1] = 000, 001, ..., 101 for states A, B, ..., F, respectively. Implement just the next-state logic for y[{index}] in Verilog. The output Y{index} is y[{index}].

module top_module(
\tinput [3:1] y,
\tinput w,
\toutput reg Y{index});"""

code_solution_template = """module top_module(
\tinput [3:1] y,
\tinput w,
\toutput reg Y{index});

\treg [2:0] next;
{parameters}

{next_state_transition}

{output_assignment}

endmodule"""

reasoning_solution_template = """I will first use the following parameters: {parameters}
Thus the table will be as follow:
{transition_table}

The transition logic is then:
{transition_logic}

{output_logic}.

Finally, below is the Verilog code for the finite state machine:
```
{code}
```"""

def generate_question(inverse_state_names, index):
    g = generate_transition_graph(num_states=num_states)
    while not g:
        g = generate_transition_graph(num_states=num_states)
    g, state_graph = assign_output_to_state_and_print_graph(g, num_states=num_states, inverse_state_names=inverse_state_names)

    parameters = get_parameters(g, num_states=num_states)
    input_encoding = get_input_encoding(parameters)

    transition_table = reformat_transition_table(print_state_table(g, num_states=num_states))

    return g, problem_template.format(transition_table=state_graph, reset_state=g.nodes[0]['name'], input_encoding=input_encoding, index=index), state_graph

def generate_reasoning_solution(g, reset_state, index):
    transition_table = print_state_table(g, num_states=num_states)

    parameters = get_parameters(g, num_states=num_states)

    transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    transition_logic_r = transition_logic.split('\n')[3:-3]
    transition_logic_r = [x.strip() for x in transition_logic_r]

    output_logic, output_asserted_states = print_output_logic(g, num_states=num_states, output_signal_name=f'Y{index}')
    #output_logic = output_logic.replace('state', 'y')
    output_logic_r = output_logic.strip()

    output_y_reason, output_y_code = output_Y_reason(index=index)
    output_logic_r = output_y_reason + f"\nThus the code is `{output_y_code.strip().replace('next', 'next_state')}`"
    output_logic = "\n" + output_y_code 

    
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state, index=index)
    code = code.replace('next', 'next_state')
    code = code.replace('case(state)', 'case(y)')

    return reasoning_solution_template.format(parameters=parameters, transition_table=transition_table, transition_logic='\n'.join(transition_logic_r), output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code)