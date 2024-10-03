
"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem 2014_q3c.

```
Given the state-assigned table shown below, implement the logic functions Y[0] and z.
// Present state y[2:0] | Next state Y[2:0] x=0, Next state Y[2:0] x=1 | Output z
// 000 | 000, 001 | 0
// 001 | 001, 100 | 0
// 010 | 010, 001 | 0
// 011 | 001, 010 | 1
// 100 | 011, 100 | 1
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
    if index == 0:
        reason = "Y0 corresponds to 001 (B), 011 (D)"
        code = "\tassign Y0 = ( next == B || next == D );"
        return reason, code
    
    elif index == 1:
        reason = "Y1 corresponds to 010 (C), 011 (D)"
        code = "\tassign Y1 = ( next == C || next == D );"
        return reason, code
    
    reason = "Y2 corresponds to 100 (E)"
    code = "\tassign Y2 = ( next == E );"
    return reason, code

decontaminate = """// Present state y[2:0] | Next state Y[2:0] x=0, Next state Y[2:0] x=1 | Output z
// 000 | 000, 001 | 0
// 001 | 001, 100 | 0
// 010 | 010, 001 | 0
// 011 | 001, 010 | 1
// 100 | 011, 100 | 1"""

problem_template = """Given the state-assigned table shown below, implement the logic functions Y[{index}] and z.
{transition_table}

module top_module (
\tinput clk,
\tinput x,
\tinput [2:0] y,
\toutput reg Y{index},
\toutput reg z
);"""

code_solution_template = """module top_module (
 input clk,
 input x,
 input [2:0] y,
 output reg Y{index},
 output reg z
);

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

The output is 1 for states: {output_asserted_states}.
Thus the output logic is: {output_logic}.

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

    return g, problem_template.format(transition_table=transition_table, reset_state=g.nodes[0]['name'], input_encoding=input_encoding, index=index), transition_table

def generate_reasoning_solution(g, reset_state, index):
    transition_table = print_state_table(g, num_states=num_states)

    parameters = get_parameters(g, num_states=num_states)

    transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    transition_logic_r = transition_logic.split('\n')[3:-3]
    transition_logic_r = [x.strip() for x in transition_logic_r]

    output_logic, output_asserted_states = print_output_logic(g, num_states=num_states, output_signal_name=output_signal_name)
    output_logic = output_logic.replace('state', 'y')
    output_logic_r = output_logic.strip()

    output_y_reason, output_y_code = output_Y_reason(index=index)
    output_logic_r += "\n" + output_y_reason
    output_logic += "\n" + output_y_code

    
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state, index=index)
    code = code.replace('next', 'next_state')
    code = code.replace('case(state)', 'case(y)')

    return reasoning_solution_template.format(parameters=parameters, transition_table=transition_table, transition_logic='\n'.join(transition_logic_r), output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code)