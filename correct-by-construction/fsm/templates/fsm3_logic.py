"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem fsm3comb.

```
The following is the state transition table for a Moore state machine with one input, one output, and four states. Use the following state encoding: A=2'b00, B=2'b01, C=2'b10, D=2'b11.Implement only the state transition logic and output logic (the combinational logic portion) for this state machine. Given the current state (state), compute the next_state and output (out) based on the state transition table.
// State | Next state in=0, Next state in=1 | Output
// A | A, B | 0
// B | C, B | 0
// C | A, D | 0
// D | C, B | 1
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

decontaminate = """// state | Next state in=0, Next state in=1 | Output
// A | A, B | 0
// B | C, B | 0
// C | A, D | 0
// D | C, B | 1"""

problem_template = """The following is the state transition table for a Moore state machine with one input, one output, and four states. Use the following state encoding: {input_encoding}.Implement only the state transition logic and output logic (the combinational logic portion) for this state machine. Given the current state (state), compute the next_state and output (out) based on the state transition table.
{transition_table}

module top_module (
 input in,
 input [1:0] state,
 output reg [1:0] next_state,
 output out
);"""

code_solution_template = """module top_module (
 input in,
 input [1:0] state,
 output reg [1:0] next_state,
 output out
);

{parameters}

{next_state_transition}

{output_assignment}

endmodule"""

reasoning_solution_template = """The state transition logic is as follows:
{transition_logic}

The output is 1 for states: {output_asserted_states}.
Thus the output logic is: `{output_logic}`.

Finally, below is the Verilog code for the finite state machine:
```
{code}
```"""

def generate_question(inverse_state_names):
    g = generate_transition_graph(num_states=num_states)
    g, state_graph = assign_output_to_state_and_print_graph(g, num_states=num_states, inverse_state_names=inverse_state_names)

    parameters = get_parameters(g, num_states=num_states)
    input_encoding = get_input_encoding(parameters)

    transition_table = print_state_table(g, num_states=num_states)

    return g, problem_template.format(transition_table=transition_table, reset_state=g.nodes[0]['name'], input_encoding=input_encoding), transition_table

def generate_reasoning_solution(g, reset_state):
    transition_table = print_state_table(g, num_states=num_states)

    parameters = get_parameters(g, num_states=num_states)

    transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    transition_logic_r = transition_logic.split('\n')[3:-3]
    transition_logic_r = [x.strip() for x in transition_logic_r]

    output_logic, output_asserted_states = print_output_logic(g, num_states=num_states, output_signal_name=output_signal_name)
    output_logic_r = output_logic.strip()

    
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state)
    code = code.replace('next', 'next_state')

    return reasoning_solution_template.format(transition_table=transition_table, transition_logic='\n'.join(transition_logic_r), output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code)