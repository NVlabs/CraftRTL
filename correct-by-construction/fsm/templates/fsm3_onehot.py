"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem ece241_2014_q5b.
The original problem only has 2 states, we sample for 4 states. No need to decontaminate.

The following diagram is a Mealy machine implementation of the 2's complementer. Implement in Verilog using one-hot encoding. Resets into state A and reset is asynchronous active-high.

// A --x=0 (z=0)--> A
// A --x=1 (z=1)--> B
// B --x=0 (z=1)--> B
// B --x=1 (z=0)--> B

Note: we also generalize the problem statement to `The following diagram is a Mealy machine.` 
We also adjust the problem statement to the following during evaluation:
```
The following diagram is a Mealy machine. Implement in Verilog using one-hot encoding. Resets into state A and reset is asynchronous active-high.
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

def get_code_and_logic(g, num_states=4):
    transitions = [] # (input, previous_state)
    for i in range(num_states):
        cur_transition = []
        for x in g.predecessors(i):
            cur_transition.append(  (g[x][i]['state'], g.nodes[x]['name'])   )
        transitions.append(sorted(cur_transition))
    # We check if state A have incoming edges
    # other states are guaranteed to have incoming edges due to reachability
    if len(transitions[0]) == 0:
        return None, None
    all_code = []
    all_reasons = ['Based on the state transition table, we can obtain the next state from observing the row (previous state) and column (input).']
    for i in range(num_states):
        code = []
        reason = f"Next state is {g.nodes[i]['name']} on the following (row, column):"
        for in_state, previous_state in transitions[i]:
            if in_state == 0:
                code.append( f"state[{previous_state}] & ~in" )
            else:
                code.append( f"state[{previous_state}] & in" )
            reason += f" ({previous_state}, in={in_state})"
        reason += f". This correspond to the following logic: `{' || '.join(code)}`."
        all_code.append(  f"\tassign next_state[{g.nodes[i]['name']}] = {' || '.join(code)};" )
        all_reasons.append(reason)
    
    return '\n'.join(all_code), '\n'.join(all_reasons)

def print_output_logic_one_hot(g, num_states=4, output_signal_name='z'):
    """
    This only works for output_bit=1; 
    """
    logic = []
    output_logic_states = []
    state_names = get_state_names_from_graph(g, num_states)
    for i in range(num_states):
        if g.nodes[i]['output'] != 0:
            logic += [f" state[{state_names[i]}] "]
            output_logic_states.append( state_names[i])
    return f"\tassign {output_signal_name} = ({'||'.join(logic)});", ", ".join(output_logic_states)

decontaminate = """// State | Next state in=0, Next state in=1 | Output
// A | A, B | 0
// B | C, B | 0
// C | A, D | 0
// D | C, B | 1"""

problem_template = """The following is the state transition table for a Moore state machine with one input, one output, and four states. Use the following one-hot state encoding: {input_encoding}. Derive state transition and output logic equations by inspection assuming a one-hot encoding. Implement only the state transition logic and output logic (the combinational logic portion) for this state machine. 
{transition_table}

module top_module (
\tinput in,
\tinput [3:0] state,
\toutput reg [3:0] next_state,
\toutput out
);"""

code_solution_template = """module top_module (
 input in,
 input [3:0] state,
 output reg [3:0] next_state,
 output out
);

{parameters}

{next_state_transition}

{output_assignment}

endmodule"""

reasoning_solution_template = """{transition_logic}

The output is 1 for states: {output_asserted_states}.
Thus the output logic is: `{output_logic}`.

Finally, below is the Verilog code for the finite state machine:
```
{code}
```"""

def generate_question(inverse_state_names):
    g = generate_transition_graph(num_states=num_states)
    g, state_graph = assign_output_to_state_and_print_graph(g, num_states=num_states, inverse_state_names=inverse_state_names)

    parameters = get_parameters(g, num_states=num_states, one_hot=True)
    input_encoding = get_input_encoding(parameters)

    transition_table = print_state_table(g, num_states=num_states)

    return g, problem_template.format(transition_table=transition_table, reset_state=g.nodes[0]['name'], input_encoding=input_encoding), transition_table

def generate_reasoning_solution(g, reset_state):
    transition_table = print_state_table(g, num_states=num_states)

    # Interstingly we stick with this for onehot cases....
    parameters = get_parameters(g, num_states=num_states) #, one_hot=True)

    #transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    #transition_logic_r = transition_logic.split('\n')[3:-3]
    #transition_logic_r = [x.strip() for x in transition_logic_r]
    transition_logic, transition_logic_r = get_code_and_logic(g, num_states)
    if transition_logic is None:
        return None

    output_logic, output_asserted_states = print_output_logic_one_hot(g, num_states=num_states, output_signal_name=output_signal_name)
    output_logic_r = output_logic.strip()


    
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state)
    #code = code.replace('next', 'next_state')

    return reasoning_solution_template.format(transition_table=transition_table, transition_logic=transition_logic_r, output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code)