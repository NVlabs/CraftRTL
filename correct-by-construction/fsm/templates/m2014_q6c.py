
"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

From problem m2014_q6b:
```
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

// Resets into state A. For this part, assume that a one-hot code is used with the state assignment y[6:1] = 000001, 000010, 000100, 001000, 010000, 100000 for states A, B,..., F, respectively.

// Write Verilog for the next-state signals Y2 and Y4 corresponding to signal y[2] and y[4]. Derive the logic equations by inspection assuming a one-hot encoding.
```
"""
from utils import *

def import_global(a, b, c):
    global num_states, input_signal_name, output_signal_name
    num_states = a
    input_signal_name = b
    output_signal_name = c

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
        if len(transitions[i]) == 0:
            assert i == 0
            reason = f"Next state for {g.nodes[i]['name']} is not reachable on any (row, colum) and will be only initialized by reset."
            all_reasons.append(reason)
            continue

        for in_state, previous_state in transitions[i]:
            if in_state == 0:
                code.append( f"y[{previous_state}] & ~w" )
            else:
                code.append( f"y[{previous_state}] & w" )
            reason += f" ({previous_state}, in={in_state})"
        reason += f". This correspond to the following logic: `{' || '.join(code)}`."
        all_code.append(  f"\tassign next[{g.nodes[i]['name']}] = {' || '.join(code)};" )
        all_reasons.append(reason)
    
    return '\n'.join(all_code), '\n'.join(all_reasons)

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
    reason, code = "", ""
    for i in index:
        index_mapping = ["6'b000001", "6'b000010", "6'b000100", "6'b001000", "6'b010000", "6'b100000"]
        name_mapping  = ['A', 'B', 'C', 'D', 'E', 'F']
        reason, code = "", ""
        for i in index:
            code += f"\tassign Y{i} = next[{name_mapping[i-1]}];\n"
            reason += f"Y{i} correspondes to {index_mapping[i-1]} ({name_mapping[i-1]})\n" + f"Thus the logic is: \tassign Y{i} = next[{name_mapping[i-1]}];\n"
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

// Resets into state A. For this part, assume that a one-hot code is used with the state assignment y[6:1] = 000001, 000010, 000100, 001000, 010000, 100000 for states A, B,..., F, respectively.

// Write Verilog for the next-state signals Y2 and Y4 corresponding to signal y[{index[0]}] and y[{index[1]}]. Derive the logic equations by inspection assuming a one-hot encoding.

module top_module (
\tinput [6:1] y,
\tinput w,
\toutput Y{index[0]},
\toutput Y{index[1]}
);"""

code_solution_template = """module top_module (
\tinput [6:1] y,
\tinput w,
\toutput Y{index[0]},
\toutput Y{index[1]}
);

\treg [6:1] next;
{parameters}

{next_state_transition}

{output_assignment}

endmodule"""

reasoning_solution_template = """The state transition table will be as follow:
{transition_table}

The transition logic is then:
{transition_logic}

{output_logic}

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

    parameters = get_parameters(g, num_states=num_states)#, one_hot=True)

    #transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    #transition_logic_r = transition_logic.split('\n')[3:-3]
    #transition_logic_r = [x.strip() for x in transition_logic_r]
    transition_logic, transition_logic_r = get_code_and_logic(g, num_states)
    if transition_logic is None:
        return None

    output_logic, output_asserted_states = print_output_logic_one_hot(g, num_states=num_states, output_signal_name='Y')
    #output_logic = output_logic.replace('state', 'y')
    output_logic_r = output_logic.strip()

    output_y_reason, output_y_code = output_Y_reason(index=index)
    output_logic_r = output_y_reason 
    output_logic = "\n" + output_y_code 

    parameters = "parameter A=1, B=2, C=3, D=4, E=5, F=6;"    
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state, index=index)
    code = code.replace('next', 'next_state')
    code = code.replace('case(state)', 'case(y)')

    return reasoning_solution_template.format(parameters=parameters, transition_table=transition_table, transition_logic=transition_logic_r, output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code)