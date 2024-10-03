"""
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

Template based problem and solution generation for problem fsm_onehot.

```
Given the follow state machine with 1 input and 2 outputs (the outputs are given as "(out1, out2)"):

// S0 (0, 0) --0--> S0
// S0 (0, 0) --1--> S1
// S1 (0, 0) --0--> S0
// S1 (0, 0) --1--> S2
// S2 (0, 0) --0--> S0
// S2 (0, 0) --1--> S3
// S3 (0, 0) --0--> S0
// S3 (0, 0) --1--> S4
// S4 (0, 0) --0--> S0
// S4 (0, 0) --1--> S5
// S5 (0, 0) --0--> S8
// S5 (0, 0) --1--> S6
// S6 (0, 0) --0--> S9
// S6 (0, 0) --1--> S7
// S7 (0, 1) --0--> S0
// S7 (0, 1) --1--> S7
// S8 (1, 0) --0--> S0
// S8 (1, 0) --1--> S1
// S9 (1, 1) --0--> S0
// S9 (1, 1) --1--> S1

// Suppose this state machine uses one-hot encoding, where state[0] through state[9] correspond to the states S0 though S9, respectively. The outputs are zero unless otherwise specified.

// Write Verilog implementing the state transition logic and output logic portions of the state machine (but not the state flip-flops). You are given the current state in state[9:0] and must produce next_state[9:0] and the two outputs. Derive the logic equations by inspection assuming a one-hot encoding.
```
"""

from utils import *

def import_global(a, b, c):
    global num_states, input_signal_name, output_signal_name
    num_states = a
    input_signal_name = b
    output_signal_name = c

def assign_output_to_state_local(g, num_states=4, output_bit=1, output_names=['out1', 'out2']):
    

    for output_name in output_names:
        all_output_states = list(range(2**output_bit))
        available_states = set(range(num_states))
    # ensure all output states covered
        for state in all_output_states:
            t = random.choice(list(available_states))
            g.nodes[t][output_name] = state
            available_states.remove(t)

        # annotate all other states
        for state in available_states:
            t = random.choice(list(all_output_states))
            g.nodes[state][output_name] = t
    return g

def assign_output_to_state_and_print_graph_local(g, num_states=4, input_bit=1, output_bit=1, inverse_state_names=False, style=0, state_names=None, output_names=['out1', 'out2']):
    g = assign_output_to_state_local(g, num_states, output_bit)
    g = assign_state_names_to_node(g, num_states, inverse_state_names, state_names)
    all_lines = []

    for i in range(num_states):
        local_lines = []
        for j in g.successors(i):

            line = f"// {g.nodes[i]['name']} ({   convert_to_binary(g.nodes[i]['out1'], output_bit)    }, {   convert_to_binary(g.nodes[i]['out2'], output_bit)    }) --{    convert_to_binary(g[i][j]['state'], input_bit)     }--> {   g.nodes[j]['name']    }"
            local_lines.append( (g[i][j]['state'], line)  )

        local_lines.sort()
        for i_, line in local_lines:
            all_lines.append(line)
    return g, '\n'.join(all_lines)

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

def print_output_logic_one_hot(g, num_states=4, output_signal_name='out2'):
    """
    This only works for output_bit=1; 
    """
    logic = []
    output_logic_states = []
    state_names = get_state_names_from_graph(g, num_states)
    for i in range(num_states):
        if g.nodes[i][output_signal_name] != 0:
            #print(g.nodes[i][output_signal_name], g.nodes[i]['out2'])
            logic += [f" state[{state_names[i]}] "]
            output_logic_states.append( state_names[i])
    return f"\tassign {output_signal_name} = ({'||'.join(logic)});", ", ".join(output_logic_states)

def print_state_table_local(g, num_states=4, input_bit=1, output_bit=1):
    """
    This only works for input_bit=1; output_bit=1
    """
    all_lines = ["// state | Next state in=0, Next state in=1 | (out1, out2)"]
    state_names = get_state_names_from_graph(g, num_states)

    for i in range(num_states):
        pos_state, neg_state = None, None
        for j in g.successors(i):
            if g[i][j]['state'] != 0:
                pos_state = j
            else:
                neg_state = j

        assert pos_state is not None
        assert neg_state is not None
        line = f"// {state_names[i]} | {state_names[neg_state]}, {state_names[pos_state]} | ({g.nodes[i]['out1']}, {g.nodes[i]['out2']})"
        all_lines.append(line)

    return '\n'.join(all_lines)

decontaminate = """// S0 (0, 0) --0--> S0
// S0 (0, 0) --1--> S1
// S1 (0, 0) --0--> S0
// S1 (0, 0) --1--> S2
// S2 (0, 0) --0--> S0
// S2 (0, 0) --1--> S3
// S3 (0, 0) --0--> S0
// S3 (0, 0) --1--> S4
// S4 (0, 0) --0--> S0
// S4 (0, 0) --1--> S5
// S5 (0, 0) --0--> S8
// S5 (0, 0) --1--> S6
// S6 (0, 0) --0--> S9
// S6 (0, 0) --1--> S7
// S7 (0, 1) --0--> S0
// S7 (0, 1) --1--> S7
// S8 (1, 0) --0--> S0
// S8 (1, 0) --1--> S1
// S9 (1, 1) --0--> S0
// S9 (1, 1) --1--> S1"""

problem_template = """Given the follow state machine with 1 input and 2 outputs (the outputs are given as \"(out1, out2)\"):

{state_graph}

// Suppose this state machine uses one-hot encoding, where state[0] through state[9] correspond to the states S0 though S9, respectively. The outputs are zero unless otherwise specified.

// Write Verilog implementing the state transition logic and output logic portions of the state machine (but not the state flip-flops). You are given the current state in state[9:0] and must produce next_state[9:0] and the two outputs. Derive the logic equations by inspection assuming a one-hot encoding.

module top_module (
\tinput in,
\tinput [9:0] state,
\toutput [9:0] next_state,
\toutput out1,
\toutput out2);"""

code_solution_template = """module top_module (
\tinput in,
\tinput [9:0] state,
\toutput [9:0] next_state,
\toutput out1,
\toutput out2);

{parameters}

{next_state_transition}

{output_assignment}

endmodule"""

reasoning_solution_template = """The state transition logic is as follows:
{transition_table}

{transition_logic}

out1 is 1 for states: {output_asserted_states_1}.
Thus the output logic is: `{output_logic_1}`.

out2 is 1 for states: {output_asserted_states_2}.
Thus the output logic is: `{output_logic_2}`.

Finally, below is the Verilog code for the finite state machine:
```
{code}
```"""
def get_parameters_local(g, num_states=4, one_hot=False):
    state_names = get_state_names_from_graph(g, num_states)

    if one_hot:
        state_bits = generate_one_hot_sequences(num_states)
        state_bits = [f"{num_states}'b{i}" for i in state_bits]
    else:    
        num_bits = math.ceil(math.log2(num_states))
        if num_bits > 1 and random.random() > 1.5:
            state_bits = [f"{num_bits}'b{convert_to_binary(i, num_bits)}" for i in range(num_states)]
        else:
            state_bits = list(range(num_states))

    all_lines = []
    for i, j in zip(state_names, state_bits):
        line = f"{i}={j}"
        all_lines.append(line)

    all_lines = ', '.join(all_lines)

    return f"parameter {all_lines};"

def generate_question(inverse_state_names, state_names):
    g = None
    while g is None:
        g = generate_transition_graph(num_states=num_states)
    g, state_graph = assign_output_to_state_and_print_graph_local(g, num_states=num_states, inverse_state_names=inverse_state_names, state_names=state_names)

    parameters = get_parameters(g, num_states=num_states, one_hot=True)
    input_encoding = get_input_encoding(parameters)

    transition_table = print_state_table_local(g, num_states=num_states)

    return g, problem_template.format(transition_table=transition_table, reset_state=g.nodes[0]['name'], input_encoding=input_encoding, state_graph=state_graph), state_graph

def generate_reasoning_solution(g, reset_state):
    transition_table = print_state_table_local(g, num_states=num_states)

    # Interstingly we stick with this for onehot cases....
    parameters = get_parameters_local(g, num_states=num_states)#, one_hot=True)

    #transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    #transition_logic_r = transition_logic.split('\n')[3:-3]
    #transition_logic_r = [x.strip() for x in transition_logic_r]
    transition_logic, transition_logic_r = get_code_and_logic(g, num_states)
    if transition_logic is None:
        return None

    output_logic_1, output_asserted_states_1 = print_output_logic_one_hot(g, num_states=num_states, output_signal_name='out1')
    output_logic_r_1 = output_logic_1.strip()

    output_logic_2, output_asserted_states_2 = print_output_logic_one_hot(g, num_states=num_states, output_signal_name='out2')
    output_logic_r_2 = output_logic_2.strip()


    
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic_1 + "\n" + output_logic_2 , reset_state=reset_state)
    #code = code.replace('next', 'next_state')

    return reasoning_solution_template.format(transition_table=transition_table, transition_logic=transition_logic_r, output_logic_1=output_logic_r_1, output_asserted_states_1=output_asserted_states_1, output_logic_2=output_logic_r_2, output_asserted_states_2=output_asserted_states_2, code=code)