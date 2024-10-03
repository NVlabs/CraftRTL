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

state_names = ['A', 'B', 'C', 'D']

def import_global(a, b, c, d):
    global num_states, input_signal_name, output_signal_name, state_names
    num_states = a
    input_signal_name = b
    output_signal_name = c
    state_names = d

def assign_output_transition(g, num_states=4, input_bit=1):
    """
    This would always generate with 1 bit output and 1 bit input
    """

    # add transition logic
    input_states = list(range(2**input_bit))
    for i in range(num_states):
        random.shuffle(input_states)
        for x, j in enumerate(g.successors(i)):
            g[i][j]['output'] = random.choice([0, 1])#input_states[x]
    return g

def print_transition_logic_fsm2(g, num_states=2, input_signal_name=['j', 'k']):
    """
    This only works for input_bit=2; 
    """
    state_names = get_state_names_from_graph(g, num_states)
    all_lines = ["\talways_comb begin\n\t\tcase(state)\n"]

    for i in range(num_states):
        pos_state, neg_state = None, None
        for j in g.successors(i):
            if g[i][j]['state'] != 0:
                pos_state = j
            else:
                neg_state = j

        assert pos_state is not None
        assert neg_state is not None
        line = f"\t\t\t{state_names[i]}: next = x ? {state_names[pos_state]} : {state_names[neg_state]};"
        all_lines.append(line)

    all_lines += ["\t\t\tdefault: next = 'x;\n\t\tendcase\n\tend"]
    return '\n'.join(all_lines)

def format_input_sigals(input_signal_name=['j', 'k']):
    all_input = []
    for x in input_signal_name:
        all_input.append( f" input {x},"  )
    return '\n'.join(all_input)

def assign_output_to_state_and_print_graph_fsm2(g, num_states=4, input_bit=1, output_bit=1, inverse_state_names=False, style=0, state_names=None, input_signal_name=['output']):
    g = assign_output_to_state(g, num_states, output_bit)
    g = assign_state_names_to_node(g, num_states, inverse_state_names, state_names)
    all_lines = []

    for i in range(num_states):
        cur_lines = []
        for j in g.successors(i):
            line = f"// {g.nodes[i]['name']} --x={g[i][j]['state']} (z={    convert_to_binary(g[i][j]['output'], input_bit)     })--> {   g.nodes[j]['name']    }"
            cur_lines.append(  (-g[i][j]['state'], line)  )
        cur_lines.sort()
        for line in cur_lines:
            all_lines.append(line[1])
    return g, '\n'.join(all_lines)

def print_state_table_no_output(g, num_states=4, input_bit=1, output_bit=1):
    """
    This only works for input_bit=1; output_bit=1
    """
    all_lines = ["// state | next state in=0, next state in=1"]
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
        line = f"// {state_names[i]} | {state_names[neg_state]}, {state_names[pos_state]}"
        all_lines.append(line)

    return '\n'.join(all_lines)

def print_output_logic_transition(g, num_states=4, output_signal_name='z'):
    """
    This only works for output_bit=1; 
    """
    logic = []
    output_logic_states = []
    state_names = get_state_names_from_graph(g, num_states)
    for i in range(num_states):
        for j in g.successors(i):
            if g[i][j]['output'] != 0:
                if g[i][j]['state']:
                    logic += [f" ( state == {state_names[i]} & x ) "]
                    output_logic_states.append( f"({state_names[i]}, x)" )
                else:
                    logic += [f" ( state == {state_names[i]} & ~x ) "]
                    output_logic_states.append( f"({state_names[i]}, ~x)" )
    assert len(output_logic_states) > 0
    return f"\tassign {output_signal_name} = ({'||'.join(logic)});", ", ".join(output_logic_states)

problem_template = """The following diagram is a Mealy machine. Implement in Verilog using one-hot encoding. Resets into state A and reset is asynchronous active-high.

{state_graph}

module top_module (
\tinput clk,
\tinput areset,
\tinput x,
\toutput z
);"""

code_solution_template = """module top_module (
\tinput clk,
\tinput areset,
\tinput x,
\toutput z
);

{parameters}
\treg [1:0] state;
\treg [1:0] next;

{next_state_transition}

\talways @(posedge clk, posedge areset) begin
\t\tif (areset) state <= {reset_state};
\t\telse state <= next;
\tend

{output_assignment}

endmodule"""

reasoning_solution_template = """From the transition diagram, we have the following transition logic and output logic:
{transition_table}

Thus the state transition logic is as follows:
{transition_logic}

The output is 1 for states: {output_asserted_states}.
Thus the output logic is: `{output_logic}`.

Finally, below is the Verilog code for the finite state machine:
```
{code}
```"""

def generate_question(inverse_state_names):
    g = None
    while not g:
        g = generate_transition_graph(num_states=num_states)
    g = assign_output_transition(g, num_states=num_states)
    g, state_graph = assign_output_to_state_and_print_graph_fsm2(g, num_states=num_states, inverse_state_names=inverse_state_names, style=1, state_names=state_names, input_signal_name=input_signal_name)
    if random.random() > 0.5:
        state_graph = swap_state_graph(state_graph)

    return g, problem_template.format(state_graph=state_graph, reset_state=g.nodes[0]['name'], additional_inputs=format_input_sigals(input_signal_name)), state_graph

def generate_reasoning_solution(g, reset_state):
    transition_table = print_state_table_no_output(g, num_states=num_states)

    transition_logic = print_transition_logic_fsm2(g, num_states=num_states, input_signal_name=input_signal_name)
    transition_logic_r = transition_logic.split('\n')[3:-3]
    transition_logic_r = [x.strip() for x in transition_logic_r]

    output_logic, output_asserted_states = print_output_logic_transition(g, num_states=num_states, output_signal_name=output_signal_name)
    output_logic_r = output_logic.strip()

    parameters = get_parameters(g, num_states=num_states)
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state, additional_inputs=format_input_sigals(input_signal_name))

    code = code.replace('next', 'next_state')

    return reasoning_solution_template.format(transition_table=transition_table, transition_logic='\n'.join(transition_logic_r), output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code, additional_inputs=format_input_sigals(input_signal_name))