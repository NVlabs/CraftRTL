# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from utils_fsm import *
from simulation_fsm import obtain_waveform

def import_global(a, b, c):
    global num_states, input_signal_name, output_signal_name
    num_states = a
    input_signal_name = b
    output_signal_name = c


problem_template = """This is a sequential circuit. Read the simulation waveforms to determine what the circuit does, then implement it.

{waveform}

module top_module (
 input clk,
 input in,
 input reset,
 output out
);"""

code_solution_template = """module top_module (
 input clk,
 input in,
 input reset,
 output out
);

{parameters}
\treg state;
\treg next;

{next_state_transition}

\talways @(posedge clk) begin
\t\tif (reset) state <= {reset_state};
\t\telse state <= next;
\tend

{output_assignment}

endmodule"""

reasoning_solution_template = """From the waveform, we have the following transition logic and output logic:
{transition_table}

Thus the state transition logic is as follows:
{transition_logic}

The output is 1 for states: {output_asserted_states}.
Thus the output logic is: `{output_logic}`.

Finally, below is the Verilog code for the finite state machine:
```
{code}
```"""

with open('./test/test_fsm1s.v', 'r') as f:
    test = f.read()

def generate_question(inverse_state_names):
    g = generate_transition_graph(num_states=num_states)
    g, state_graph = assign_output_to_state_and_print_graph(g, num_states=num_states, inverse_state_names=inverse_state_names, style=1)
    return g, problem_template.format(waveform=" ", reset_state=g.nodes[0]['name']), state_graph

def generate_reasoning_solution(g, reset_state):
    transition_table = print_state_table(g, num_states=num_states)

    transition_logic = print_transition_logic(g, num_states=num_states, input_signal_name=input_signal_name)
    transition_logic_r = transition_logic.split('\n')[3:-3]
    transition_logic_r = [x.strip() for x in transition_logic_r]

    output_logic, output_asserted_states = print_output_logic(g, num_states=num_states, output_signal_name=output_signal_name)
    output_logic_r = output_logic.strip()

    parameters = get_parameters(g, num_states=num_states)
    code = code_solution_template.format(parameters='\t'+parameters, next_state_transition=transition_logic, output_assignment=output_logic, reset_state=reset_state)

    cur_test = test + "\n" + code
    waveform = obtain_waveform(cur_test)[0]
    assert waveform is not None

    def contain_meaningful_output(waveform):
        """
        Since we perform random test inputs, we can not guarantee that the output state will even flip to 1....
        """
        lines = waveform.split('\n')[1:] #remove first line of header
        for t in lines:
            # return true if output flips
            t = "".join(t.split()) # remove whitespace
            if t[-1] != '0': # last char in each line is output bit
                return True 
        return False

    if not contain_meaningful_output(waveform):
        return None, None
    problem = problem_template.format(waveform=waveform, reset_state=g.nodes[0]['name'])

    return reasoning_solution_template.format(transition_table=transition_table, transition_logic='\n'.join(transition_logic_r), output_logic=output_logic_r, output_asserted_states=output_asserted_states, code=code), problem