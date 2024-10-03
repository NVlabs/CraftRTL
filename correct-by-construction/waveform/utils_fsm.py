# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import networkx as nx
import random
import numpy as np
import math

def swap_state_graph(state_graph):
    state_graph = state_graph.split('\n')
    new_state_graph = []
    for i in range(len(state_graph)//2):
        new_state_graph +=   [   state_graph[2*i+1]  , state_graph[2*i]  ]
    return '\n'.join(new_state_graph)

def generate_transition_graph(num_states=4, input_bit=1):
    """
    This would always generate with 1 bit output and 1 bit input
    """
    # ensure reachability by generating a tree
    G = nx.generators.trees.random_tree(num_states,create_using=nx.DiGraph())

    # find root this is reset state 
    root = [n for n,d in G.in_degree() if d==0] 
    assert len(root) == 1
    root = root[0]
    assert root == 0

    # Ensure input coverage, made intersting since all different states, include self loops
    for node in range(num_states):
        if G.out_degree(node) > 2**input_bit:
            return None
        while G.out_degree(node) < 2**input_bit:
            all_nodes = set(G.nodes())
            neighbors = set(G.successors(node))
            non_neighbors = all_nodes - neighbors

            if len(non_neighbors) == 0:
                return None
            
            new_neighbor = random.choice(list(non_neighbors))
            G.add_edge(node, new_neighbor)


    # add transition logic
    input_states = list(range(2**input_bit))
    for i in range(num_states):
        random.shuffle(input_states)
        for x, j in enumerate(G.successors(i)):
            G[i][j]['state'] = input_states[x]
    return G


def convert_to_binary(num, bit):
    num = bin(num)[2:]
    if len(num) < bit:
        num = '0' * (bit-len(num)) + num
    return num


def assign_output_to_state(g, num_states=4, output_bit=1):
    all_output_states = list(range(2**output_bit))
    available_states = set(range(num_states))

    # ensure all output states covered
    for state in all_output_states:
        t = random.choice(list(available_states))
        g.nodes[t]['output'] = state
        available_states.remove(t)

    # annotate all other states
    for state in available_states:
        t = random.choice(list(all_output_states))
        g.nodes[state]['output'] = t
    return g

def assign_state_names_to_node(g, num_states=4, inverse_state_names=False, state_names=None):
    if state_names is None:
        state_names = [chr(ord('A')+i) for i in range(num_states)]
    if inverse_state_names:
        state_names = state_names[::-1]

    for i in range(num_states):
        g.nodes[i]['name'] = state_names[i]
    return g

def assign_output_to_state_and_print_graph(g, num_states=4, input_bit=1, output_bit=1, inverse_state_names=False, style=0, state_names=None):
    g = assign_output_to_state(g, num_states, output_bit)
    g = assign_state_names_to_node(g, num_states, inverse_state_names, state_names)
    all_lines = []

    for i in range(num_states):
        for j in g.successors(i):
            if style == 1:
                line = f"// {g.nodes[i]['name']} (out={   convert_to_binary(g.nodes[i]['output'], output_bit)    }) --in={    convert_to_binary(g[i][j]['state'], input_bit)     }--> {   g.nodes[j]['name']    }"
            else:
                line = f"// {g.nodes[i]['name']} ({   convert_to_binary(g.nodes[i]['output'], output_bit)    }) --{    convert_to_binary(g[i][j]['state'], input_bit)     }--> {   g.nodes[j]['name']    }"
            all_lines.append(line)
    return g, '\n'.join(all_lines)

def generate_one_hot_sequences(num_bits):
    num_sequences = num_bits
    one_hot_sequences = []
    for i in range(num_sequences):
        sequence = [0] * num_bits
        sequence[i] = 1
        sequence = ''.join([str(i) for i in sequence])
        one_hot_sequences.append(sequence)
    return one_hot_sequences[::-1]

def get_state_names_from_graph(g, num_states):
    names = []
    for i in range(num_states):
        names.append(  g.nodes[i]['name'] )
    return names

def get_parameters(g, num_states=4, one_hot=False):
    state_names = get_state_names_from_graph(g, num_states)

    if one_hot:
        state_bits = generate_one_hot_sequences(num_states)
        state_bits = [f"{num_states}'b{i}" for i in state_bits]
    else:    
        num_bits = math.ceil(math.log2(num_states))
        if num_bits > 1 and random.random() > 0.5:
            state_bits = [f"{num_bits}'b{convert_to_binary(i, num_bits)}" for i in range(num_states)]
        else:
            state_bits = list(range(num_states))

    all_lines = []
    for i, j in zip(state_names, state_bits):
        line = f"{i}={j}"
        all_lines.append(line)

    all_lines = ', '.join(all_lines)

    return f"parameter {all_lines};"

def print_state_table(g, num_states=4, input_bit=1, output_bit=1):
    """
    This only works for input_bit=1; output_bit=1
    """
    if random.random() > 0.2:
        all_lines = ["// state | Next state in=0, Next state in=1 | Output"]
    else:
        all_lines = ["// state | next state in=0, next state in=1 | output"]
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
        line = f"// {state_names[i]} | {state_names[neg_state]}, {state_names[pos_state]} | {g.nodes[i]['output']}"
        all_lines.append(line)

    return '\n'.join(all_lines)

def print_transition_logic(g, num_states=4, input_signal_name='w'):
    """
    This only works for input_bit=1; 
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
        line = f"\t\t\t{state_names[i]}: next = {input_signal_name} ? {state_names[pos_state]} : {state_names[neg_state]};"
        all_lines.append(line)

    all_lines += ["\t\t\tdefault: next = 'x;\n\t\tendcase\n\tend"]
    return '\n'.join(all_lines)

def print_output_logic(g, num_states=4, output_signal_name='z'):
    """
    This only works for output_bit=1; 
    """
    logic = []
    output_logic_states = []
    state_names = get_state_names_from_graph(g, num_states)
    for i in range(num_states):
        if g.nodes[i]['output'] != 0:
            logic += [f" state == {state_names[i]} "]
            output_logic_states.append( state_names[i])
    return f"\tassign {output_signal_name} = ({'||'.join(logic)});", ", ".join(output_logic_states)
