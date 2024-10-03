# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from sympy.logic import SOPform
from sympy import symbols
import random
import numpy as np
import math
import itertools

def generate_symbols(num_variable):
    starting_symbol = ['x', 'a']
    start = random.sample(starting_symbol, 1)[0]
    if num_variable > 3 and start == 'x':
        increment_number = True
    else:
        if random.random() > 0.5:
            increment_number = True
        else:
            increment_number = False

    if random.random() > 0.5:
        start = start.upper()
    if increment_number:
        solution =  [ f'{start}{i}' for i in range(num_variable)  ]
    else:
        solution = [chr(ord(start) + i) for i in range(num_variable) ]
    if random.random() > 0.5:
        return [symbols(x) for x in solution[::-1]]
    return [symbols(x) for x in solution]


def random_sample_minterm(num_variable, sample_no_care=False, max_no_care=4, return_max_terms=False):
    total_terms = 2**num_variable
    max_selected = total_terms - 1 - max_no_care# make it interesting
    all_terms = list(range(total_terms))
    num_min_terms = random.randint(1, max_selected)
    if sample_no_care and num_min_terms > 1:
        num_no_care = random.randint(1, min(max_no_care, num_min_terms-1, total_terms - num_min_terms) ) # let's not sample too much don't care.
    else:
        num_no_care = 0

    selected_terms = random.sample(all_terms, num_min_terms + num_no_care)
    min_terms = random.sample(selected_terms, num_min_terms)
    no_care = [item for item in selected_terms if item not in min_terms]

    if return_max_terms:
        max_terms = [item for item in all_terms if item not in min_terms and item not in no_care]
        return sorted(min_terms), sorted(no_care), sorted(max_terms)
    return sorted(min_terms), sorted(no_care)

def enumerate_min_terms(num_variable):
    numbers = list(range(2**num_variable))

    # Generate all possible combinations of different lengths
    all_combinations = []
    for r in range(1, len(numbers) - 1):
        combinations = list(itertools.combinations(numbers, r))
        all_combinations.extend(combinations)
    return all_combinations


def convert_single_term(term, symbols):
    binary_string = bin(term)[2:]
    bit_sequence = [int(bit) for bit in binary_string]
    if len(bit_sequence) < len(symbols):
        return [0] * (len(symbols)-len(bit_sequence)) + bit_sequence
    return bit_sequence

def format_line(items):
    return " | ".join([str(i) for i in items])

def convert_min_term_sop(min_terms, symbols, style=0):
    def convert_single_binary_term_to_equation(t, x):
        if t == 0:
            if style:
                return ['~' + str(x), ~x]
            else:
                return [str(x) + "'", ~x]
        return [str(x), x]
    
    def convert_binary_term_to_equation(term):
        equation = None
        for t, x, in zip(term, symbols):
            if equation is None:
                equation = convert_single_binary_term_to_equation(t,x)
            else:
                eq_string, eq_eq = convert_single_binary_term_to_equation(t,x)
                if style:
                    equation[0] = equation[0] + ' & ' + eq_string
                else:
                    equation[0] = equation[0] + eq_string
                equation[1] = equation[1] & eq_eq
        return ['(' + equation[0] + ')', equation[1]]
    
    compact_term = None
    for term in min_terms:
        binary_terms = convert_single_term(term, symbols)
        if compact_term is None:
            compact_term = convert_binary_term_to_equation(binary_terms)
        else:
            eq_string, eq_eq = convert_binary_term_to_equation(binary_terms)
            if style:
                compact_term[0] = compact_term[0] + ' | ' + eq_string
            else:
                compact_term[0] = compact_term[0] + ' + ' + eq_string
            compact_term[1] = compact_term[1] | eq_eq
    return compact_term

def convert_max_term_pos(max_terms, symbols, style=0):
    def convert_single_binary_term_to_equation(t, x):
        if t == 0:
            if style:
                return ['~' + str(x), ~x]
            else:
                return [str(x) + "'", ~x]
        return [str(x), x]
    
    def convert_binary_term_to_equation(term):
        equation = None
        for t, x, in zip(term, symbols):
            if equation is None:
                equation = convert_single_binary_term_to_equation(t,x)
            else:
                eq_string, eq_eq = convert_single_binary_term_to_equation(t,x)
                if style:
                    equation[0] = equation[0] + ' | ' + eq_string
                else:
                    equation[0] = equation[0] + ' + ' + eq_string
                equation[1] = equation[1] & eq_eq
        return ['(' + equation[0] + ')', equation[1]]
    
    compact_term = None
    for term in max_terms:
        binary_terms = convert_single_term(term, symbols)
        if compact_term is None:
            compact_term = convert_binary_term_to_equation(binary_terms)
        else:
            eq_string, eq_eq = convert_binary_term_to_equation(binary_terms)
            if style:
                compact_term[0] = compact_term[0] + ' & ' + eq_string
            else:
                compact_term[0] = compact_term[0] +  eq_string
            compact_term[1] = compact_term[1] | eq_eq
    return compact_term


def print_table_minterms(terms, no_care, symbols, no_care_symbol='x', comment=False):
    if comment:
        comment_string = "//"
    else:
        comment_string = ""
    
    all_lines = [ comment_string + " " + format_line(symbols + ['f'])] 

    all_terms = 2**len(symbols)
    for i in range(all_terms):
        if i in terms:
            all_lines.append( comment_string + " " + format_line(convert_single_term(i, symbols) + [1] ) )
        elif i in no_care:
            all_lines.append( comment_string + " " + format_line(convert_single_term(i, symbols) + [no_care_symbol] ) )
        else:
            all_lines.append( comment_string + " " + format_line(convert_single_term(i, symbols) + [0]  ))
    return "\n".join(all_lines)


def generate_gray_code(n):
    if n == 0:
        return ["0"]
    if n == 1:
        return ["0", "1"]

    # Recursive call to generate (n-1)-bit Gray code
    previous_gray_code = generate_gray_code(n - 1)

    # Prefixing with '0' and '1'
    result = []

    # Prefix with '0'
    for code in previous_gray_code:
        result.append("0" + code)

    # Prefix with '1'
    for code in reversed(previous_gray_code):
        result.append("1" + code)

    return result

def permuate_table_to_gray_code(table):
    new_table = np.copy(table)
    row, col = int(math.log2(table.shape[0])), int(math.log2(table.shape[1]))
    row_index = [int(x,2) for x in generate_gray_code(row)]
    col_index = [int(x,2) for x in generate_gray_code(col)]
    for i in range(len(row_index)):
        for j in range(len(col_index)):
            new_table[i][j] = table[row_index[i]][col_index[j]]
    return new_table

def random_permute_gray_code_table(table):
    row, col = int(math.log2(table.shape[0])), int(math.log2(table.shape[1]))
    row_index = [int(x,2) for x in generate_gray_code(row)]
    col_index = [int(x,2) for x in generate_gray_code(col)]
    choice = random.random() 
    if choice < 0.25:
        return table[::-1, :], row_index[::-1], col_index
    elif choice < 0.5:
        return table[:, ::-1], row_index, col_index[::-1]
    elif choice < 0.75:
        a, b = random.sample(range(len(row_index)), 2)
        table[[a,b], :] = table[[b,a], :]
        row_index[a], row_index[b] = row_index[b], row_index[a]
        return table, row_index, col_index
    
    a, b = random.sample(range(len(col_index)), 2)
    table[:, [a,b]] = table[:, [b,a]]
    col_index[a], col_index[b] = col_index[b], col_index[a]
    return table, row_index, col_index

def print_karnaugh_map(terms, no_care, symbols, no_care_symbol='d', permute=False, comment=False):

    num_variables = len(symbols)
    all_terms = 2**len(symbols)
    table = []
    for i in range(all_terms):
        if i in terms:
            table.append('1')
        elif i in no_care:
            table.append(no_care_symbol)
        else:
            table.append('0')
    if comment:
        comment_string = "//"
    else:
        comment_string = ""
    if num_variables % 2 == 0:
        row, col = num_variables // 2, num_variables // 2
    else:
        row, col = num_variables // 2 + 1, num_variables // 2 
    #elif random.random() > 0.5:
    #    row, col = num_variables // 2 + 1, num_variables // 2
    #else:
    #    row, col = num_variables // 2, num_variables // 2 + 1

    ori_row, ori_col = int(row), int(col)
    row, col = 2**row, 2**col
    table = np.array(table).reshape(row, col)
    table = permuate_table_to_gray_code(table)
    if permute:
        table, row_index, col_index = random_permute_gray_code_table(table)
    else:
        row_index = [int(x,2) for x in generate_gray_code(ori_row)]
        col_index = [int(x,2) for x in generate_gray_code(ori_col)]


    row_variable = symbols[:ori_row]
    col_variable = symbols[ori_row:]
    col_index = [bin(i)[2:].zfill(ori_col) for i in col_index]
    row_index = [bin(i)[2:].zfill(ori_row) for i in row_index]
    col_variable = [str(i) for i in col_variable]
    row_variable = [str(i) for i in row_variable]
    all_lines = [  comment_string + ' '* (len(row_variable)+3) + ''.join(col_variable) ]

    col_space, row_space = '', ''
    if len(col_index[0]) < 4:
        col_space = ' ' * (4 - len(col_index[0]))
    if len(col_index[0]) >= 4:
        col_space = ' '
        row_space = ' ' * (len(col_index[0]) - 4 + 1)
    all_lines += [ f'{comment_string} ' + ''.join(row_variable) + f'{col_space}'.join([''] + col_index)  ]
    for i in range(row):
        all_lines += [  f'{comment_string} ' +    f' | {row_space}'.join( [row_index[i]] + list(table[i,:])  )       ]

    return "\n".join(all_lines)

def convert_single_term_local(term, length):
    binary_string = bin(term)[2:]
    bit_sequence = [str(int(bit)) for bit in binary_string]
    if len(bit_sequence) < length:
        return ['0'] * (length-len(bit_sequence)) + bit_sequence
    return bit_sequence
  
def convert_min_terms(min_sterms_string, min_terms, length, add_int_prefix=False):
    if '+' in min_sterms_string:
        term_string = min_sterms_string.split('+')
        if add_int_prefix:
            terms = [  f"{x} => ({''.join(convert_single_term_local(x, length))})" for x in min_terms]
        else:
            terms = [  f"({''.join(convert_single_term_local(x, length))})" for x in min_terms]
    else:
        term_string = min_sterms_string.split('|')
        if add_int_prefix:
            terms = [  f"{x} => ({''.join(convert_single_term_local(x, length))})" for x in min_terms]
        else:
            terms = [  f"({','.join(convert_single_term_local(x, length))})" for x in min_terms]

    converted_string = []
    for a, b in zip(terms, term_string):
        converted_string.append(  f"{a} => {b}"  )
    return "\n".join(converted_string)

def convert_max_terms(min_sterms_string, min_terms, length, add_int_prefix=False):
    if '+' in min_sterms_string:
        term_string = min_sterms_string.split('+')
        if add_int_prefix:
            terms = [  f"{x} => ({''.join(convert_single_term_local(x, length))})" for x in min_terms]
        else:
            terms = [  f"({''.join(convert_single_term_local(x, length))})" for x in min_terms]
    else:
        term_string = min_sterms_string.split('&')
        if add_int_prefix:
            terms = [  f"{x} => ({''.join(convert_single_term_local(x, length))})" for x in min_terms]
        else:
            terms = [  f"({','.join(convert_single_term_local(x, length))})" for x in min_terms]

    converted_string = []
    for a, b in zip(terms, term_string):
        converted_string.append(  f"{a} => {b}"  )
    return "\n".join(converted_string)