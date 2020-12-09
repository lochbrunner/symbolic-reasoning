import sympy as sp
import itertools
import functools
import operator
from tqdm import tqdm
from string import ascii_lowercase as alphabet
import logging

logger = logging.getLogger(__name__)


def create_linear(left_length=3, right_length=3, num_symbols=2):
    reservoir = sp.symbols(tuple(alphabet[:num_symbols]) + ('x',)) + (1, 0)
    x = sp.symbols('x')
    operations_reservoir = ('-', '+', '*', '/')

    def combis(length):
        symbols_combi = itertools.combinations(reservoir, length)
        operations_combi = itertools.product(operations_reservoir, repeat=length-1)
        symbols_combi = list(symbols_combi)
        operations_combi = list(operations_combi)
        return (operations_combi, symbols_combi), len(operations_combi)*len(symbols_combi)

    left_combi, left_size = combis(left_length)
    right_combi, right_size = combis(right_length)
    size = left_size*right_size

    total_combis = itertools.product(*left_combi, *right_combi)

    def create_term(operations, symbols):
        a = symbols[0]
        for b, operation in zip(symbols[1:], operations):
            if operation == '+':
                a = a + b
            elif operation == '-':
                a = a - b
            elif operation == '*':
                a = a * b
            elif operation == '/':
                a = a / b

        return a

    problems = []
    for left_operations, left_symbols, right_operations, right_symbols in tqdm(total_combis, total=size, smoothing=0.):
        left_term = create_term(left_operations, left_symbols)
        right_term = create_term(right_operations, right_symbols)
        equation = f'{left_term} = {right_term}'
        solution = sp.solve(sp.Eq(left_term, right_term), x)
        if len(solution) == 0:
            continue
        if len(solution) != 1:
            logger.debug(f'Equation {equation} has solutions: {solution}')
            continue

        solution = solution[0]
        problems.append(f'{equation} => x = {solution}'.replace('**', '^'))

    return problems
