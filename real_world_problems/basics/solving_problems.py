import sympy as sp
import itertools
import functools
import operator
from tqdm import tqdm
from string import ascii_lowercase as alphabet
import logging

logger = logging.getLogger(__name__)


def create_linear(left_length=3, right_length=3, num_symbols=2, **kwargs):
    used_symbols = tuple(alphabet[:num_symbols]) + ('x',)
    used_numbers = (1, 0)
    reservoir = sp.symbols(used_symbols) + used_numbers
    x = sp.symbols('x')
    operations_reservoir = ('-', '+', '*', '/')
    used_idents = [*operations_reservoir, *used_symbols, *(str(n) for n in used_numbers)]

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
                if b == 0:
                    raise ZeroDivisionError()
                a = a / b

        return a

    problems = []
    for left_operations, left_symbols, right_operations, right_symbols in tqdm(total_combis, total=size, smoothing=0.):
        try:
            left_term = create_term(left_operations, left_symbols)
            right_term = create_term(right_operations, right_symbols)
        except ZeroDivisionError:
            continue
        equation = f'{left_term} = {right_term}'
        solution = sp.solve(sp.Eq(left_term, right_term), x)
        if len(solution) == 0:
            continue
        if len(solution) != 1:
            logger.debug(f'Equation {equation} has solutions: {solution}')
            continue

        solution = solution[0]
        problems.append(f'{equation} => x = {solution}'.replace('**', '^'))

    return problems, used_idents
