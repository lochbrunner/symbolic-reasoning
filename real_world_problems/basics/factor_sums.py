import itertools
import sympy as sp
from typing import Tuple


class Problem:
    def __init__(self, combi, signs):
        self.problem = create_sum(combi, signs)**2
        self.solution = sp.expand(self.problem)

    def __str__(self):
        return f'{self.solution} = {self.problem}'.replace('**', '^')


def create_sum(symbols: Tuple[sp.Symbol], signs: Tuple[str]):
    a = symbols[0]
    for b, sign in zip(symbols[1:], signs):
        if sign == '+':
            a = a + b
        else:
            a = a - b
    return a


def create(term_length=2):
    reservoir = sp.symbols(('a', 'b', 'c', 'd')) + (1,)
    reservoir_signs = ('-', '+')

    signs_combi = itertools.product(reservoir_signs, repeat=term_length-1)
    symbol_combi = itertools.combinations(reservoir, term_length)

    return [
        Problem(combi, sign_combi) for combi, sign_combi in itertools.product(symbol_combi, signs_combi)
    ]
