#!/usr/bin/env python3

import argparse
import itertools
import logging
from pathlib import Path
from string import ascii_lowercase as alphabet
from tqdm import tqdm

import sympy as sp

from common.creator_utils import Replacer, collect, create_term
from pycore import Symbol

logger = logging.getLogger(__name__)


def simplify_problems(num_symbols, length, **kwargs):
    operations_reservoir = ('+', '-', '*', '/', '^')
    used_symbols = tuple(alphabet[:num_symbols]) + ('x',)
    used_numbers = (1, 0)
    reservoir = sp.symbols(used_symbols) + used_numbers
    x = sp.symbols('x')
    problems = []
    used_idents = [
        *operations_reservoir,
        *used_symbols,
        *(str(n) for n in used_numbers),
    ]

    def combis(length):
        symbols_combi = itertools.combinations(reservoir, length)
        operations_combi = itertools.product(operations_reservoir, repeat=length - 1)

        symbols_combi = list(symbols_combi)
        operations_combi = list(operations_combi)
        return (operations_combi, symbols_combi), len(operations_combi) * len(
            symbols_combi
        )

    combi, size = combis(length)
    total_combis = itertools.product(*combi)

    for operations, symbols in tqdm(total_combis, total=size, smoothing=0.0, leave=False, desc='create'):

        try:
            term = create_term(operations, symbols)
        except ZeroDivisionError:
            continue

        solution = sp.diff(term, x)

        problems.append(f'D({term}, x) => x = {solution}'.replace('**', '^'))

    return problems, used_idents


def main(args):

    me = Path(__file__).absolute()

    config_path = me.parent / 'config.yaml'
    collect(
        args,
        config_path=config_path,
        factories=[
            simplify_problems,
        ],
        replacer=Replacer("sqrt", "root", Symbol.number(2))
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Problems creator')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--length', default=5, type=int)
    parser.add_argument('--num-symbols', default=3, type=int)
    main(parser.parse_args())
