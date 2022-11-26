import sympy as sp
import itertools
from tqdm import tqdm
from string import ascii_lowercase as alphabet
from common.creator_utils import create_term
import logging

logger = logging.getLogger(__name__)


def create_solving_problems(
    operations_reservoir,
    left_length=3,
    right_length=3,
    num_symbols=2,
    additional_reservoir=None,
    solution_filter=None,
    desc='create',
    **kwargs,
):
    """Creates problems by permutation over all possibilities and using sympy to obtain solutions."""
    used_symbols = tuple(alphabet[:num_symbols]) + ('x',)
    used_numbers = (1, 0)
    reservoir = sp.symbols(used_symbols) + used_numbers
    x = sp.symbols('x')
    used_idents = [
        *operations_reservoir,
        *used_symbols,
        *(str(n) for n in used_numbers),
    ]
    if additional_reservoir is not None:
        reservoir += additional_reservoir
        used_idents += [str(r) for r in additional_reservoir]

    def combis(length):
        symbols_combi = itertools.combinations(reservoir, length)
        operations_combi = itertools.product(operations_reservoir, repeat=length - 1)
        symbols_combi = list(symbols_combi)
        operations_combi = list(operations_combi)
        return (operations_combi, symbols_combi), len(operations_combi) * len(
            symbols_combi
        )

    left_combi, left_size = combis(left_length)
    right_combi, right_size = combis(right_length)
    size = left_size * right_size

    total_combis = itertools.product(*left_combi, *right_combi)

    problems = []
    seen = set()
    for left_operations, left_symbols, right_operations, right_symbols in tqdm(
        total_combis, total=size, smoothing=0.0, leave=False, desc=desc
    ):
        try:
            left_term = create_term(left_operations, left_symbols)
            right_term = create_term(right_operations, right_symbols)
        except ZeroDivisionError:
            continue
        equation = f'{left_term} = {right_term}'
        if equation in seen:
            continue
        seen.add(equation)
        try:
            solution = sp.solve(sp.Eq(left_term, right_term), x)
        except NotImplementedError:
            continue
        if solution_filter is not None:
            if not solution_filter(solution):
                continue
        elif len(solution) == 0:
            continue
        elif len(solution) != 1:
            logger.debug(f'Equation {equation} has multiple solutions: {solution}')
            continue

        solution = solution[0]
        # Sort out trivial problems
        if equation == f'x = {solution}':
            logger.debug(f'Filter out "{equation}" as it has a trivial solution')
            continue
        problems.append(f'{equation} => x = {solution}'.replace('**', '^'))

    return problems, used_idents
